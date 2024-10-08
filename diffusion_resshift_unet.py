import gc, math, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, swa_utils
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only
from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
from torchmetrics.multimodal import CLIPImageQualityAssessment
from diffusers import UNet2DConditionModel, DDPMScheduler, get_cosine_schedule_with_warmup, AutoencoderKL, VQModel
from data import SimpleImageDataModule
from layers import ResBlock, DownSample, AttentionBlock, NACBlock, Rearrange
from models import UNetConditionalCat, UnetConditionalCompact
from sampler import ResShiftDiffusion, ResShiftDiffusionEps

class NoVAE(nn.Module):
    def __init__(self):
        super().__init__()
    def encode_img(self, x):
        return x
    def decode(self, x, return_dict=False):
        return (x,)

class DiffuserSRResShift(L.LightningModule):
    def __init__(self,
                 pred_x0=True,
                 base_channels=128,
                 base_channels_multiples=(1,2,2,4),
                 apply_attention = (True, True, True, True),
                 n_layers=1, dropout_rate=0.0, scale_factor=4,
                 image_size=256,
                 cross_attention_dim=768,
                lr=1e-4, scheduler_type="one_cycle", warmup_steps=500,
                vae_chkp="madebyollin/sdxl-vae-fp16-fix", vae_type="vae",
                use_scale_shift_norm=False,
                compact_model=True,
                n_heads=1,
                pixel_shuffle=True,
                use_cross_attn=False,
                max_vit_attn=True,
                window_size=8,
                timesteps=15, resshift_p=0.3, kappa=2.0):
        super().__init__()
        self.save_hyperparameters()
        self.scale_factor = scale_factor
        self.pred_x0 = pred_x0
        if pred_x0:
            self.sampler = ResShiftDiffusion(timesteps, resshift_p, kappa)
        else:
            self.sampler = ResShiftDiffusionEps(timesteps, resshift_p, kappa)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.scheduler_type = scheduler_type
        if vae_type=="vae":
            self.vae = AutoencoderKL.from_pretrained(vae_chkp)
            AutoencoderKL.encode_img = lambda vae, x: AutoencoderKL.encode(vae, x, return_dict=False)[0].mode()
            self.vae_compresion = 8
            in_chs = 4
        elif vae_type=="vqvae":
            self.vae = VQModel.from_pretrained(vae_chkp)
            VQModel.encode_img = lambda vae, x: VQModel.encode(vae, x, return_dict=False)[0]
            self.vae_compresion = 4
            in_chs = 3
        elif vae_type=="no_vae":
            self.vae = NoVAE()
            self.vae_compresion = 1
            in_chs = 3
        else:
            raise ValueError(f"{vae_type} is invalid")
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad_(False)
        
        cond_channels=3
        
        MODEL = UnetConditionalCompact if compact_model else UNetConditionalCat

        self.unet = MODEL(timesteps, input_channels=in_chs,
                                       cond_channels=cond_channels, output_channels=in_chs,
                                        base_channels=base_channels,
                                        max_vit_attn=max_vit_attn,
                                        num_res_blocks=n_layers, use_scale_shift_norm=use_scale_shift_norm,
                                        n_heads=n_heads, window_size=window_size,
                                        base_channels_multiples=base_channels_multiples,
                                        apply_attention=apply_attention, dropout_rate=dropout_rate,
                                        pixel_shuffle=pixel_shuffle)
        
        # self.ema = swa_utils.AveragedModel(self.unet,
        #                                    multi_avg_fn=swa_utils.get_ema_multi_avg_fn(0.999))

        self.psnr = PeakSignalNoiseRatio(data_range=(-1,1))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(-1,1))
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
        self.lpips.eval()
        self.clipiqa = CLIPImageQualityAssessment()
        self.clipiqa.anchors = self.clipiqa.anchors.clone()
        self.clipiqa.eval()

        self.log_batch_low_res = torch.zeros(0, 3, image_size, image_size)
        self.log_batch_high_res = torch.zeros(0, 3, image_size, image_size)
        self.log_batch_cond = torch.zeros(0, 3, image_size//self.vae_compresion, image_size//self.vae_compresion)
    
    # def on_before_zero_grad(self, *args, **kwargs):
    #     self.ema.update_parameters(self.unet)

    def training_step(self, batch, batch_idx):

        if (self.trainer.is_global_zero and
            self.log_batch_high_res.shape[0] >= 16 and
            (self.global_step % self.trainer.log_every_n_steps) == 0):
            self.do_log()

        high_res = batch
        low_res = F.interpolate(high_res,
                                scale_factor=1./self.scale_factor, mode='bicubic', antialias=True)
        cond = F.interpolate(low_res, scale_factor=float(self.scale_factor)/self.vae_compresion, mode='bicubic', antialias=True)
        up_low_res = F.interpolate(low_res, scale_factor=self.scale_factor,
                                   mode='bicubic', antialias=True).clamp(-1,1)
            
        encoded_high_res = self.vae.encode_img(high_res)
        encoded_low_res = self.vae.encode_img(up_low_res)

        t = torch.randint(low=1, high=self.sampler.timesteps, size=(high_res.shape[0],), device=high_res.device)
        noise_gt = torch.randn_like(encoded_high_res)
        xnoise = self.sampler.add_noise(encoded_high_res, encoded_low_res, noise_gt, t)

        pred = self.unet(xnoise, t, cond)
        ground_truth = encoded_high_res if self.pred_x0 else noise_gt
        loss = F.mse_loss(pred, ground_truth)
        self.log("train_loss", loss)

        if self.trainer.is_global_zero and self.log_batch_low_res.shape[0] < 16:
            self.log_batch_high_res = self.log_batch_high_res.to(high_res)
            self.log_batch_high_res = torch.cat([self.log_batch_high_res, high_res.detach()])
            self.log_batch_high_res.requires_grad_(False)

            self.log_batch_low_res = self.log_batch_low_res.to(up_low_res)
            self.log_batch_low_res = torch.cat([self.log_batch_low_res, up_low_res.detach()])
            self.log_batch_low_res.requires_grad_(False)
            
            self.log_batch_cond = self.log_batch_cond.to(cond)
            self.log_batch_cond = torch.cat([self.log_batch_cond, cond.detach()])
            self.log_batch_cond.requires_grad_(False)

            if self.log_batch_high_res.shape[0] >= 16:
                self.logger.experiment.add_image("high_res", 
                                                make_grid(self.log_batch_high_res*0.5+0.5, nrow=4),self.global_step)
                self.logger.experiment.add_image("bicubic", 
                                                make_grid(self.log_batch_low_res*0.5+0.5, nrow=4),self.global_step)

        return loss

    @rank_zero_only
    @torch.no_grad()
    def do_log(self):
        encoded_low_res = self.vae.encode_img(self.log_batch_low_res)
        xnoise = self.sampler.prior_sample(encoded_low_res,
                                           torch.randn_like(encoded_low_res))

        self.unet.eval()
        pred = self.predict(xnoise, self.log_batch_cond, encoded_low_res)
        self.unet.train()
        
        pred = self.vae.decode(pred, return_dict=False)[0].clamp(-1,1)
        pred = torch.nan_to_num(pred)
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image("pred",
                                            make_grid((pred*0.5+0.5).cpu(), nrow=4),
                                            self.global_step)
        gc.collect()
        torch.cuda.empty_cache()

        psnr = self.psnr(pred, self.log_batch_high_res).item()
        ssim = self.ssim(pred, self.log_batch_high_res).item()
        lpips = self.lpips(pred, self.log_batch_high_res).item()
        self.log_dict({'train_psnr':psnr, 'train_ssim': ssim, 'train_lpips':lpips},
                      rank_zero_only=True)
        gc.collect()

    @torch.no_grad()
    def predict(self, xnoise:torch.Tensor, cond:torch.Tensor, encoded_low_res:torch.Tensor):
        for time_step in reversed(range(self.sampler.timesteps)):
            ts = torch.ones(xnoise.shape[0], dtype=torch.long, device=xnoise.device) * time_step
            pred = self.unet(xnoise, ts, cond)
            if self.pred_x0:
                xnoise = self.sampler.backward_step(xnoise, pred, ts)
            else:
                xnoise = self.sampler.backward_step(xnoise, encoded_low_res, pred, ts)
        return xnoise
        
    def configure_optimizers(self):
        params = self.unet.parameters()
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        if self.scheduler_type is None:
            return [optimizer]
        if self.scheduler_type == "cosine_warmup":
            scheduler = get_cosine_schedule_with_warmup(optimizer, self.warmup_steps,
                                                        num_training_steps=self.trainer.estimated_stepping_batches)
        if self.scheduler_type == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
                                                            total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            high_res = batch
            low_res = F.interpolate(high_res,
                                    scale_factor=1./self.scale_factor, mode='bicubic', antialias=True)
            cond = F.interpolate(low_res, scale_factor=float(self.scale_factor)/self.vae_compresion,
                                 mode='bicubic', antialias=True)
            up_low_res = F.interpolate(low_res, scale_factor=self.scale_factor,
                                       mode='bicubic', antialias=True).clamp(-1,1)
            encoded_low_res = self.vae.encode_img(up_low_res)
            xnoise = self.sampler.prior_sample(encoded_low_res, torch.randn_like(encoded_low_res))

            pred = self.predict(xnoise, cond, encoded_low_res)
            pred = self.vae.decode(pred, return_dict=False)[0].clamp(-1,1)
            
            psnr = self.psnr(pred, high_res).item()
            ssim = self.ssim(pred, high_res).item()
            lpips = self.lpips(pred, high_res).item()
            clipiqa = self.clipiqa(pred*0.5+0.5).mean().item()
            self.log_dict({'valid_psnr':psnr, 'hp_metric':psnr,
                        'valid_ssim':ssim, 'valid_lpips':lpips, 'valid_clipiqa':clipiqa},
                        on_epoch=True
                        )
            gc.collect()
    
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx):
        with torch.inference_mode():
            low_res = batch
            cond = F.interpolate(low_res, scale_factor=self.scale_factor/8.0, mode='bicubic', antialias=True)
            up_low_res = F.interpolate(low_res, scale_factor=self.scale_factor,
                                       mode='bicubic', antialias=True).clamp(-1,1)
            encoded_low_res = self.vae.encode_img(up_low_res, return_dict=False)[0].mode()
            xnoise = self.sampler.prior_sample(encoded_low_res, torch.randn_like(encoded_low_res))
            pred = self.predict(xnoise, cond)
            pred = self.vae.decode(pred, return_dict=False)[0].clamp(-1, 1)
            return pred
    
def main():
    checkpoint_callback = ModelCheckpoint(save_top_k=10, every_n_train_steps=20_000, monitor="hp_metric")
    trainer_defaults = dict(enable_checkpointing=True, callbacks=[checkpoint_callback],
                            enable_progress_bar=False, log_every_n_steps=5_000)
    
    cli = LightningCLI(model_class=DiffuserSRResShift,
                       trainer_defaults=trainer_defaults)
if __name__ == "__main__":
    main()