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
from pixart import PixArtImgEmbCrossAttn, PixArtCat
from sampler import ResShiftDiffusion, ResShiftDiffusionEps

class NoVAE(nn.Module):
    def __init__(self):
        super().__init__()
    def encode_img(self, x):
        return x
    def decode_img(self, x):
        return x

class DiffuserSRResShift(L.LightningModule):
    def __init__(self,
                 input_size=32, patch_size=4,
                 hidden_size=512, n_layers=8, scale_factor=4,
                 mlp_ratio=4.0, class_dropout_prob=0.1, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, lewei_scale=1.0,
                lr=1e-4, scheduler_type="one_cycle", warmup_steps=500, img_emb_patch_size=4,
                vae_chkp="madebyollin/sdxl-vae-fp16-fix", vae_type="vae", pred_x0=True,
                n_heads=32, image_size=256,
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
            AutoencoderKL.encode_img = lambda vae, x: AutoencoderKL.encode(vae, x, return_dict=False)[0].mode()*vae.config.scaling_factor
            AutoencoderKL.decode_img = lambda vae, x: AutoencoderKL.decode(vae, x/vae.config.scaling_factor, return_dict=False)[0]
            self.vae_compresion = 8
            in_chs = 4
        elif vae_type=="vqvae":
            self.vae = VQModel.from_pretrained(vae_chkp)
            VQModel.encode_img = lambda vae, x: VQModel.encode(vae, x, return_dict=False)[0]
            VQModel.decode_img = lambda vae, x: VQModel.decode(vae, x, return_dict=False)[0]
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
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.scheduler_type = scheduler_type
        
        # self.denoiser = PixArtImgEmbCrossAttn(input_size=input_size, patch_size=patch_size, in_channels=in_chs,
        #                                       hidden_size=hidden_size, depth=n_layers, num_heads=n_heads,
        #                                       pred_sigma=False, mlp_ratio=mlp_ratio, class_dropout_prob=class_dropout_prob, drop_path = drop_path, window_size=window_size, window_block_indexes=window_block_indexes, use_rel_pos=use_rel_pos, cond_channels=in_chs, cond_img_size=image_size // self.vae_compresion, img_emb_patch_size=img_emb_patch_size, lewei_scale=lewei_scale)

        self.denoiser = PixArtCat(input_size=input_size, patch_size=patch_size, in_channels=in_chs+3,
                                            out_channels=in_chs,
                                              hidden_size=hidden_size, depth=n_layers, num_heads=n_heads,
                                              pred_sigma=False, mlp_ratio=mlp_ratio, class_dropout_prob=class_dropout_prob, drop_path = drop_path, window_size=window_size, window_block_indexes=window_block_indexes, use_rel_pos=use_rel_pos, img_emb_patch_size=img_emb_patch_size, lewei_scale=lewei_scale)
        
        # self.ema = swa_utils.AveragedModel(self.denoiser,
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
        self.log_batch_cond = torch.zeros(0, 3, image_size//self.vae_compresion, image_size//self.vae_compresion) #TODO
    
    # def on_before_zero_grad(self, *args, **kwargs):
    #     self.ema.update_parameters(self.denoiser)

    def training_step(self, batch, batch_idx):

        if (self.trainer.is_global_zero and
            self.log_batch_high_res.shape[0] >= 16 and
            (self.global_step % self.trainer.log_every_n_steps) == 0):
            self.do_log()

        high_res = batch
        low_res = F.interpolate(high_res,
                                scale_factor=1./self.scale_factor, mode='bicubic', antialias=True)
        up_low_res = F.interpolate(low_res, scale_factor=self.scale_factor,
                                   mode='bicubic', antialias=True).clamp(-1,1)
            
        encoded_high_res = self.vae.encode_img(high_res)
        encoded_low_res = self.vae.encode_img(up_low_res)
        # cond = encoded_low_res # TODO
        cond = F.interpolate(low_res, scale_factor=float(self.scale_factor)/self.vae_compresion, mode='bicubic', antialias=True)

        t = torch.randint(low=1, high=self.sampler.timesteps, size=(high_res.shape[0],), device=high_res.device)
        noise_gt = torch.randn_like(encoded_high_res)
        xnoise = self.sampler.add_noise(encoded_high_res, encoded_low_res, noise_gt, t)

        pred = self.denoiser(xnoise, t, cond)
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

        self.denoiser.eval()
        pred = self.predict(xnoise, self.log_batch_cond, encoded_low_res)
        self.denoiser.train()
        
        pred = self.vae.decode_img(pred).clamp(-1,1)
        pred = torch.nan_to_num(pred)
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image("pred",
                                            make_grid((pred*0.5+0.5).cpu(), nrow=4),
                                            self.global_step)
        gc.collect()
        torch.cuda.empty_cache()

        psnr = self.psnr(pred, self.log_batch_high_res).item()
        ssim = self.ssim(pred, self.log_batch_high_res).item()
        self.log_dict({'train_psnr':psnr, 'train_ssim': ssim},
                      rank_zero_only=True)
        gc.collect()

    @torch.no_grad()
    def predict(self, xnoise:torch.Tensor, cond:torch.Tensor, encoded_low_res:torch.Tensor):
        for time_step in reversed(range(self.sampler.timesteps)):
            ts = torch.ones(xnoise.shape[0], dtype=torch.long, device=xnoise.device) * time_step
            pred = self.denoiser(xnoise, ts, cond)
            if self.pred_x0:
                xnoise = self.sampler.backward_step(xnoise, pred, ts)
            else:
                xnoise = self.sampler.backward_step(xnoise, encoded_low_res, pred, ts)
        return xnoise
        
    def configure_optimizers(self):
        params = self.denoiser.parameters()
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
            up_low_res = F.interpolate(low_res, scale_factor=self.scale_factor,
                                       mode='bicubic', antialias=True).clamp(-1,1)
            encoded_low_res = self.vae.encode_img(up_low_res)
            xnoise = self.sampler.prior_sample(encoded_low_res, torch.randn_like(encoded_low_res))
            # cond = encoded_low_res # TODO
            cond = F.interpolate(low_res, scale_factor=float(self.scale_factor)/self.vae_compresion, mode='bicubic', antialias=True)

            pred = self.predict(xnoise, cond, encoded_low_res)
            pred = self.vae.decode_img(pred).clamp(-1,1)
            
            psnr = self.psnr(pred, high_res).item()
            ssim = self.ssim(pred, high_res).item()
            lpips = self.lpips(pred, high_res).item()
            clipiqa = self.clipiqa(pred*0.5+0.5).mean().item()
            self.log_dict({'valid_psnr':psnr, 'hp_metric':psnr,
                        'valid_ssim':ssim, 'valid_lpips':lpips, 'valid_clipiqa':clipiqa},
                        on_epoch=True)
            gc.collect()
    
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx):
        with torch.inference_mode():
            low_res = batch
            up_low_res = F.interpolate(low_res, scale_factor=self.scale_factor,
                                       mode='bicubic', antialias=True).clamp(-1,1)
            encoded_low_res = self.vae.encode_img(up_low_res, return_dict=False)[0].mode()
            # cond = encoded_low_res # TODO
            cond = F.interpolate(low_res, scale_factor=float(self.scale_factor)/self.vae_compresion, mode='bicubic', antialias=True)
            xnoise = self.sampler.prior_sample(encoded_low_res, torch.randn_like(encoded_low_res))
            pred = self.predict(xnoise, cond, encoded_low_res)
            pred = self.vae.decode_img(pred).clamp(-1, 1)
            return pred
    
def main():
    checkpoint_callback = ModelCheckpoint(save_top_k=10, every_n_train_steps=20_000, monitor="hp_metric")
    trainer_defaults = dict(enable_checkpointing=True, callbacks=[checkpoint_callback],
                            enable_progress_bar=False, log_every_n_steps=5_000)
    
    cli = LightningCLI(model_class=DiffuserSRResShift,
                       trainer_defaults=trainer_defaults,
                       save_config_kwargs={"overwrite": True})
if __name__ == "__main__":
    main()