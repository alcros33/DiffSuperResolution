import gc, math, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
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
from data import MultiImageDataModule
from layers import ResBlock, DownSample, AttentionBlock, NACBlock, Rearrange, UpSample
from models import UNetConditionalCrossAttn, UNetConditionalCat
from sampler import ResShiftDiffusion

class UnetRefiner(L.LightningModule):
    def __init__(self, input_chs=3, output_chs=3,
                 base_channels=64,
                 base_channels_multiples=(1,2),
                 apply_attention = (False, False),
                 n_layers=1, dropout_rate=0.0,
                lr=1e-4, scheduler_type="one_cycle",
                scale_factor=4,
                loss_w=[1,0,0,0],
                n_heads=4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.scale_factor = scale_factor
        encoder_layers = [nn.Conv2d(input_chs, base_channels, 3, 1, 1)]
        c_in = base_channels
        for it, (mult, attn) in enumerate(zip(base_channels_multiples, apply_attention)):
            ch = base_channels*mult
            block = []
            if it != 0:
                block.append(DownSample(c_in))
            for n in range(n_layers):
                block.append(ResBlock(c_in, ch, dropout=dropout_rate))
                c_in = ch
            encoder_layers.append(nn.Sequential(*block))
        self.encoder = nn.ModuleList(encoder_layers)

        self.middle_block = nn.Sequential(
            DownSample(c_in),
            ResBlock(c_in, c_in, dropout=dropout_rate),
            ResBlock(c_in, ch, dropout=dropout_rate),
            UpSample(c_in)
        )

        decoder_layers = []
        n_blocks = len(base_channels_multiples)
        for it, mult in enumerate(reversed([1]+base_channels_multiples[:-1])):
            ch = base_channels*mult
            block = [NACBlock(c_in*2, c_in, nn.SiLU())]
            for n in range(n_layers):
                block.append(ResBlock(c_in, ch, dropout=dropout_rate))
                c_in = ch
            if it != n_blocks-1:
                block.append(UpSample(c_in))
            decoder_layers.append(nn.Sequential(*block))
        decoder_layers.append(NACBlock(c_in*2, output_chs, nn.SiLU()))
        self.decoder = nn.ModuleList(decoder_layers)

        self.metrics = {
            'psnr' : PeakSignalNoiseRatio(data_range=(-1,1)),
            'lpips' : LearnedPerceptualImagePatchSimilarity(net_type="vgg"),
            "ssim":  StructuralSimilarityIndexMeasure(data_range=(-1,1)),
        }
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
        self.loss_w = loss_w
    
    def loss_fn(self, x, xhat):
        return (F.mse_loss(x,xhat)*self.loss_w[0] + F.l1_loss(x,xhat)*self.loss_w[1]
                + F.smooth_l1_loss(x,xhat)*self.loss_w[2] + self.lpips(x, xhat.clamp(-1,1))*self.loss_w[3])
    
    def forward(self, x):
        skip = []
        h = x
        for l in self.encoder:
            h = l(h)
            skip.append(h)

        h = self.middle_block(h)

        for l in self.decoder:
            outs = skip.pop()
            input = torch.cat([h, outs], dim=1)
            h = l(input)
        
        return h

    def training_step(self, batch, batch_idx):
        up_low_res = F.interpolate(batch[0], scale_factor=self.scale_factor, mode='bicubic', antialias=True)
        if len(batch) == 3:
            low_res, rough_pred, high_res = batch
            input = torch.cat([up_low_res, rough_pred], dim=1)
        else:
            low_res, rough_pred, visible, high_res = batch
            input = torch.cat([up_low_res, rough_pred, visible], dim=1)
        
        pred = self.forward(input)
        loss = self.loss_fn(high_res, pred)
        self.log("train_loss", loss)

        if self.trainer.is_global_zero and getattr(self, "log_batch_cond", None) is None:
            self.logger.experiment.add_image("high_res", 
                                            make_grid(high_res*0.5+0.5, nrow=4),self.global_step)
            self.logger.experiment.add_image("bicubic", 
                                            make_grid(up_low_res.clamp(-1,1)*0.5+0.5, nrow=4),self.global_step)
            self.logger.experiment.add_image("visible", 
                                            make_grid(visible*0.5+0.5, nrow=4),self.global_step)
            
            self.log_batch_cond = input.clone().detach()
            self.log_batch_cond.requires_grad_(False)

        if (self.trainer.is_global_zero and
            (self.global_step % self.trainer.log_every_n_steps) == 0):
            self.do_log()
        return loss

    @rank_zero_only
    @torch.inference_mode()
    def do_log(self):
        self.encoder.eval(); self.middle_block.eval(); self.decoder.eval();
        pred = self.forward(self.log_batch_cond).clamp(-1,1)
        self.encoder.train(); self.middle_block.train(); self.decoder.train();

        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image("pred",
                                            make_grid((pred*0.5+0.5).cpu(), nrow=4),
                                            self.global_step)
        gc.collect()
        torch.cuda.empty_cache()
        
    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.middle_block.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
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
        with torch.inference_mode():
            up_low_res = F.interpolate(batch[0], scale_factor=self.scale_factor, mode='bicubic', antialias=True)
            if len(batch) == 3:
                low_res, rough_pred, high_res = batch
                input = torch.cat([up_low_res, rough_pred], dim=1)
            else:
                low_res, rough_pred, visible, high_res = batch
                input = torch.cat([up_low_res, rough_pred, visible], dim=1)
            pred = self.forward(input)
            loss = self.loss_fn(high_res, pred)
            pred = pred.clamp(-1, 1)

            for k in self.metrics.keys():
                self.metrics[k].to(pred.device)
            
            psnr = self.metrics['psnr'](pred, high_res)
            ssim = self.metrics['ssim'](pred, high_res)
            lpips = self.metrics['lpips'](pred, high_res)
            self.log_dict({'valid_psnr':psnr, 'hp_metric':psnr,
                  'valid_ssim':ssim, 'valid_lpips':lpips, 'valid_loss':loss}, sync_dist=True, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        with torch.inference_mode():
            up_low_res = F.interpolate(batch[0], scale_factor=self.scale_factor, mode='bicubic', antialias=True)
            if len(batch) == 2:
                low_res, rough_pred = batch
                input = torch.cat([up_low_res, rough_pred], dim=1)
            else:
                low_res, rough_pred, visible = batch
                input = torch.cat([up_low_res, rough_pred, visible], dim=1)
            pred = self.forward(input).clamp(-1, 1)
            return pred

def main():
    checkpoint_callback = ModelCheckpoint(save_top_k=5, every_n_epochs=5, monitor="hp_metric")
    trainer_defaults = dict(enable_checkpointing=True, callbacks=[checkpoint_callback],
                            enable_progress_bar=False, log_every_n_steps=5_000)
    
    cli = LightningCLI(model_class=UnetRefiner,
                       datamodule_class=MultiImageDataModule,
                       trainer_defaults=trainer_defaults)
if __name__ == "__main__":
    main()