import gc, math, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import lightning as L
from lightning.pytorch.cli import LightningCLI
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPImageQualityAssessment
from diffusers import AutoencoderKL, VQModel
from data import SimpleImageDataModule

class VAETester(L.LightningModule):
    def __init__(self, use_quantization=False,
                vae_chkp="madebyollin/sdxl-vae-fp16-fix"):
        super().__init__()
        self.save_hyperparameters()
        self.use_quantization = use_quantization
        self.logged = False
        if use_quantization:
            self.vae = VQModel.from_pretrained(vae_chkp)
        else:
            self.vae = AutoencoderKL.from_pretrained(vae_chkp)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad_(False)

        self.psnr = PeakSignalNoiseRatio(data_range=(-1,1))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(-1,1))
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
        self.clipiqa = CLIPImageQualityAssessment()

    def validation_step(self, batch, batch_idx):
        with torch.inference_mode():
            encoded = self.vae.encode(batch, return_dict=False)[0]
            if not self.use_quantization:
                encoded = encoded.mode()
            pred = self.vae.decode(encoded, return_dict=False)[0].clamp(-1, 1)

            low_res = F.interpolate(batch, scale_factor=1./8, mode='bicubic', antialias=True)
            up_low_res = F.interpolate(low_res, scale_factor=8, mode='bicubic', antialias=True).clamp(-1, 1)
            encoded_lr = self.vae.encode(up_low_res, return_dict=False)[0]
            if not self.use_quantization:
                encoded_lr = encoded_lr.mode()
            pred_lr = self.vae.decode(encoded_lr, return_dict=False)[0].clamp(-1, 1)

            ssim_hr = self.ssim(pred, batch)
            psnr_hr = self.psnr(pred, batch)
            lpips_hr = self.lpips(pred, batch)
            clipiqa_hr = self.clipiqa(pred*0.5+0.5).mean()
            ssim_lr = self.ssim(pred_lr, up_low_res)
            psnr_lr = self.psnr(pred_lr, up_low_res)
            lpips_lr = self.lpips(pred_lr, up_low_res)
            clipiqa_lr = self.clipiqa(pred_lr*0.5+0.5).mean()
            self.log_dict({'psnr_hr':psnr_hr, 'ssim_hr':ssim_hr, 'lpips_hr':lpips_hr, 'clipiqa_hr': clipiqa_hr,
                           'psnr_lr':psnr_lr, 'ssim_lr':ssim_lr, 'lpips_lr':lpips_lr, 'clipiqa_lr': clipiqa_lr,
                           }, sync_dist=True, on_epoch=True)

        if not self.logged:
            self.logger.experiment.add_image("high_res", 
                                             make_grid(batch*0.5+0.5, nrow=4), 0)
            self.logger.experiment.add_image("bicubic", 
                                             make_grid(up_low_res*0.5+0.5, nrow=4), 0)
            self.logger.experiment.add_image("pred", 
                                             make_grid(pred.clamp(-1,1)*0.5+0.5, nrow=4), 0)
            self.logger.experiment.add_image("pred_lr", 
                                             make_grid(pred_lr.clamp(-1,1)*0.5+0.5, nrow=4), 0)
            self.logged = True
def main():
    trainer_defaults = dict(enable_checkpointing=False, enable_progress_bar=False,)
    
    cli = LightningCLI(model_class=VAETester,
                       datamodule_class=SimpleImageDataModule,
                       trainer_defaults=trainer_defaults)
if __name__ == "__main__":
    main()