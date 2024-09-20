import torchvision.transforms.v2 as tfms
import torchvision.transforms.v2.functional as TF
from sampler import SimpleDiffusion
import torch
from PIL import Image
from augments import TPSWarp
from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
from torchmetrics.multimodal import CLIPImageQualityAssessment
from torchvision.utils import make_grid
from calflops import calculate_flops

# IMG = "../Imagenet/CLS-LOC/test/ILSVRC2012_test_00000063.JPEG"
IMG = "test_imgs/test_im2.jpg"
DEVICE = "cpu"
SIZE = 1200

TRANSFORMS = tfms.Compose([
            tfms.ToImage(), tfms.ToDtype(torch.uint8, scale=True),
            tfms.RandomCrop(SIZE),
            tfms.Resize(512),
            tfms.ToDtype(torch.float32, scale=True),
            ])

def open_img(fname):
    return TRANSFORMS(Image.open(fname))

def flops(checkpoint):
    from diffusion_resshift_unet import DiffuserSRResShift
    MODEL_TYPE = DiffuserSRResShift

    model = MODEL_TYPE.load_from_checkpoint(checkpoint, map_location="cpu")
    model = model.to(DEVICE)
    batch_size = 1
    input_shape = (batch_size, 3, 64, 64)
    x = torch.randn(input_shape, device=DEVICE)
    cond = torch.randn(input_shape, device=DEVICE)
    t = torch.randint(low=1, high=model.sampler.timesteps, size=(batch_size,), device=DEVICE)

    flops, macs, params = calculate_flops(model=model.unet, 
                                        input_shape=None,
                                        args = [x, t, cond],
                                        output_as_string=True,
                                        output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

def ddpm_forward(img):
    img = img*2-1
    eps = torch.randn_like(img)

    sampler = SimpleDiffusion()

    results = [img]
    results.append(-torch.ones(4, SIZE, 140))
    for ti in [180, 250, 360, 540, 700]:
        f_img = sampler.forward(img, ti*t, eps)[0]
        results.append(torch.cat([f_img, torch.ones(1, SIZE, SIZE)], dim=0))
        results.append(-torch.ones(4, SIZE, 140))

    results[0] = torch.cat([results[0][0], torch.ones(1, SIZE, SIZE)], dim=0)
    results.pop()
    R = torch.cat(results, dim=-1).clamp(-1,1)*0.5+0.5
    TF.to_pil_image(R, "RGBA").save("forward.png")

def metrics(img):
    psnr = PeakSignalNoiseRatio(data_range=(-1,1))
    ssim = StructuralSimilarityIndexMeasure(data_range=(-1,1))
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
    lpips.eval()
    clipiqa = CLIPImageQualityAssessment()
    clipiqa.anchors = clipiqa.anchors.clone()
    clipiqa.eval()

    aguments = [tfms.ElasticTransform(alpha=250.0), TPSWarp(),
                tfms.GaussianNoise(sigma=0.4),
                tfms.GaussianBlur(kernel_size=(11,11), sigma=10), 
                tfms.RandomEqualize(p=1), 
                # tfms.ColorJitter(brightness=.5, hue=(-0.3, -0.3),)
                ]

    results = [img[0]]

    norm_img = img*2-1
    print("Stats:", (psnr(norm_img, norm_img).item(), ssim(norm_img, norm_img).item(),
            lpips(norm_img, norm_img).item(), clipiqa(img).mean().item()))

    for a in aguments:
        augmented_img = a(img).clamp(0,1)
        norm_augmented = augmented_img*2-1
        results.append(augmented_img[0])
        print("Stats: ",(psnr(norm_augmented, norm_img).item(), ssim(norm_augmented, norm_img).item(),
            lpips(norm_augmented, norm_img).item(), clipiqa(augmented_img).mean().item()))

    result_img = make_grid(torch.stack(results), nrow=3, padding=0)
    TF.to_pil_image(result_img).save("metrics.png")


if __name__ == "__main__":
    t = torch.ones(size=(1,), device=DEVICE, dtype=torch.long)
    img = open_img(IMG).to(DEVICE)[None]