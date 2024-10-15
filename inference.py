import sys
import argparse
from enum import Enum
from pathlib import Path
from PIL import Image
from time import perf_counter
import torch
import torchvision.transforms.v2 as tfms
import torchvision.transforms.v2.functional as TF
import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat, einsum
from torchvision.utils import make_grid
import torch.nn.functional as F
from diffusion_resshift_pixart import DiffuserSRResShift as ModelTransformer
from diffusion_resshift_unet import DiffuserSRResShift as ModelTUNet
from utils import chunks
from calflops import calculate_flops

def split_windows_padded_overlap(img:torch.Tensor, w_size, overlap=5):

    def get_padding(dim, w_size, overlap):
        n = (dim-w_size)//(w_size-overlap) + bool((dim-w_size)%(w_size-overlap))
        return (w_size + n*(w_size-overlap)) - dim
    
    orig_size = (img.shape[1], img.shape[2])
    pad_h = get_padding(orig_size[0], w_size, overlap)
    pad_w = get_padding(orig_size[1], w_size, overlap)
    padded_img = F.pad(img,(0, pad_w, 0, pad_h), mode='constant', value=-1)

    splitted_img = padded_img.unfold(1, w_size, w_size-overlap)
    splitted_img = splitted_img.unfold(2, w_size, w_size-overlap)
    splitted_img = rearrange(splitted_img, "c nh nw h w -> (nh nw) c h w", h=w_size, w=w_size)
    return splitted_img, orig_size, (padded_img.shape[1], padded_img.shape[2])

def merge_windows_padded_overlap(splitted_img:torch.Tensor, orig_size, padded_size, w_size, overlap):
    img = torch.zeros(3, padded_size[0], padded_size[1],
                      device=splitted_img.device, dtype=splitted_img.dtype)
    img_weights = torch.zeros(3, padded_size[0], padded_size[1],
                      device=splitted_img.device, dtype=splitted_img.dtype)
    
    stride = w_size - overlap
    n_rows = ((padded_size[0]-w_size) // stride) + 1
    n_cols = splitted_img.shape[0] // n_rows
    for i in range(n_rows):
        for j in range(n_cols):
            img[:, i*stride:i*stride+w_size, j*stride:j*stride+w_size] += splitted_img[i*n_cols + j]
            img_weights[:, i*stride:i*stride+w_size, j*stride:j*stride+w_size] += torch.ones_like(splitted_img[i*n_cols + j])

    return (img / img_weights)[:, :orig_size[0], :orig_size[1]]

def uncollate_fn(batch, sizes):
    last_size = 0
    result = []
    for it in range(len(sizes)):
        result.append(batch[last_size:last_size + sizes[it]])
        last_size += sizes[it]
    return result

TRANSFORMS = tfms.Compose([
            tfms.ToImage(), tfms.ToDtype(torch.uint8, scale=True),
            tfms.ToDtype(torch.float32, scale=True),
            tfms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
            ])

class ModelType(Enum):
    unet = 'unet'
    transformer = 'transformer'

    def __str__(self):
        return self.value

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint')
parser.add_argument('type', type=ModelType, choices=list(ModelType))
parser.add_argument('--windowed', action='store_true')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('-i', '--input', type=Path)
parser.add_argument('-o', '--output-dir', default="results")
parser.add_argument('--bs', type=int, default=32)


def inference_splitted_windows(model, imgs, window_size, overlap, bs):
    batch, padded_size, orig_size, n_windows = [],[],[],[]
    for img in imgs:
        b, o, p = split_windows_padded_overlap(img, window_size, overlap)
        n_windows.append(b.shape[0])
        batch.append(b); padded_size.append(p); orig_size.append(o)
    batch = torch.cat(batch)

    batch = F.interpolate(batch,#+ 0.1*torch.randn_like(img[None]),
                        scale_factor=1./model.scale_factor,
                        mode='bicubic', antialias=True).clamp(-1,1)
    preds = []
    for it in range(0, len(batch), bs):
        preds.append(model.predict_step(batch[it:it+bs], None))

    preds = uncollate_fn(torch.cat(preds), n_windows)

    result = []
    for p,o,pad in zip(preds, orig_size, padded_size):
        result.append(merge_windows_padded_overlap(p, o, pad, window_size, overlap))
    return result

def main():
    args = parser.parse_args()
    if args.type == ModelType.unet:
        MTYPE = ModelTUNet
    elif args.type == ModelType.transformer:
        MTYPE = ModelTransformer
    else:
        raise ValueError("Type must be one of", list(ModelType))
    DEVICE = 'cpu'
    if args.cuda:
        DEVICE = 'cuda:0'
    model = MTYPE.load_from_checkpoint(args.checkpoint, map_location=DEVICE)
    model.eval()

    if args.input.is_dir():
        fnames = list(args.input.iterdir())
    elif args.input.is_file():
        fnames = [args.input]
    else:
        raise FileNotFoundError(args.input, " does not exists")

    imgs = [TRANSFORMS(Image.open(f).convert("RGB")).to(DEVICE) for f in fnames]
    orig_sizes = [(img.shape[-2], img.shape[-1]) for img in imgs]

    padded_imgs = []
    for img in imgs:
        pad_h, pad_w = (-img.shape[-2])%256, (-img.shape[-1])%256
        padded_imgs.append(F.pad(img,(0, pad_w, 0, pad_h), mode='constant', value=-1))

    with torch.no_grad():
        low_res = [F.interpolate(img[None],#+ 0.1*torch.randn_like(img[None]),
                             scale_factor=1./model.scale_factor, mode='bicubic', antialias=True).clamp(-1,1)
                               for img in padded_imgs]
        bicubic = [F.interpolate(img,
                             scale_factor=model.scale_factor, mode='bicubic', antialias=True).clamp(-1,1)
                               for img in low_res]

    
    result = []
    then = perf_counter()
    if args.windowed:
        result = inference_splitted_windows(model, padded_imgs, 256, 10, args.bs)
    else:
        for img in low_res:
            result.append(model.predict_step(img, None)[0])

    lapsed_time = perf_counter()-then
    print("Lapsed Time",lapsed_time)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    for res, o_size, lr,fname in zip(result, orig_sizes, bicubic, fnames):
        TF.to_pil_image(res[:,:o_size[0], :o_size[1]].cpu()*0.5+0.5).save(out_dir/f"{fname.stem}_pred.png")
        TF.to_pil_image(lr[0,:,:o_size[0], :o_size[1]].cpu()*0.5+0.5).save(out_dir/f"{fname.stem}_lr.png")
    print("FIN")

        
if __name__ == "__main__":
    main()
    