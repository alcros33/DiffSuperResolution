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
import itertools

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
parser.add_argument('--cuda', action='store_true')
parser.add_argument('-i', '--input', type=Path)
parser.add_argument('-o', '--output-dir', default="results")
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--macro-bs', type=int, default=32)

WINDOW_SIZE = 256
OVERLAP = 12

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
    model = MTYPE.load_from_checkpoint(args.checkpoint, map_location=DEVICE, strict=False)
    model.eval()
    SF = model.scale_factor

    if args.input.is_dir():
        fnames = list(args.input.iterdir())
    elif args.input.is_file():
        fnames = [args.input]
    else:
        raise FileNotFoundError(args.input, " does not exists")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    for curr_fnames in chunks(fnames, args.macro_bs):
        imgs = [TRANSFORMS(Image.open(f).convert("RGB")) for f in curr_fnames]

        batch, padded_size, orig_size, n_windows = [],[],[],[]
        for img in imgs:
            b, o, p = split_windows_padded_overlap(img,
                                                   WINDOW_SIZE//SF,
                                                   OVERLAP//SF)
            n_windows.append(b.shape[0])
            padded_size.append(p); orig_size.append(o)
            batch.append(b)
        
        preds = []
        batch = torch.cat(batch)
        for it in range(0, len(batch), args.bs):
            preds.append(model.predict_step(batch[it:it+args.bs].to(DEVICE), None).cpu())
        preds = uncollate_fn(torch.cat(preds), n_windows)
        result = []
        for p,o,pad in zip(preds, orig_size, padded_size):
            merged = merge_windows_padded_overlap(p,
                                                  (o[0]*SF, o[1]*SF),
                                                  (pad[0]*SF, pad[1]*SF),
                                                  WINDOW_SIZE, OVERLAP)
            result.append(merged)
        
        for res, fname in zip(result, curr_fnames):
            TF.to_pil_image(res.cpu()*0.5+0.5).save(out_dir/f"{fname.stem}_pred.png")
    print("FIN", file=sys.stderr)

        
if __name__ == "__main__":
    main()
    