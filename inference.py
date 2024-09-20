import sys
import argparse
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
from diffusion_resshift_pixart import DiffuserSRResShift, F, make_grid, PeakSignalNoiseRatio
from utils import chunks

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
    n_rows = splitted_img.shape[1]
    splitted_img = rearrange(splitted_img, "c nh nw h w -> (nh nw) c h w", h=w_size, w=w_size)
    return splitted_img, orig_size, (padded_img.shape[1], padded_img.shape[2])

def merge_windows_padded_overlap(splitted_img:torch.Tensor, orig_size, padded_size, w_size, overlap):
    img = torch.zeros(3, padded_size[0], padded_size[1],
                      device=splitted_img.device, dtype=splitted_img.dtype)
    img_weights = torch.zeros(3, padded_size[0], padded_size[1],
                      device=splitted_img.device, dtype=splitted_img.dtype)
    
    # n_rows = (orig_size[0]-w_size)//(w_size-overlap) + bool((orig_size[0]-w_size)%(w_size-overlap)) + 1
    stride = w_size - overlap
    n_rows = ((padded_size[0]-w_size) // stride) + 1
    n_cols = splitted_img.shape[0] // n_rows
    # splitted_img = rearrange(splitted_img, "(nh nw) c h w -> nh nw c h w", nh=n_rows)
    for i in range(n_rows):
        for j in range(n_cols):
            img[:, i*stride:i*stride+w_size, j*stride:j*stride+w_size] += splitted_img[i*n_cols + j]
            img_weights[:, i*stride:i*stride+w_size, j*stride:j*stride+w_size] += torch.ones_like(splitted_img[i*n_cols + j])

    return (img / img_weights)[:, :orig_size[0], :orig_size[1]]

def collate_fn(img_list):
    sizes = [img.shape[0] for img in img_list]
    return torch.cat(img_list), sizes

def uncollate_fn(batch, sizes):
    last_size = 0
    result = []
    for it in range(sizes):
        result.append(batch[last_size:last_size + sizes[it]])
        last_size += sizes[it]
    return result

TRANSFORMS = tfms.Compose([
            tfms.ToImage(), tfms.ToDtype(torch.uint8, scale=True),
            tfms.RandomCrop(256),
            tfms.ToDtype(torch.float32, scale=True),
            tfms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
            ])

def open_img(fname):
    return TRANSFORMS(Image.open(fname))

# parser = argparse.ArgumentParser()
# parser.add_argument('checkpoint')
# parser.add_argument('--refiner', action='store_true')
# parser.add_argument('--cuda', action='store_true')
# parser.add_argument('--bs', default=8)
# parser.add_argument('-d', '--base-dir', default='.')
# parser.add_argument('-i', '--input-dir', nargs='+', default=[])
# parser.add_argument('-o', '--output-dir', default="results")
# parser.add_argument('-s', '--output-dir-std', default="results_std")
# parser.add_argument('-l', '--list', default='')

NSAMPLES = 32

def sample_forward(x, fname):
    img_tensor = open_img(fname)[None]

    with torch.no_grad():
        # Maybe Padding to be multiple of scale factor (?)
        low_res = F.interpolate(img_tensor, scale_factor=1./8.0,
                                 mode='bicubic', antialias=True).clamp(-1,1)
        up_low_res = F.interpolate(low_res, scale_factor=8.0,
                                 mode='bicubic', antialias=True).clamp(-1,1)

    from sampler import ResShiftDiffusion

    STEPS = 15
    sampler = ResShiftDiffusion(timesteps=STEPS, kappa=1.0)
    steps = torch.linspace(0, STEPS-1, 5, dtype=int)[1:]

    with torch.no_grad():
        eps = torch.randn_like(up_low_res)
        result = [img_tensor[0]]
        for time_step in steps:
            ts = torch.ones(img_tensor.shape[0], dtype=torch.long, device=img_tensor.device) * time_step
            noised = sampler.add_noise(img_tensor, up_low_res, eps, ts)
            result.append(noised[0].clamp(-1,1))

    # result.append(sampler.prior_sample(up_low_res, eps)[0].clamp(-1,1))
    result = make_grid(result, nrow=5)

    # print(f"{merged_img.shape = }")
    TF.to_pil_image(result*0.5+0.5).save("result.png")
    TF.to_pil_image(up_low_res[0]*0.5+0.5).save("lr.png")
    sampled = sampler.prior_sample(up_low_res, eps)[0].clamp(-1,1)
    TF.to_pil_image(sampled[0]*0.5+0.5).save("sampled.png")

        
if __name__ == "__main__":
    model = DiffuserSRResShift.load_from_checkpoint("lightning_logs/resshift_imagenet_pixart_vqvae/checkpoints/epoch=4-step=44970.ckpt").cuda()
    model.eval()
    fnames = list(Path("test_imgs").iterdir())
    imgs = [open_img(f) for f in fnames]
    img_tensor = torch.stack(imgs).cuda()
    print(f"{img_tensor.shape = }")
    with torch.no_grad():
        # Maybe Padding to be multiple of scale factor (?)
        low_res = F.interpolate(img_tensor, scale_factor=1./model.scale_factor,
                                 mode='bicubic', antialias=True).clamp(-1,1)
        up_low_res = F.interpolate(low_res, scale_factor=model.scale_factor,
                                 mode='bicubic', antialias=True).clamp(-1,1)
    
    # batch, orig_size, padded_size = split_windows_padded_overlap(up_low_res[0], 256, 10)
    # print(f"{batch.shape = }")

    with torch.inference_mode():
        pred = model.predict_step(low_res, None)
    
    
    # merged_img = merge_windows_padded_overlap(pred, orig_size, padded_size, 256, 10)
    comparison_img = torch.cat([make_grid(img_tensor, nrow=4), make_grid(pred, nrow=4)], dim=-1).cpu()
    TF.to_pil_image(comparison_img*0.5+0.5).save("result_test.png")

    # args = parser.parse_args()
    # if args.refiner:
    #     model = UnetRefiner.load_from_checkpoint(args.checkpoint)
    # else:
    #     model = DiffuserSRResShift.load_from_checkpoint(args.checkpoint)
    # model.eval()
    # if args.cuda:
    #     model.cuda()
    # BASE_DIR = Path(args.base_dir)
    # output_dir = Path(args.output_dir)
    # output_dir.mkdir(exist_ok=True)
    # output_dir_std = Path(args.output_dir_std)
    # output_dir_std.mkdir(exist_ok=True)
    # files = []
    # n_inputs = max(1, len(args.input_dir))
    # if args.list:
    #     with open(args.list, 'r') as f:
    #         files = [BASE_DIR/fname[:-1] for fname in f]
    # else:
    #     files = list((BASE_DIR/args.input_dir[0]).iterdir())
    # files = [files] + [[BASE_DIR/folder/f.name for f in files]
    #                     for folder in args.input_dir[1:]]
    
    # for fnames in chunks(list(zip(*files)), args.bs):
    #     batch = []
    #     for i in range(n_inputs):
    #         batch.append(torch.stack([open_img(fname[i]).cuda()
    #                                 if args.cuda else open_img(fname[i])
    #                                 for fname in fnames]))
    #     batch = tuple(batch)
    #     with torch.inference_mode():
    #         pred_list = []
    #         for _ in range(NSAMPLES):
    #             pred = model.predict_step(batch, None)*0.5+0.5
    #             pred_list.append(pred)
    #         pred_std = torch.stack(pred_list).std(dim=0)
    #     for it, fname in enumerate(fnames):
    #         # (output_dir/fname[0].name).parent.mkdir(exist_ok=True)
    #         # (output_dir_std/fname[0].name).parent.mkdir(exist_ok=True)
    #         TF.to_pil_image(pred[it]).save(output_dir/fname[0].name)
    #         TF.to_pil_image(pred_std[it]).save(output_dir_std/fname[0].name)
    # print("DONE", file=sys.stderr)
    