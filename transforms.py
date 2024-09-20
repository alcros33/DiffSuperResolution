import random
from PIL import Image
import numpy as np
import cv2
import torch
from einops import rearrange, reduce, repeat, einsum

def split_windows_padded_overlap(img:torch.Tensor, w_size:int, overlap=5):

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

def merge_windows_padded_overlap(splitted_img:torch.Tensor, orig_size:int, padded_size:int,
                                 w_size:int, overlap:int):
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

def collate_fn_splitted(img_list):
    sizes = [img.shape[0] for img in img_list]
    return torch.cat(img_list), sizes

def uncollate_fn_splitted(batch, sizes):
    last_size = 0
    result = []
    for it in range(sizes):
        result.append(batch[last_size:last_size + sizes[it]])
        last_size += sizes[it]
    return result


def degradation_bsrgan_variant(image, sf=4, jpeg_prob=0.9, scale2_prob=0.25):
    """
    This is the degradation model of BSRGAN from the paper
    "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"
    ----------
    sf: scale factor
    isp_model: camera ISP model
    Returns
    -------
    img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
    hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
    """
    image = util.uint2single(image)
    isp_prob, jpeg_prob, scale2_prob = 0.25, 0.9, 0.25
    sf_ori = sf

    # Ensure is multiple of scale_factor
    h1, w1 = image.shape[:2]
    image = image.copy()[:w1 - w1 % sf, :h1 - h1 % sf, ...]  # mod crop
    h, w = image.shape[:2]

    hq = image.copy()

    # with a random prob rescale downscale two times instead of one
    if sf == 4 and random.random() < scale2_prob:  # downsample1
        if np.random.rand() < 0.5:
            # random method of rescaling
            image = cv2.resize(image, (int(1 / 2 * image.shape[1]), int(1 / 2 * image.shape[0])),
                               interpolation=random.choice([1, 2, 3]))
        else:
            image = util.imresize_np(image, 1 / 2, True)
        image = np.clip(image, 0.0, 1.0)
        sf = 2

    shuffle_order = random.sample(range(7), 7)
    idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
    if idx1 > idx2:  # keep downsample3 last
        shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]

    for i in shuffle_order:

        if i == 0 or i == 1:
            image = add_blur(image, sf=sf)

        elif i == 2:
            a, b = image.shape[1], image.shape[0]
            # downsample2
            if random.random() < 0.75:
                sf1 = random.uniform(1, 2 * sf)
                image = cv2.resize(image, (int(1 / sf1 * image.shape[1]), int(1 / sf1 * image.shape[0])),
                                   interpolation=random.choice([1, 2, 3]))
            else:
                k = fspecial('gaussian', 25, random.uniform(0.1, 0.6 * sf))
                k_shifted = shift_pixel(k, sf)
                k_shifted = k_shifted / k_shifted.sum()  # blur with shifted kernel
                image = ndimage.filters.convolve(image, np.expand_dims(k_shifted, axis=2), mode='mirror')
                image = image[0::sf, 0::sf, ...]  # nearest downsampling
            image = np.clip(image, 0.0, 1.0)

        elif i == 3:
            # downsample3
            image = cv2.resize(image, (int(1 / sf * a), int(1 / sf * b)), interpolation=random.choice([1, 2, 3]))
            image = np.clip(image, 0.0, 1.0)

        elif i == 4:
            # add Gaussian noise
            image = add_Gaussian_noise(image, noise_level1=2, noise_level2=25)

        elif i == 5:
            # add JPEG noise
            if random.random() < jpeg_prob:
                image = add_JPEG_noise(image)

        # elif i == 6:
        #     # add processed camera sensor noise
        #     if random.random() < isp_prob and isp_model is not None:
        #         with torch.no_grad():
        #             img, hq = isp_model.forward(img.copy(), hq)

    # add final JPEG compression noise
    image = add_JPEG_noise(image)
    image = util.single2uint(image)
    example = {"image":image}
    return example