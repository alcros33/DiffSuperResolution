import random
from PIL import Image
from pathlib import Path
import pandas as pd
import torch
import torchvision.transforms.v2 as tfms, torchvision.transforms.v2.functional as TF
from torch.utils.data import Dataset, DataLoader
import lightning as L
from functools import partial
from utils import random_split

class Pad2Square:
	def __call__(self, image):
		_, h, w = image.shape
		max_wh = max(w,h)
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		return TF.pad(image,(hp, vp), 0, 'constant')

class SimpleImageDataset(Dataset):
    def __init__(self, img_files, transforms, mode="RGB"):
        self.img_files = img_files
        self.transforms = transforms
        self.mode = mode
    
    def __getitem__(self, index):
        return self.transforms(Image.open(self.img_files[index]).convert(self.mode))

    def __len__(self):
        return len(self.img_files)

class SimpleImageDataModule(L.LightningDataModule):
    def __init__(self, data_dir:Path, batch_size: int = 8,
                 img_size=256, 
                 train_img_list=None, valid_img_list=None, test_img_list=None,
                 valid_pct=0.05, test_pct=0.05,
                 mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], img_mode="RGB", num_workers=1):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        
        if train_img_list is None:
            self.train_img_files = [f for f in data_dir.iterdir()]
        else:
            with open(train_img_list,'r') as f:
                self.train_img_files = [data_dir/fname[:-1] for fname in f]

        if test_img_list is None:
            self.train_img_files, self.test_img_files = random_split(self.train_img_files, test_pct)
        else:
            with open(test_img_list,'r') as f:
                self.test_img_files = [data_dir/fname[:-1] for fname in f]
        
        if valid_img_list is None:
            self.train_img_files, self.valid_img_files = random_split(self.train_img_files, valid_pct)
        else:
            with open(valid_img_list,'r') as f:
                self.valid_img_files = [data_dir/fname[:-1] for fname in f]

        self.transforms = tfms.Compose([
            tfms.ToImage(), tfms.ToDtype(torch.uint8, scale=True),
            tfms.RandomCrop(img_size),
            tfms.RandomHorizontalFlip(),
            # tfms.RandomAdjustSharpness(1.5, 0.3),
            # tfms.RandomAutocontrast(0.3),
            # tfms.RandomEqualize(0.3),
            tfms.ToDtype(torch.float32, scale=True),
            tfms.Normalize(mean, std),
            ])

        self.test_transforms = tfms.Compose([
            tfms.ToImage(), tfms.ToDtype(torch.uint8, scale=True),
            tfms.RandomCrop(img_size),
            tfms.ToDtype(torch.float32, scale=True),
            tfms.Normalize(mean, std),
            ])
    
        self.dataset_train = SimpleImageDataset(self.train_img_files, self.transforms, mode=img_mode)
        self.dataset_valid = SimpleImageDataset(self.valid_img_files, self.test_transforms, mode=img_mode)
        self.dataset_test = SimpleImageDataset(self.test_img_files, self.test_transforms, mode=img_mode)
    
    def setup(self, stage: str):
        pass
    
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)