import os
import torch
import numpy as np
from einops import rearrange, pack

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F

from tqdm import tqdm

from .image_processing import rescale_imshow_rgb, calculate_dnbr, resize_and_pad, random_crop, get_overlapping_tiles


STATISTICS = {
    # 'mean': [97.09689363, 97.28427036, 98.57064272, 116.35852306, 114.47721449, 3.76334845],
    # 'std': [64.17795164, 64.1141886, 65.90140603, 64.79696877, 66.58678831, 30.22436968]
    'mean': [0.38547045, 0.38573036, 0.39038286, 0.45860687, 0.45115775, 0.04383795],
    'std': [0.27072474, 0.26539338, 0.27010038, 0.26388097, 0.26939374, 13.463576]
}

from pathlib import Path
import os

import rasterio
import rioxarray
import xarray as xr

from torch.utils.data import Dataset

def load_image(file_path, channels=[2, 3, 4]):
    image = xr.open_dataset(file_path).to_array().squeeze().transpose('y', 'x', 'band')
    # image = rescale_imshow_rgb(image)
    image = image[:, :, channels]
    return image

# regions that are completely NaNs
NAN_REGIONS = {'CA4090112136820140731', 'CA3791712013219870831', 'CA3742412156820200816', 'CA3256911672719951128'}

class ImageMaskDataset(Dataset):
    def __init__(self, directory = Path("~/data/processed"), post_only=False, dnbr=False, sample=False, k=100):
        """
        directory:
        post_only (bool): if True, include pre-fire images
        dnbr (bool): if True, include dNBR as an extra image channel
        sample (bool): whether to return full images or sample of pixels
        k (int): if sample is True, number of pixels to sample
        """
        if isinstance(directory, str):
            directory = Path(directory)
        # directory = Path("~/data/gdrive/nonCA_processed_all")
        # mask_root_dir = directory / "washington-masks"
        # pre_root_dir = directory / "washington-pre-scaled-fire-images"
        # post_root_dir = directory / "washington-post-fire-images"
        mask_root_dir = directory / "masks"
        pre_root_dir = directory / "pre-scaled-fire-images"
        post_root_dir = directory / "post-fire-images"

        self.regions = list(set([Path(f).stem for f in os.listdir(mask_root_dir)]) & set([Path(f).stem for f in os.listdir(pre_root_dir)]) & set([Path(f).stem for f in os.listdir(post_root_dir)]))
        
        for r in NAN_REGIONS:
            self.regions.remove(r)
            
        self.files_mask = {f: mask_root_dir / f'{f}.npy' for f in self.regions}
        self.files_pre = {f: pre_root_dir / f'{f}.tif' for f in self.regions}
        self.files_post = {f: post_root_dir / f'{f}.tif' for f in self.regions}

        self.post_only = post_only
        self.dnbr = dnbr
        self.k = k
        self.sample = sample

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        region = self.regions[idx]

        pre_data = load_image(self.files_pre[region], channels=[0, 1, 2, 3, 4, 5])
        post_data = load_image(self.files_post[region], channels=[0, 1, 2, 3, 4, 5])
        mask = np.load(self.files_mask[region])

        if self.dnbr:
            dnbr = np.nan_to_num(calculate_dnbr(pre_data, post_data).values)
            
        pre_data, post_data = rescale_imshow_rgb(pre_data), rescale_imshow_rgb(post_data)

        if not self.sample:
            mean = STATISTICS['mean'].copy()
            std = STATISTICS['std'].copy()

            if self.post_only:
                image = post_data
            else:
                mean = mean * 2
                std = std * 2
                image, _ = pack([post_data, pre_data], 'x y *')

            if self.dnbr:
                mean += [0.]
                std += [1.]
                image, _ = pack([image, dnbr], 'x y *')
                
            image = torch.tensor(image, dtype=torch.float)
            ground_truth_mask = torch.tensor(mask, dtype=torch.float)
    
            transform = T.Normalize(mean=mean, std=std)
            image = transform(rearrange(image, 'x y c -> c x y'))
            mask = rearrange(ground_truth_mask, 'x y -> 1 x y')
            
            return {'image': image, 'mask': mask}
        
        # Total number of pixels in one channel of an image
        total_pixels = np.prod(pre_data.shape[:-1])
    
        # Generate k random indices into the flattened image
        indices = np.random.choice(total_pixels, size=self.k, replace=False)
    
        # Convert the 1D indices into 2D coordinates
        x_cords, y_cords = np.unravel_index(indices, pre_data.shape[:-1])
    
        pre_data_samples = pre_data[x_cords, y_cords, :]
        post_data_samples = post_data[x_cords, y_cords, :]
        mask_samples = mask[x_cords, y_cords]

        if self.post_only:
            sample = post_data_samples
        else:
            sample, _ = pack([pre_data_samples, post_data_samples], 'n *')

        if self.dnbr:
            dnbr_samples = dnbr[x_cords, y_cords]
            sample, _ = pack([sample, dnbr_samples], 'n *')
        
        return {'image': torch.tensor(sample), 'mask': torch.tensor(mask_samples)}


class PrecomputedDataset(Dataset):
    def __init__(self, dataset, crop_method='random', crop_size=256, overlap=64):
        print("Precomputing Dataset:")
        self.dataset = [d for d in tqdm(dataset)]
        self.crop_size = crop_size
        self.overlap = overlap
        self.crop_method = crop_method

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        If crop_method = 'random', image is of shape (c x y)
        If crop method = 'tile', image is of shape (t c crop_size crop_size) where t is the number of tiles
        """
        sample = self.dataset[idx]
        if self.crop_method == 'random':
            image, mask = random_crop(sample['image'], sample['mask'], self.crop_size)
            image_shape = image.shape[-2:]
            return {'image': image, 'mask': mask, 'mode': 'random_crop'}
        elif self.crop_method == 'tile':
            image = torch.stack(get_overlapping_tiles(sample['image'], self.crop_size, overlap=self.overlap))
            mask = sample['mask']
            image_shape = sample['image'].shape[-2:]
            return {'image': image, 'mask': mask, 'image_shape': image_shape, 'mode': 'tiled', 'crop_size': self.crop_size, 'overlap': self.overlap}
        elif self.crop_method == 'scale':
            out_dim = (self.crop_size, self.crop_size)
            image, mask = resize_and_pad(sample['image'], out_dim), resize_and_pad(sample['mask'], out_dim)
            return {'image': image, 'mask': mask, 'mode': 'scale'}
        else:
            raise ValueError(f"Unknown crop method {self.crop_method}")

# precomputed_dataset = PrecomputedDataset(data)
# precomputed_dataset.crop_method = 'none'

# # Define the proportions for the train/val split
# train_prop = int(0.8 * len(precomputed_dataset))  # 80% of the dataset
# val_prop = len(precomputed_dataset) - train_prop  # 20% of the dataset

# # Create the train and validation datasets
# train_dataset, val_dataset = random_split(precomputed_dataset, [train_prop, val_prop], generator=torch.Generator().manual_seed(42))

# train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=8, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=8, shuffle=False)