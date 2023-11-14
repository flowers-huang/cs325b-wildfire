import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import numpy as np
from einops import rearrange, pack

from pprint import pprint
from torch.utils.data import DataLoader

params = {
    'mean': [97.09689363, 97.28427036, 98.57064272, 116.35852306, 114.47721449, 3.76334845],
    'std': [64.17795164, 64.1141886, 65.90140603, 64.79696877, 66.58678831, 30.22436968]
}

ROBUST_PERCENTILE = 2.0

def rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True, axis=-1):
    assert robust or vmin is not None or vmax is not None

    ndim = len(darray.shape)
    if axis < 0:
        axis = ndim + axis

    reduce_dim = list(range(ndim))
    reduce_dim.remove(axis)

    # Calculate vmin and vmax automatically for `robust=True`
    # Assume that the last dimension of the array represents color channels
    # Make sure to apply np.nanpercentile over this dimension by specifying axis=-1
    if robust:
        if vmax is None:
            vmax = np.nanpercentile(darray, 100 - ROBUST_PERCENTILE, axis=reduce_dim, keepdims=True)
        if vmin is None:
            vmin = np.nanpercentile(darray, ROBUST_PERCENTILE, axis=reduce_dim, keepdims=True)
    # If not robust and one bound is None, calculate the default other bound
    # and check that an interval between them exists.
    elif vmax is None:
        vmax = 255 if np.issubdtype(darray.dtype, np.integer) else 1
        if np.any(vmax < vmin):
            raise ValueError(
                f"vmin={vmin!r} less than the default vmax ({vmax!r}) - you must supply "
                "a vmax > vmin in this case."
            )
    elif vmin is None:
        vmin = 0
        if np.any(vmin > vmax):
            raise ValueError(
                f"vmax={vmax!r} is less than the default vmin (0) - you must supply "
                "a vmin < vmax in this case."
            )
    # Compute a mask for where vmax equals vmin
    vmax_equals_vmin = np.isclose(vmax, vmin)

    # Avoid division by zero by replacing zero divisors with 1
    divisor = np.where(vmax_equals_vmin, vmax, vmax - vmin)

    # Scale interval [vmin .. vmax] to [0 .. 1], using darray as 64-bit float
    darray = ((darray.astype("f8") - vmin) / divisor).astype("f4")
    
    return np.nan_to_num(np.minimum(np.maximum(darray, 0), 1) * 255).astype(np.uint8)


from pathlib import Path
import os

import rasterio
import rioxarray
import xarray as xr

from torch.utils.data import Dataset

def load_image(file_path, channels=[2, 3, 4]):
    image = xr.open_dataset(file_path).to_array().squeeze().transpose('y', 'x', 'band')
    image = rescale_imshow_rgb(image)
    image = image[:, :, channels]
    return image

bad_regions = {'CA4090112136820140731', 'CA3791712013219870831', 'CA3742412156820200816', 'CA3256911672719951128'}

class ImageMaskDataset(Dataset):
    def __init__(self, directory = Path("/lfs/turing3/0/kaif/data/processed")):
        mask_root_dir = directory / "masks"
        pre_root_dir = directory / "pre-scaled-fire-images"
        post_root_dir = directory / "post-fire-images"
        
        self.regions = [Path(f).stem for f in os.listdir(mask_root_dir)]
        for r in bad_regions:
            self.regions.remove(r)
        
        self.files_mask = {f: mask_root_dir / f'{f}.npy' for f in self.regions}
        self.files_pre = {f: pre_root_dir / f'{f}.tif' for f in self.regions}
        self.files_post = {f: post_root_dir / f'{f}.tif' for f in self.regions}

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        region = self.regions[idx]

        pre_data = load_image(self.files_pre[region], channels=[0, 1, 2, 3, 4, 5])
        post_data = load_image(self.files_post[region], channels=[0, 1, 2, 3, 4, 5])
        mask = np.load(self.files_mask[region])

        return pre_data, post_data, mask


from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F

def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]
    
    return bbox

def resize_and_pad(img, output_size=(256,256), means=params['mean'], stds=params['std'], padding_mode='constant'):
    # Get initial dimensions
    c, h, w = img.shape

    # Normalise the image
    if means is not None and stds is not None:
        transform = T.Normalize(mean=means, std=stds)
        img = transform(img)

    # Calculate new dimension while maintaining the aspect ratio
    if h >= w:
        new_h, new_w = output_size[0], int(output_size[0] * w / h)
    else:
        new_h, new_w = int(output_size[1] * h / w), output_size[1]

    # Resize the image
    img = F.resize(img, (new_h, new_w))

    # This is padding, it requires padding_left, padding_right,
    # padding_top and padding_bottom respectively.
    pad_height = max(output_size[0] - img.shape[1], 0)
    pad_width = max(output_size[1] - img.shape[2], 0)

    # Center padding
    pad_height1 = pad_height // 2
    pad_height2 = pad_height - pad_height1
    pad_width1 = pad_width // 2
    pad_width2 = pad_width - pad_width1

    # Pad the image
    img = F.pad(img, (pad_width1, pad_height1, pad_width2, pad_height2), padding_mode=padding_mode)

    return img

class SAMDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pre_data, post_data, mask = self.dataset[idx]
        image, _ = pack([post_data, pre_data], 'x y *')
        image = torch.tensor(image, dtype=torch.float)
        ground_truth_mask = torch.tensor(mask, dtype=torch.float)

        image = resize_and_pad(rearrange(image, 'x y c -> c x y'), means=params['mean']*2, stds=params['std']*2)
        mask = resize_and_pad(rearrange(ground_truth_mask, 'x y -> 1 x y'), means=None, stds=None)
        return {'image': image, 'mask': mask}

import torch

torch.set_num_threads(5)
torch.set_num_interop_threads(5)


from torch.utils.data import DataLoader
from torch.utils.data import random_split

dataset = ImageMaskDataset()
sam_dataset = SAMDataset(dataset=dataset)

# Define the proportions for the train/val split
train_prop = int(0.8 * len(dataset))  # 80% of the dataset
val_prop = len(dataset) - train_prop  # 20% of the dataset

# Create the train and validation datasets
train_dataset, val_dataset = random_split(sam_dataset, [train_prop, val_prop], generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_dataset, batch_size=24, num_workers=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=24, num_workers=8, shuffle=False)

class UNetModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        # params = smp.encoders.get_preprocessing_params(encoder_name)
        # self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        # self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.training_step_outputs = []
        self.valid_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # normalize image here
        # image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, outputs, stage):
        
        image = batch['image']

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch['mask']

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        metrics = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        outputs.append(metrics)
        return metrics

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        if stage in ["valid", "test"]:
            tensorboard = self.logger.experiment
            plt.hist(smp.metrics.iou_score(tp, fp, fn, tn).cpu().flatten())
            tensorboard.add_figure(stage, plt.gcf())

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_accuracy": smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro"),
            f"{stage}_f1": smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro"),
            f"{stage}_balanced_accuracy": smp.metrics.balanced_accuracy(tp, fp, fn, tn, reduction="micro")
        }
        
        self.log_dict(metrics, prog_bar=True)
        
        outputs.clear()

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, self.training_step_outputs, "train")            

    def on_train_epoch_end(self):
        return self.shared_epoch_end(self.training_step_outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, self.valid_step_outputs, "valid")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end(self.valid_step_outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, self.test_step_outputs, "test")  

    def on_test_epoch_end(self):
        return self.shared_epoch_end(self.test_step_outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


dataset = ImageMaskDataset()
sam_dataset = SAMDataset(dataset=dataset)


from tqdm import tqdm
data = [d for d in tqdm(sam_dataset)]

class PrecomputedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

precomputed_dataset = PrecomputedDataset(data)

# Define the proportions for the train/val split
train_prop = int(0.8 * len(precomputed_dataset))  # 80% of the dataset
val_prop = len(precomputed_dataset) - train_prop  # 20% of the dataset

# Create the train and validation datasets
train_dataset, val_dataset = random_split(precomputed_dataset, [train_prop, val_prop], generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=8, shuffle=False)


from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def train_model(model_name, backbone, device_id):
    model = UNetModel(model_name, backbone, in_channels=12, out_classes=1)
  
    logger = TensorBoardLogger(save_dir='logs', name=f"{model_name}_{backbone}")

    early_stop_callback = EarlyStopping(
        monitor="valid_dataset_iou", 
        min_delta=0.00, 
        patience=5, 
        verbose=False, 
        mode="max"
    )

    trainer = pl.Trainer(
        accelerator='auto',
        devices=[device_id],
        max_epochs=20,
        logger=logger,
        callbacks=[early_stop_callback]
    )

    trainer.fit(
        model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader,
    )

import argparse
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from multiprocessing import Process

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()

    processes = []
    # i = 0
    # devices = [0, 3, 4, 5, 6, 7]
    for model_type in args.model_type.split(','):
        for backbone in args.backbone.split(','):
            # Start a new process for each combination
            train_model(model_type, backbone, args.device)
            # p.start()
            # processes.append(p)
            # i += 1

    # Wait for all processes to finish
    # for p in processes:
    #     p.join()
    
    # models = ["Unet", "Unet++", "FPN", "DeepLabV3+"]
    # backbones = ["resnet18", "resnet34", "resnet50", "resnext101_32x8d", "timm-gernet_l"]