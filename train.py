from torch.utils.data import random_split
import torch
import pytorch_lightning as pl

from data.dataloader import ImageMaskDataset, PrecomputedDataset
from torch.utils.data import DataLoader

from model.unet_model import UNetModel

import argparse

torch.set_num_threads(5)
torch.set_num_interop_threads(5)

from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def train_model(model_name, backbone, device_id, in_channels=12, pretrained=True, log_dir="logs"):
    encoder_weights = 'imagenet' if pretrained else None
    model = UNetModel(model_name, backbone, in_channels=in_channels, out_classes=1, encoder_weights=encoder_weights)
  
    logger = TensorBoardLogger(save_dir=log_dir, name=f"{model_name}_{backbone}")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--pretrained", type=bool, default=True)
    
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_dir", type=str, default="~/logs")
    parser.add_argument("--data_dir", type=str, default="~/data/processed")

    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--crop_method", type=str, default='scale')  # scale, random, tile
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)

    parser.add_argument("--sample", type=bool, default=False)
    parser.add_argument("--post_only", type=bool, default=False)
    parser.add_argument("--dnbr", type=bool, default=True)
    parser.add_argument("--k", type=int, default=1000)
    args = parser.parse_args()
    
    dataset = ImageMaskDataset(directory=args.data_dir, sample=args.sample, post_only=args.post_only, dnbr=args.dnbr, k=args.k)
    precomputed_dataset = PrecomputedDataset(dataset, args.crop_method, args.crop_size, args.overlap)
    
    # Define the proportions for the train/val split
    train_prop = int(args.split * len(precomputed_dataset))
    val_prop = len(precomputed_dataset) - train_prop
    
    # Create the train and validation datasets
    train_dataset, val_dataset = random_split(precomputed_dataset, [train_prop, val_prop], generator=torch.Generator().manual_seed(42))
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    in_channels = 6 * (2 - int(args.post_only)) + int(args.dnbr)
    train_model(args.model_type, args.backbone, args.device, in_channels, args.pretrained, args.log_dir)