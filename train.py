import torch
import numpy as np
import os
from torch.utils.data import Dataset
from einops import rearrange
from einops import pack, unpack
from torch.utils.data import random_split
from pathlib import Path
import rasterio
# from torchvision import transforms

class ImageMaskDataset(Dataset):
    def __init__(self, directory = Path("/lfs/turing3/0/kaif/data/processed"), both=True, patch_size=32):
        mask_root_dir = directory / "masks"
        pre_root_dir = directory / "pre-fire-images"
        post_root_dir = directory / "post-fire-images"
        
        self.regions = [Path(f).stem for f in os.listdir(mask_root_dir)]
        
        self.files_mask = {f: mask_root_dir / f'{f}.npy' for f in self.regions}
        self.files_pre = {f: pre_root_dir / f'{f}.tif' for f in self.regions}
        self.files_post = {f: post_root_dir / f'{f}.tif' for f in self.regions}

        self.both = both
        self.patch_size = patch_size

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        region = self.regions[idx]

        with rasterio.open(self.files_pre[region]) as src:
            pre_data = src.read()
        
        with rasterio.open(self.files_post[region]) as src:
            post_data = src.read()

        mask = np.load(self.files_mask[region])

        if self.both:
            image_data, _ = pack([pre_data, post_data], '* x y')
        else:
            image_data = post_data

        height, width = image_data.shape[1], image_data.shape[2] 
        start_x = np.random.randint(0, width - self.patch_size + 1)
        start_y = np.random.randint(0, height - self.patch_size + 1)

        # Crop the images to get the patches
        end_x, end_y = start_x + self.patch_size, start_y + self.patch_size

        image_patch = np.nan_to_num(image_data[:, start_y:end_y, start_x:end_x]) / 25000
        mask_patch = np.nan_to_num(mask[start_y:end_y, start_x:end_x])#.mean()
        
        return torch.tensor(image_patch, dtype=torch.float), torch.tensor(mask_patch, dtype=torch.float)


import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 10, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(10, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = torch.sigmoid(self.conv5(x))  # Using sigmoid because it's a binary mask (0 - 1)
        return x.squeeze(dim=1)
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(64, 1)

#     def forward(self, x):
#         x = nn.functional.relu(self.conv1(x))
#         x = nn.functional.relu(self.conv2(x))
#         x = nn.functional.relu(self.conv3(x))
#         x = self.avg_pool(x)
#         x = torch.flatten(x, start_dim=1)  # flatten all dimensions except batch
#         x = self.fc(x)
#         x = torch.sigmoid(x)
#         return x.squeeze(dim=1)

class PixelWiseLogisticRegression(nn.Module):
    def __init__(self, chanels):
        super(PixelWiseLogisticRegression, self).__init__()

        self.linear = nn.Linear(chanels, 1)

    def forward(self, input):
        b, c, x, y = input.shape
        input = rearrange(input, 'b c x y -> (b x y) c')
        out = self.linear(input)
        out = torch.sigmoid(out)
        out = rearrange(out, '(b x y) 1 -> b x y', x=x, y=y)
        return out

import lightning.pytorch as pl

def IoU(pred, target):
    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    return np.sum(intersection) / np.sum(union)

def pixel_accuracy(pred, target):
    return np.mean(pred == target)
    

class MaskPredictor(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = PixelWiseLogisticRegression(6)
        self.criterion = nn.BCELoss()
        # self.criterion = nn.MSELoss()

    # Forward pass
    def forward(self, x):
        return self.model(x)

    # Training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss_mse', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.criterion(y_hat, y)

        self.log('val_mean', y_hat.cpu().numpy().mean(), on_epoch=True)
        self.log('val_std', y_hat.cpu().numpy().std(), on_epoch=True)

        self.log('val_mean2', y.cpu().numpy().mean(), on_epoch=True)
        self.log('val_std2', y.cpu().numpy().std(), on_epoch=True)
        
        # Convert the predictions and targets to boolean arrays
        pred = y_hat.cpu().numpy() > 0.5
        target = y.cpu().numpy() > 0.5
        
        # Compute the metrics
        iou = IoU(pred, target)
        acc = pixel_accuracy(pred, target)
        
        self.log('val_loss_mse', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('iou', iou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('pixel_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    # Configure the optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


import matplotlib.pyplot as plt

def plot_masks(true_mask, predicted_mask, filename):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(true_mask, cmap='gray')
    ax[0].set_title('True Mask')
    ax[0].axis('off')

    ax[1].imshow(predicted_mask, cmap='gray')
    ax[1].set_title('Predicted Mask')
    ax[1].axis('off')
    
    plt.tight_layout()
    
    # save the figure
    fig.savefig(f'mask_plots/{filename}.png') 
    plt.close(fig)  # close the figure

# Create the mask_plots directory if it doesn't exist
if not os.path.exists('mask_plots'):
    os.makedirs('mask_plots')

from lightning.pytorch.callbacks import EarlyStopping
early_stopping = EarlyStopping('iou_epoch/dataloader_idx_1', patience=10, mode='max')

if __name__ == "__main__":
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="2"

    dataset = ImageMaskDataset(directory=Path("/lfs/turing3/0/kaif/data/processed"), both=False)
    
    # Define the proportions for the train/val split
    train_prop = int(0.8 * len(dataset))  # 80% of the dataset
    val_prop = len(dataset) - train_prop  # 20% of the dataset
    
    # Create the train and validation datasets
    train_dataset, val_dataset = random_split(dataset, [train_prop, val_prop])
    
    # Create data loaders for the train and validation datasets
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = MaskPredictor()
    
    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=20, accelerator='gpu')#, callbacks=[early_stopping])
    
    # Train the model
    trainer.fit(model, train_loader, [train_loader, val_loader])

    # Use the model to get the predictions
    model.eval()
    with torch.no_grad():
        j = 0
        for batch in val_loader:
            j += 1
            if j > 10:
                break
                
            images, true_masks = batch
            images, true_masks = images.to(model.device), true_masks.to(model.device)
            predicted_masks = model(images)  # Forward pass
            
            # Convert the tensors to numpy arrays for visualization
            true_masks = true_masks.cpu().numpy()
            predicted_masks = predicted_masks.cpu().numpy()
    
            # Plot the first image and mask in the batch
            plot_masks(true_masks[0], predicted_masks[0], f'batch_{j}_mask')