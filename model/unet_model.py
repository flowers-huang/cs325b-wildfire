import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import numpy as np
from einops import rearrange, pack
import matplotlib.pyplot as plt

from data.image_processing import combine_tiles

class UNetModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, encoder_weights='imagenet', max_batch_size=16, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.training_step_outputs = []
        self.valid_step_outputs = []
        self.test_step_outputs = []

        self.max_batch_size = max_batch_size

    def forward(self, image):
        # normalize image here
        # image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def predict(self, batch):
        image = batch['image']
        image_mode = batch['mode'][0]
        
        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert (image.ndim == 4 and image_mode in ['random_crop', 'scale']) or (image.ndim == 5 and image_mode == 'tiled')
        h, w = image.shape[-2:]

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        assert h % 32 == 0 and w % 32 == 0
        
        if image_mode in ['random_crop', 'scale']:
            logits_mask = self.forward(image)
        elif image_mode == 'tiled':
            overlap = batch['overlap'][0]
            crop_size = batch['crop_size'][0]
            
            b, t, c, x, y = image.shape
            image = rearrange(image, 'b t c x y -> (b t) c x y')

            max_batch_size = self.max_batch_size

            # ensure we get OOM error
            if b*t > max_batch_size:
                image_batches = torch.split(image, max_batch_size, dim=0)
                
                result_batches = []
                
                for img in image_batches:
                    with torch.no_grad():
                        logits_mask = self.forward(img).cpu()
                    # Append the output logits to results
                    result_batches.append(logits_mask)
                   
                # Concatenate the result batches
                logits_mask = torch.cat(result_batches, dim=0)
                
            else:
                with torch.no_grad():
                    logits_mask = self.forward(image).cpu()
            
            # logits_mask = self.forward(image).cpu()
            logits_mask = rearrange(logits_mask, '(b t) c x y -> b t c x y', b=b, t=t)

            # untile the predicted masks
            combined = []
            for i in range(b):
                mask = combine_tiles(logits_mask[i], *batch['image_shape'][i], crop_size, overlap)
                combined.append(mask)
            logits_mask = torch.stack(combined)
        else:
            raise ValueError(f"unknown image mode {image_mode}")

        return logits_mask
    
    def shared_step(self, batch, outputs, stage):
        
        image = batch['image']
        image_mode = batch['mode'][0]

        # # Shape of the image should be (batch_size, num_channels, height, width)
        # # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        # assert (image.ndim == 4 and image_mode == 'random_crop') or (image.ndim == 5 and image_mode == 'tiled')

        # # Check that image dimensions are divisible by 32, 
        # # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # # and we will get an error trying to concat these features
        # h, w = image.shape[-2:]
        # assert h % 32 == 0 and w % 32 == 0

        mask = batch['mask']

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0
        
        logits_mask = self.predict(batch)
            
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

        # if stage in ["valid", "test"]:
        #     tensorboard = self.logger.experiment
        #     plt.hist(smp.metrics.iou_score(tp, fp, fn, tn).cpu().flatten())
        #     tensorboard.add_figure(stage, plt.gcf())

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