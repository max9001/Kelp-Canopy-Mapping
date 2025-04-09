import pytorch_lightning as pl
import torch
import torch.nn as nn
import tifffile as tiff
import torchvision
from typing import List, Tuple
import numpy as np
from pathlib import Path
from pyprojroot import here
from pathlib import Path
import os
import sys
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
from tqdm import tqdm # Keep tqdm for potential use elsewhere if needed

# --- Metrics ---
# Import necessary metrics if you want to log them during testing
from torchmetrics import Accuracy, JaccardIndex

dir_root = here()
sys.path.append(str(dir_root))
from utils.get_data import prepare_filenames



class KelpSegmentationModel(pl.LightningModule):
    def __init__(self, target_size=(25, 25), learning_rate=1e-4): # Add learning_rate
        super().__init__()
        self.save_hyperparameters() # Save learning_rate etc.
        self.target_size = target_size
        # Encoder
        base_encoder = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        base_encoder.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(
            base_encoder.conv1, base_encoder.bn1, base_encoder.relu, base_encoder.maxpool,
            base_encoder.layer1, base_encoder.layer2, base_encoder.layer3, base_encoder.layer4
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.criterion = nn.BCEWithLogitsLoss()

        # --- Metrics for Validation and Testing ---
        self.val_iou = JaccardIndex(task="binary")
        self.test_iou = JaccardIndex(task="binary")
        self.test_accuracy = Accuracy(task="binary")

        # --- Output Directory for Test Predictions ---
        self.output_dir = Path().resolve().parent / "output" / "predictions_resnet_test"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Test predictions will be saved to: {self.output_dir}")


    def forward(self, x):
        if x.ndim == 4 and x.shape[1] != 7:
             x = x.permute(0, 3, 1, 2)
        features = self.encoder(x)
        logits = self.decoder(features)
        output = F.interpolate(logits, size=self.target_size, mode='bilinear', align_corners=False)
        return output

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        preds_prob = torch.sigmoid(logits)
        preds_binary = (preds_prob > 0.5)
        iou = self.val_iou(preds_binary, masks.int()) # Ensure masks are int for metric

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # No need to return loss explicitly unless used by callbacks

    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)

        # --- Process Predictions ---
        preds_prob = torch.sigmoid(logits)
        preds_binary = (preds_prob > 0.5) # Shape: (N, 1, H, W)

        # --- Calculate Metrics ---
        iou = self.test_iou(preds_binary, masks.int())
        acc = self.test_accuracy(preds_binary, masks.int())

        # --- Log Metrics ---
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('test_iou', iou, on_step=False, on_epoch=True, logger=True)
        self.log('test_accuracy', acc, on_step=False, on_epoch=True, logger=True)

        # --- Save Predictions ---
        preds_to_save = preds_binary.cpu().numpy().astype(np.uint8) # (N, 1, H, W)

        for i in range(preds_to_save.shape[0]): # Iterate through batch
            pred_array = preds_to_save[i].squeeze() # Remove channel dim -> (H, W)
            # Construct filename (using batch_idx and index within batch)
            # TODO: Ideally, pass original filenames through dataset/loader for better naming
            filename = f"prediction_batch{batch_idx}_idx{i}.tif"
            save_path = self.output_dir / filename
            try:
                tiff.imwrite(save_path, pred_array)
            except Exception as e:
                print(f"Error saving prediction {filename}: {e}")

        # Return metrics dictionary (optional)
        return {'test_loss': loss, 'test_iou': iou, 'test_accuracy': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate) # Use hparams
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class KelpDataset(torch.utils.data.Dataset):
    def __init__(self, satellite_paths: List[str], mask_paths: List[str]):
        self.satellite_paths = satellite_paths
        self.mask_paths = mask_paths
        if len(satellite_paths) != len(mask_paths):
             raise ValueError("Satellite and mask path lists must have the same length.")
        if not satellite_paths:
             raise ValueError("Satellite path list is empty.")

    def __len__(self):
        return len(self.satellite_paths)

    def load_image(self, filename: str):
        try:
            img = tiff.imread(filename)
            if img is None:
                 raise IOError(f"tifffile returned None for {filename}")
            return img
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            raise

    def __getitem__(self, idx):
        sat_path = self.satellite_paths[idx]
        mask_path = self.mask_paths[idx]

        sat_img = self.load_image(sat_path)
        if sat_img.shape[-1] != 7:
             raise ValueError(f"Satellite image {sat_path} has {sat_img.shape[-1]} channels, expected 7.")
        sat_img = sat_img.astype('float32') / 255.0 # Basic normalization

        mask = self.load_image(mask_path)
        mask = mask.astype('float32')

        sat_tensor = torch.from_numpy(sat_img).permute(2, 0, 1) # C, H, W
        mask_tensor = torch.from_numpy(mask).unsqueeze(0) # 1, H, W

        if mask_tensor.shape[0] != 1:
             raise ValueError(f"Mask {mask_path} has unexpected channel dimension: {mask_tensor.shape}")

        return sat_tensor, mask_tensor

# Removed the separate predict_and_save function

# Usage example:
def main():
    pl.seed_everything(42)

    # Prepare data
    try:
        filenames = prepare_filenames("tile")
        # --- Make sure all 6 lists are unpacked ---
        train_sat, train_mask, val_sat, val_mask, test_sat, test_mask = filenames
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR during data preparation: {e}")
        return

    # Create datasets and loaders
    print("Creating Datasets...")
    try:
        train_dataset = KelpDataset(train_sat, train_mask)
        val_dataset = KelpDataset(val_sat, val_mask)
        test_dataset = KelpDataset(test_sat, test_mask) # Use the test split
    except ValueError as e:
         print(f"ERROR creating dataset: {e}")
         return

    print("Creating DataLoaders...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=4, # Often use larger batch for val/test
        pin_memory=True, persistent_workers=True
    )
    # --- Create the Test DataLoader ---
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True
    )

    # Initialize model and trainer
    print("Initializing Model and Trainer...")
    model = KelpSegmentationModel(learning_rate=1e-4) # Pass LR

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss', dirpath='checkpoints_resnet/',
        filename='kelp-resnet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3, mode='min',
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=30, # Adjust as needed
        log_every_n_steps=10,
        callbacks=[checkpoint_callback]
        # limit_train_batches=0.1, # Use fraction for percentage
        # limit_val_batches=0.1,
        # limit_test_batches=0.1, # Limit test batches if needed for quick checks
    )

    # Train
    print("Starting Training...")
    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
         print(f"ERROR during training: {e}")
         return

    # Test (this will automatically call test_step and save predictions)
    print("Starting Testing (predictions will be saved)...")
    # Load the best checkpoint before testing
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from: {best_model_path}")
        trainer.test(model, dataloaders=test_loader, ckpt_path=best_model_path)
    else:
        print("No best model checkpoint found. Testing with last model state.")
        trainer.test(model, dataloaders=test_loader) # Use the test_loader

    print("Testing finished. Predictions saved.")

if __name__ == "__main__":
    main()