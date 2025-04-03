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
from torchmetrics import Accuracy, JaccardIndex

dir_root = here()
sys.path.append(str(dir_root))
# Assuming get_data.py contains prepare_filenames
# from utils.get_data import prepare_filenames # Make sure this import works

# --- Dummy prepare_filenames for testing if needed ---
def prepare_filenames(data_type="tile"):
    print(f"Using dummy prepare_filenames for type: {data_type}")
    # Replace with your actual data loading logic
    root = Path().resolve().parent / "data" / "balanced_tiled_40_60" # Example path
    train_kelp_dir = root / "train_kelp"
    train_sat_dir = root / "train_satellite"

    # Get base filenames (without _kelp.tif or _satellite.tif)
    base_filenames = [p.stem.replace("_kelp", "") for p in train_kelp_dir.glob("*_kelp.tif")]
    if not base_filenames:
        raise FileNotFoundError(f"No kelp files found in {train_kelp_dir}")

    # Split (simple example, use proper splitting)
    num_files = len(base_filenames)
    train_split = int(0.7 * num_files)
    val_split = int(0.15 * num_files)

    train_base = base_filenames[:train_split]
    val_base = base_filenames[train_split:train_split + val_split]
    test_base = base_filenames[train_split + val_split:]

    # Construct full paths
    train_sat_paths = [str(train_sat_dir / f"{f}_satellite.tif") for f in train_base]
    train_mask_paths = [str(train_kelp_dir / f"{f}_kelp.tif") for f in train_base]
    val_sat_paths = [str(train_sat_dir / f"{f}_satellite.tif") for f in val_base]
    val_mask_paths = [str(train_kelp_dir / f"{f}_kelp.tif") for f in val_base]
    test_sat_paths = [str(train_sat_dir / f"{f}_satellite.tif") for f in test_base]
    test_mask_paths = [str(train_kelp_dir / f"{f}_kelp.tif") for f in test_base]

    return train_sat_paths, train_mask_paths, val_sat_paths, val_mask_paths, test_sat_paths, test_mask_paths
# --- End Dummy ---


# --- KelpSegmentationModel (Changes in forward, steps) ---
class KelpSegmentationModel(pl.LightningModule):
    # --- REMOVED target_size from init ---
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        # self.target_size = target_size # REMOVED - Output size will match mask size
        # Encoder
        base_encoder = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Adjust first layer for 7 input channels
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
            nn.Conv2d(32, 1, kernel_size=1) # Final layer outputs logits
        )
        self.criterion = nn.BCEWithLogitsLoss() # Use this loss as it includes sigmoid

        self.val_iou = JaccardIndex(task="binary")
        self.test_iou = JaccardIndex(task="binary")
        self.test_accuracy = Accuracy(task="binary")

        self.output_dir = Path().resolve().parent / "output" / "predictions_resnet_test"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Test predictions will be saved to: {self.output_dir}")


    def forward(self, x):
        if x.ndim == 4 and x.shape[1] != 7:
             x = x.permute(0, 3, 1, 2)
        elif x.ndim != 4 or x.shape[1] != 7:
             raise ValueError(f"Unexpected input shape: {x.shape}. Expected (N, 7, H, W)")

        features = self.encoder(x)
        logits = self.decoder(features)
        # --- REMOVED the final F.interpolate ---
        # output = F.interpolate(logits, size=self.target_size, mode='bilinear', align_corners=False)
        return logits # Return raw logits from the decoder

    # --- Helper to resize logits to mask size ---
    def _resize_logits_to_mask(self, logits, masks):
        """Resizes logits to match the spatial dimensions of the masks."""
        if logits.shape[-2:] != masks.shape[-2:]:
            return F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        return logits

    def training_step(self, batch, batch_idx):
        images, masks, _ = batch
        current_batch_size = images.size(0)
        raw_logits = self(images) # Get raw logits from forward pass

        # --- Resize logits to match mask size BEFORE loss calculation ---
        logits = self._resize_logits_to_mask(raw_logits, masks)

        loss = self.criterion(logits, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size = current_batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks, _ = batch
        current_batch_size = images.size(0)
        raw_logits = self(images)

        # --- Resize logits to match mask size BEFORE loss/metrics ---
        logits = self._resize_logits_to_mask(raw_logits, masks)

        loss = self.criterion(logits, masks)
        preds_prob = torch.sigmoid(logits) # Apply sigmoid here for metrics
        preds_binary = (preds_prob > 0.5)
        iou = self.val_iou(preds_binary, masks.int())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = current_batch_size)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = current_batch_size)

    def test_step(self, batch, batch_idx):
        images, masks, original_filenames = batch
        current_batch_size = images.size(0)
        raw_logits = self(images)

        # --- Resize logits to match mask size BEFORE loss/metrics/saving ---
        logits = self._resize_logits_to_mask(raw_logits, masks)

        loss = self.criterion(logits, masks)
        preds_prob = torch.sigmoid(logits) # Apply sigmoid here
        preds_binary = (preds_prob > 0.5)

        iou = self.test_iou(preds_binary, masks.int())
        acc = self.test_accuracy(preds_binary, masks.int())

        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, batch_size = current_batch_size)
        self.log('test_iou', iou, on_step=False, on_epoch=True, logger=True, batch_size = current_batch_size)
        self.log('test_accuracy', acc, on_step=False, on_epoch=True, logger=True, batch_size = current_batch_size)

        # --- Save the correctly sized binary predictions ---
        preds_to_save = preds_binary.cpu().numpy().astype(np.uint8)

        for i in range(preds_to_save.shape[0]):
            # --- Squeeze should now result in (H, W) e.g., (25, 25) ---
            pred_array = preds_to_save[i].squeeze()
            if pred_array.ndim != 2:
                 print(f"Warning: Squeezed prediction has unexpected dimensions: {pred_array.shape}. Skipping save for this item.")
                 continue

            original_fname = original_filenames[i]
            filename = f"prediction_{original_fname}.tif"
            save_path = self.output_dir / filename
            try:
                tiff.imwrite(save_path, pred_array)
            except Exception as e:
                print(f"Error saving prediction {filename}: {e}")

        return {'test_loss': loss, 'test_iou': iou, 'test_accuracy': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


# --- KelpDataset (No changes needed here if it loads 25x25 correctly) ---
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

        try:
            sat_img = self.load_image(sat_path)
            if sat_img.ndim == 3 and sat_img.shape[-1] == 7: # H, W, C
                 pass # Expected format
            elif sat_img.ndim == 2: # Grayscale? Or single channel mask loaded as sat?
                 raise ValueError(f"Satellite image {sat_path} appears to be 2D (grayscale?). Expected 7 channels.")
            else:
                 raise ValueError(f"Satellite image {sat_path} has unexpected shape: {sat_img.shape}. Expected (H, W, 7).")

            sat_img = sat_img.astype('float32')
            # Per-channel normalization (more robust than simple min-max)
            min_vals = sat_img.min(axis=(0, 1), keepdims=True)
            max_vals = sat_img.max(axis=(0, 1), keepdims=True)
            sat_img = (sat_img - min_vals) / (max_vals - min_vals + 1e-6)

            mask = self.load_image(mask_path)
            if mask.ndim != 2:
                 raise ValueError(f"Mask {mask_path} has unexpected shape: {mask.shape}. Expected (H, W).")
            mask = mask.astype('float32')

            sat_tensor = torch.from_numpy(sat_img).permute(2, 0, 1) # C, H, W
            mask_tensor = torch.from_numpy(mask).unsqueeze(0) # 1, H, W

            base_filename = Path(sat_path).stem.replace("_satellite", "")

            return sat_tensor, mask_tensor, base_filename

        except Exception as e:
            print(f"Error in __getitem__ for index {idx}, paths: {sat_path}, {mask_path}: {e}")
            raise


# --- main function (No significant changes needed here) ---
def main():
    pl.seed_everything(42)

    # Prepare data
    try:
        train_sat_paths, train_mask_paths, val_sat_paths, val_mask_paths, test_sat_paths, test_mask_paths = prepare_filenames("tile")
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR during data preparation: {e}")
        return

    # Create datasets and loaders
    print("Creating Datasets...")
    try:
        train_dataset = KelpDataset(train_sat_paths, train_mask_paths)
        val_dataset = KelpDataset(val_sat_paths, val_mask_paths)
        test_dataset = KelpDataset(test_sat_paths, test_mask_paths)
    except ValueError as e:
         print(f"ERROR creating dataset: {e}")
         return

    print("Creating DataLoaders...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True
    )

    # Initialize model and trainer
    print("Initializing Model and Trainer...")
    model = KelpSegmentationModel(learning_rate=1e-4)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_iou',
        dirpath='checkpoints_resnet/',
        filename='kelp-resnet-{epoch:02d}-{val_iou:.3f}',
        save_top_k=3,
        mode='max',
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=10,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, lr_monitor],
        # limit_train_batches=0.1, # Uncomment for faster debugging
        # limit_val_batches=0.1,
        # limit_test_batches=0.1,
    )

    # Train
    print("Starting Training...")
    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
         print(f"ERROR during training: {e}")
         import traceback
         traceback.print_exc()
         return

    # Test
    print("Starting Testing (predictions will be saved)...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and Path(best_model_path).exists():
        print(f"Loading best model from: {best_model_path}")
        trainer.test(model, dataloaders=test_loader, ckpt_path=best_model_path)
    else:
        print("No best model checkpoint found or path invalid. Testing with last model state.")
        trainer.test(model, dataloaders=test_loader)

    print("Testing finished. Predictions saved.")

if __name__ == "__main__":
    main()