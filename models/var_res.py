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
    # --- CHOOSE YOUR DATASET PATH HERE ---
    # dataset_name = "balanced_tiled_40_60" # Example for 25x25
    dataset_name = "" # Example for 350x350 (assuming you have this)
    # dataset_name = "some_other_50x50_dataset" # Example for 50x50
    # ---

    root = Path().resolve().parent / "data" / dataset_name
    print(f"Attempting to load data from: {root}")

    # Adjust subdirectories based on your actual structure
    # Assuming structure like: data/dataset_name/train_kelp and data/dataset_name/train_satellite
    train_kelp_dir = root / "train_kelp"
    train_sat_dir = root / "train_satellite"

    if not train_kelp_dir.exists():
         raise FileNotFoundError(f"Kelp directory not found: {train_kelp_dir}")
    if not train_sat_dir.exists():
         raise FileNotFoundError(f"Satellite directory not found: {train_sat_dir}")


    # Get base filenames (without _kelp.tif or _satellite.tif)
    # Use kelp dir as the source of truth for filenames
    base_filenames = [p.stem.replace("_kelp", "") for p in train_kelp_dir.glob("*_kelp.tif")]
    if not base_filenames:
        raise FileNotFoundError(f"No kelp files found in {train_kelp_dir}")
    print(f"Found {len(base_filenames)} base filenames.")

    # Split (simple example, use proper splitting like sklearn's train_test_split)
    num_files = len(base_filenames)
    train_split = int(0.7 * num_files)
    val_split = int(0.15 * num_files)

    # Ensure splits don't create empty lists if dataset is very small
    train_base = base_filenames[:train_split]
    val_base = base_filenames[train_split:min(train_split + val_split, num_files)]
    test_base = base_filenames[min(train_split + val_split, num_files):]

    if not train_base or not val_base or not test_base:
        print("Warning: One or more data splits (train/val/test) are empty. Check dataset size and split ratios.")

    # Construct full paths
    train_sat_paths = [str(train_sat_dir / f"{f}_satellite.tif") for f in train_base]
    train_mask_paths = [str(train_kelp_dir / f"{f}_kelp.tif") for f in train_base]
    val_sat_paths = [str(train_sat_dir / f"{f}_satellite.tif") for f in val_base]
    val_mask_paths = [str(train_kelp_dir / f"{f}_kelp.tif") for f in val_base]
    test_sat_paths = [str(train_sat_dir / f"{f}_satellite.tif") for f in test_base]
    test_mask_paths = [str(train_kelp_dir / f"{f}_kelp.tif") for f in test_base]

    print(f"Splits: Train={len(train_sat_paths)}, Val={len(val_sat_paths)}, Test={len(test_sat_paths)}")

    return train_sat_paths, train_mask_paths, val_sat_paths, val_mask_paths, test_sat_paths, test_mask_paths
# --- End Dummy ---


# --- KelpSegmentationModel (No changes needed here for dynamic size) ---
class KelpSegmentationModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
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

        self.val_iou = JaccardIndex(task="binary")
        self.test_iou = JaccardIndex(task="binary")
        self.test_accuracy = Accuracy(task="binary")

        self.output_dir = Path().resolve().parent / "output" / "predictions_resnet_dynamic" # Changed output dir name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Test predictions will be saved to: {self.output_dir}")


    def forward(self, x):
        if x.ndim == 4 and x.shape[1] != 7:
             x = x.permute(0, 3, 1, 2)
        elif x.ndim != 4 or x.shape[1] != 7:
             raise ValueError(f"Unexpected input shape: {x.shape}. Expected (N, 7, H, W)")

        features = self.encoder(x)
        logits = self.decoder(features)
        return logits # Return raw logits

    # Helper remains the same - it's already dynamic
    def _resize_logits_to_mask(self, logits, masks):
        """Resizes logits to match the spatial dimensions of the masks."""
        if logits.shape[-2:] != masks.shape[-2:]:
            return F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        return logits

    def training_step(self, batch, batch_idx):
        images, masks, _ = batch
        current_batch_size = images.size(0)
        raw_logits = self(images)
        logits = self._resize_logits_to_mask(raw_logits, masks) # Resize dynamically
        loss = self.criterion(logits, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size = current_batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks, _ = batch
        current_batch_size = images.size(0)
        raw_logits = self(images)
        logits = self._resize_logits_to_mask(raw_logits, masks) # Resize dynamically
        loss = self.criterion(logits, masks)
        preds_prob = torch.sigmoid(logits)
        preds_binary = (preds_prob > 0.5)
        iou = self.val_iou(preds_binary, masks.int())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = current_batch_size)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = current_batch_size)

    def test_step(self, batch, batch_idx):
        images, masks, original_filenames = batch
        current_batch_size = images.size(0)
        raw_logits = self(images)
        logits = self._resize_logits_to_mask(raw_logits, masks) # Resize dynamically
        loss = self.criterion(logits, masks)
        preds_prob = torch.sigmoid(logits)
        preds_binary = (preds_prob > 0.5)
        iou = self.test_iou(preds_binary, masks.int())
        acc = self.test_accuracy(preds_binary, masks.int())
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, batch_size = current_batch_size)
        self.log('test_iou', iou, on_step=False, on_epoch=True, logger=True, batch_size = current_batch_size)
        self.log('test_accuracy', acc, on_step=False, on_epoch=True, logger=True, batch_size = current_batch_size)

        preds_to_save = preds_binary.cpu().numpy().astype(np.uint8)
        for i in range(preds_to_save.shape[0]):
            pred_array = preds_to_save[i].squeeze()
            if pred_array.ndim != 2:
                 print(f"Warning: Squeezed prediction has unexpected dimensions: {pred_array.shape}. Skipping save.")
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


# --- KelpDataset (Changes in __init__ to get size) ---
class KelpDataset(torch.utils.data.Dataset):
    def __init__(self, satellite_paths: List[str], mask_paths: List[str]):
        self.satellite_paths = satellite_paths
        self.mask_paths = mask_paths
        self.height = None # Initialize height
        self.width = None  # Initialize width

        if len(satellite_paths) != len(mask_paths):
             raise ValueError("Satellite and mask path lists must have the same length.")
        if not satellite_paths:
             raise ValueError("Satellite path list is empty.")

        # --- Determine image size from the first image ---
        try:
            print(f"Determining image size from: {self.satellite_paths[0]}")
            temp_img = self.load_image(self.satellite_paths[0])
            if temp_img.ndim == 3 and temp_img.shape[-1] == 7: # H, W, C
                self.height = temp_img.shape[0]
                self.width = temp_img.shape[1]
                print(f"Detected image size: {self.height} x {self.width}")
            else:
                 raise ValueError(f"First image {self.satellite_paths[0]} has unexpected shape: {temp_img.shape}")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load first image to determine size: {e}")
            raise # Stop execution if size cannot be determined

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
            # Basic shape check based on determined size (optional but good)
            if sat_img.shape[0] != self.height or sat_img.shape[1] != self.width:
                 print(f"Warning: Image {sat_path} has size {sat_img.shape[:2]}, expected {self.height}x{self.width}. Check dataset consistency.")
                 # You might want to resize here if inconsistency is expected, or raise an error
                 # For now, we'll proceed, assuming the model handles it, but log a warning.

            if sat_img.ndim != 3 or sat_img.shape[-1] != 7:
                 raise ValueError(f"Satellite image {sat_path} has unexpected shape: {sat_img.shape}. Expected ({self.height}, {self.width}, 7).")

            sat_img = sat_img.astype('float32')
            min_vals = sat_img.min(axis=(0, 1), keepdims=True)
            max_vals = sat_img.max(axis=(0, 1), keepdims=True)
            sat_img = (sat_img - min_vals) / (max_vals - min_vals + 1e-6) # Per-channel norm

            mask = self.load_image(mask_path)
            if mask.ndim != 2 or mask.shape[0] != self.height or mask.shape[1] != self.width:
                 raise ValueError(f"Mask {mask_path} has unexpected shape: {mask.shape}. Expected ({self.height}, {self.width}).")
            mask = mask.astype('float32')

            sat_tensor = torch.from_numpy(sat_img).permute(2, 0, 1) # C, H, W
            mask_tensor = torch.from_numpy(mask).unsqueeze(0) # 1, H, W

            base_filename = Path(sat_path).stem.replace("_satellite", "")

            return sat_tensor, mask_tensor, base_filename

        except Exception as e:
            print(f"Error in __getitem__ for index {idx}, paths: {sat_path}, {mask_path}: {e}")
            raise


# --- main function (Added print for detected size) ---
def main():
    pl.seed_everything(42)

    # Prepare data
    try:
        # --- Make sure prepare_filenames points to the desired dataset ---
        train_sat_paths, train_mask_paths, val_sat_paths, val_mask_paths, test_sat_paths, test_mask_paths = prepare_filenames("tile")
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR during data preparation: {e}")
        return

    # Create datasets and loaders
    print("Creating Datasets...")
    try:
        # Create train_dataset first to get the size
        train_dataset = KelpDataset(train_sat_paths, train_mask_paths)
        # --- Print the detected size ---
        if train_dataset.height is not None:
            print(f"--- Running with Image Size: {train_dataset.height} x {train_dataset.width} ---")
        else:
            print("--- ERROR: Could not determine image size from training data. ---")
            return # Stop if size wasn't determined

        val_dataset = KelpDataset(val_sat_paths, val_mask_paths)
        test_dataset = KelpDataset(test_sat_paths, test_mask_paths)
    except ValueError as e:
         print(f"ERROR creating dataset: {e}")
         return
    except Exception as e: # Catch other potential errors during dataset init
         print(f"Unexpected ERROR creating dataset: {e}")
         import traceback
         traceback.print_exc()
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
    model = KelpSegmentationModel(learning_rate=1e-4) # Model doesn't need the size passed

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_iou',
        dirpath='checkpoints_resnet/',
        filename='kelp-resnet-dynamic-{epoch:02d}-{val_iou:.3f}', # Changed filename slightly
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