import pytorch_lightning as pl
import torch
import torch.nn as nn
import tifffile as tiff
import torchvision
from typing import List, Tuple, Optional # Added Optional
import numpy as np
from pathlib import Path
# from pyprojroot import here # Using Path().resolve() instead for simplicity
import os
import sys
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
from torchmetrics import Accuracy, JaccardIndex
import albumentations as A # <-- Import Albumentations
from albumentations.pytorch import ToTensorV2 # <-- Use ToTensorV2 from Albumentations

# .pth file
WEIGHTS_FILENAME = "best_weights.pth" # <-- Updated filename

# .ckpt file
CHECKPOINT_FILENAME = "checkpoint" # <-- Updated filename

# where it will be saved
RUN_DIR = Path().resolve().parent / "runs" / "normalized_augmented_120_test" # <-- Updated run name

MAX_EPOCHS = 120 # Adjust as needed

# --- Imports for Utils and psutil ---
try:
    UTILS_DIR = Path().resolve().parent.parent / "utils"
    if not UTILS_DIR.exists():
        UTILS_DIR = Path().resolve().parent / "utils"
        if not UTILS_DIR.exists(): raise FileNotFoundError("Could not find utils directory")
    sys.path.append(str(UTILS_DIR.parent))
    from utils.get_data import prepare_filenames
except ImportError:
    print("Error: Could not import prepare_filenames from utils.")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"Error finding utils directory: {e}")
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("Warning: psutil not found. Defaulting num_workers.")
    psutil = None
# --- End Imports ---


# --- Model Definition (KelpSegmentationModel - No changes needed here) ---
class KelpSegmentationModel(pl.LightningModule):
    # ... (Keep the KelpSegmentationModel class exactly as before) ...
    def __init__(self, target_size=(350, 350), learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
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
            nn.Conv2d(32, 1, kernel_size=1) # Output 1 channel (logits for binary)
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.val_iou = JaccardIndex(task="binary")

    def forward(self, x):
        # Ensure input is (B, C, H, W) - Important check before encoder
        if x.ndim != 4 or x.shape[1] != 7:
             # Attempt to fix if it looks like (B, H, W, C)
             if x.ndim == 4 and x.shape[3] == 7:
                  print(f"Warning: Input tensor shape {x.shape} -> permuting to (B, C, H, W).")
                  x = x.permute(0, 3, 1, 2)
             else:
                 raise ValueError(f"Invalid input tensor shape: {x.shape}. Expected (B, 7, H, W).")

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
        iou = self.val_iou(preds_binary, masks.int())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


# --- Dataset Definition (Modified KelpDataset) ---
class KelpDataset(torch.utils.data.Dataset):
    # ============================================================================
    # CHANGE 1: Modify __init__ to accept transforms
    # ============================================================================
    def __init__(self,
                 satellite_paths: List[str],
                 mask_paths: List[str],
                 transforms: Optional[A.Compose] = None, # Accept optional transforms
                 apply_noise: bool = False, # Flag to control noise addition
                 noise_level: float = 0.01 # Standard deviation for noise
                ):
        self.satellite_paths = satellite_paths
        self.mask_paths = mask_paths
        self.transforms = transforms # Store transforms
        self.apply_noise = apply_noise
        self.noise_level = noise_level

        if len(satellite_paths) != len(mask_paths):
             raise ValueError("Satellite and mask path lists must have the same length.")
        if not satellite_paths:
             raise ValueError("Satellite path list is empty.")
        # Be less verbose during init unless debugging
        # print(f"Initialized dataset with {len(satellite_paths)} samples. Transforms: {'Yes' if transforms else 'No'}, Noise: {'Yes' if apply_noise else 'No'}")

    def __len__(self):
        return len(self.satellite_paths)

    def load_image(self, filename: str, is_mask: bool = False):
        # Keep your existing load_image logic that matches your data format
        # Ensure it returns a NumPy array
        try:
            img = tiff.imread(filename)
            if img is None: raise IOError(f"tifffile returned None for {filename}")

            if not is_mask: # Satellite Image
                # Expects (H, W, C) float32 from normalization script
                if img.dtype != np.float32:
                    # This might indicate an issue with the normalization script or loading wrong data
                    print(f"Warning: Satellite image {Path(filename).name} dtype is {img.dtype}, expected float32. Attempting conversion.")
                    img = img.astype(np.float32)
                if img.ndim != 3 or img.shape[-1] != 7:
                    raise ValueError(f"Satellite image {Path(filename).name} loaded with wrong shape: {img.shape}. Expected (H, W, 7)")
            else: # Mask
                 if img.ndim == 3: img = img.squeeze() # Remove channel if present
                 if img.dtype != np.uint8 or img.max() > 1:
                      # print(f"Debug: Mask {Path(filename).name} max value is {img.max()}. Converting to binary uint8.")
                      img = (img > 0).astype(np.uint8) # Ensure binary 0 or 1
                 # Albumentations expects mask as (H, W)
            return img
        except Exception as e:
            print(f"CRITICAL ERROR loading {filename}: {str(e)}")
            raise

    # ============================================================================
    # CHANGE 2: Modify __getitem__ to apply transforms and noise
    # ============================================================================
    def __getitem__(self, idx):
        sat_path = self.satellite_paths[idx]
        mask_path = self.mask_paths[idx]

        try:
            # Load images as NumPy arrays (H, W, C for sat, H, W for mask)
            sat_img_np = self.load_image(sat_path, is_mask=False)
            mask_np = self.load_image(mask_path, is_mask=True)

            # --- Apply Albumentations (if specified) ---
            if self.transforms:
                augmented = self.transforms(image=sat_img_np, mask=mask_np)
                sat_img_np = augmented['image']
                mask_np = augmented['mask']

            # --- Convert to Tensor (C, H, W) ---
            # Transpose satellite image HWC -> CHW
            sat_tensor = torch.from_numpy(sat_img_np.transpose(2, 0, 1))
            # Add channel dim to mask H, W -> 1, H, W and ensure type
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()

            # --- Apply Noise (if specified) ---
            # Apply noise *after* conversion to tensor and *after* other transforms
            if self.apply_noise:
                # Add Gaussian noise only to the first 5 bands (0-4)
                noise = torch.randn_like(sat_tensor[:5, :, :]) * self.noise_level
                sat_tensor[:5, :, :] = sat_tensor[:5, :, :] + noise
                # Optional: Clip values if noise pushes them outside expected range (e.g., if using [0,1] normalization)
                # sat_tensor = torch.clamp(sat_tensor, 0, 1) # Example if normalized [0,1]

            # Final shape check (optional but good)
            if sat_tensor.shape[0] != 7: raise ValueError("Sat tensor wrong channels")
            if mask_tensor.shape[0] != 1: raise ValueError("Mask tensor wrong channels")

            return sat_tensor, mask_tensor

        except Exception as e:
            print(f"Error processing index {idx} (Sat: {Path(sat_path).name}): {e}")
            raise


def main():
    pl.seed_everything(42)

    # --- Prepare Filenames ---
    try:
        print("Preparing filenames...")
        filenames = prepare_filenames()
        train_sat, train_mask, val_sat, val_mask, _, _ = filenames # Don't need test set here
        print("Filenames prepared.")
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR during data preparation: {e}"); return

    # ============================================================================
    # CHANGE 3: Define Augmentation Pipeline
    # ============================================================================
    # Define transforms for training data
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5), # Rotates 90, 180, 270 degrees
        # Note: Noise is handled separately in __getitem__ for band selectivity
    ])

    # No random geometric augmentations for validation
    val_transforms = None # Or A.Compose([ A.Resize(...) ]) if needed

    # --- Create Datasets with Transforms ---
    # ============================================================================
    # CHANGE 4: Instantiate datasets with appropriate transforms/noise flags
    # ============================================================================
    print("Creating Datasets...")
    try:
        train_dataset = KelpDataset(
            train_sat, train_mask,
            transforms=train_transforms, # Apply train transforms
            apply_noise=True,            # Add noise for training
            noise_level=0.01             # Set noise level
        )
        val_dataset = KelpDataset(
            val_sat, val_mask,
            transforms=val_transforms,   # No random transforms for validation
            apply_noise=False            # No noise for validation
        )
    except ValueError as e:
         print(f"ERROR creating dataset: {e}"); return

    # --- Create RUN_DIR ---
    RUN_DIR.mkdir(parents=True, exist_ok=True) # Ensures directory exists
    print(f"Run artifacts will be saved in: {RUN_DIR}")

    # --- Create DataLoaders (No changes needed here) ---
    print("Creating DataLoaders...")
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 16
    NUM_WORKERS = psutil.cpu_count(logical=False)//2 if psutil else 0
    # NUM_WORKERS = 0

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False,
        drop_last=True # Helps if last batch size causes issues with BN
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False
    )

    # --- Initialize Model (No changes needed here) ---
    print("Initializing Model...")
    model = KelpSegmentationModel(target_size=(350, 350), learning_rate=1e-4)

    # --- Checkpoint Callback (Using updated filenames) ---
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=RUN_DIR,
        filename=f'{CHECKPOINT_FILENAME}-{{epoch:02d}}-{{val_loss:.4f}}',
        save_top_k=1, mode='min', save_last=False, verbose=True
    )

    # --- Trainer (No changes needed here) ---
    print("Initializing Trainer...")
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, max_epochs=MAX_EPOCHS, log_every_n_steps=20,
        callbacks=[checkpoint_callback],
        precision="16-mixed" if torch.cuda.is_available() else 32
    )

    # --- Train the Model ---
    print("Starting Training...")
    try:
        trainer.fit(model, train_loader, val_loader)
        print("Training finished.")
    except Exception as e:
         print(f"ERROR during training: {e}"); # Add OOM handling if needed
         return

    # --- Save the Best Model's Weights (Using updated filenames) ---
    print("\nSaving the best model weights...")
    best_checkpoint_path = checkpoint_callback.best_model_path
    if best_checkpoint_path and Path(best_checkpoint_path).exists():
        print(f"Best checkpoint found at: {best_checkpoint_path}")
        try:
            best_model = KelpSegmentationModel.load_from_checkpoint(best_checkpoint_path)
            weights_save_path = RUN_DIR / WEIGHTS_FILENAME # Save weights in RUN_DIR
            torch.save(best_model.state_dict(), weights_save_path)
            print(f"Successfully saved best model weights to: {weights_save_path}")
        except Exception as e:
            print(f"ERROR saving best model weights: {e}")
    else:
        print("WARNING: No best model checkpoint path found.")

if __name__ == "__main__":
    main()