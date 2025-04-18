import pytorch_lightning as pl
import torch
import torch.nn as nn
import tifffile as tiff
import torchvision
from typing import List, Tuple
import numpy as np
from pathlib import Path
# from pyprojroot import here # Using Path().resolve() instead for simplicity
import os
import sys
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
# from tqdm import tqdm # Not strictly needed for this script's core logic anymore

# --- Metrics ---
from torchmetrics import Accuracy, JaccardIndex

# Assuming utils is in the parent directory or PYTHONPATH is set correctly
# Adjust relative path if needed
try:
    # Assumes script is in 'models' and utils is sibling to 'models' parent
    UTILS_DIR = Path().resolve().parent.parent / "utils"
    if not UTILS_DIR.exists():
        # Assumes script and utils are siblings
        UTILS_DIR = Path().resolve().parent / "utils"
        if not UTILS_DIR.exists():
             raise FileNotFoundError("Could not find utils directory")
    sys.path.append(str(UTILS_DIR.parent)) # Add project root to path
    from utils.get_data import prepare_filenames
except ImportError:
    print("Error: Could not import prepare_filenames from utils. Ensure utils directory is accessible.")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"Error finding utils directory: {e}")
    sys.exit(1)


class KelpSegmentationModel(pl.LightningModule):
    def __init__(self, target_size=(350, 350), learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.target_size = target_size
        # Encoder
        base_encoder = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Modify input layer for 7 channels
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

        # Metrics
        self.val_iou = JaccardIndex(task="binary")
        # Keep test metrics here if you want to see them during validation epoch end (optional)
        # self.test_iou = JaccardIndex(task="binary")
        # self.test_accuracy = Accuracy(task="binary")

        # ============================================================================
        # REMOVED test_step output saving - will be done in test.py
        # ============================================================================
        # self.output_dir = Path().resolve().parent / "output" / "350_resnet_test"
        # self.output_dir.mkdir(parents=True, exist_ok=True)
        # print(f"Test predictions (if run here) would be saved to: {self.output_dir}")


    def forward(self, x):
        # Ensure input is (B, C, H, W)
        if x.ndim == 4 and x.shape[1] != 7:
             print(f"Warning: Input tensor has shape {x.shape}, attempting permute from likely (B, H, W, C)")
             try:
                 x = x.permute(0, 3, 1, 2)
             except RuntimeError as e:
                 print(f"Error permuting input tensor: {e}. Ensure input is (B, C, H, W) or (B, H, W, C).")
                 raise
        elif x.ndim != 4:
             print(f"Error: Input tensor has unexpected dimensions: {x.ndim}. Expected 4.")
             raise ValueError("Input tensor must be 4D (B, C, H, W)")
        if x.shape[1] != 7:
             print(f"Error: Input tensor has {x.shape[1]} channels. Expected 7.")
             raise ValueError("Input tensor must have 7 channels")

        features = self.encoder(x)
        logits = self.decoder(features)
        # Interpolate to the target size defined in __init__
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
        preds_binary = (preds_prob > 0.5) # Use a threshold (e.g., 0.5)
        iou = self.val_iou(preds_binary, masks.int()) # Ensure mask is int

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss # Return loss is standard practice

    # ============================================================================
    # REMOVED test_step - will be handled in test.py
    # ============================================================================
    # def test_step(self, batch, batch_idx):
    #     # ... (logic removed) ...
    #     pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # Example scheduler (optional)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        # Simpler alternative:
        # return optimizer

# KelpDataset (Ensure this matches your data loading needs)
class KelpDataset(torch.utils.data.Dataset):
    def __init__(self, satellite_paths: List[str], mask_paths: List[str]):
        self.satellite_paths = satellite_paths
        self.mask_paths = mask_paths
        if len(satellite_paths) != len(mask_paths):
             raise ValueError("Satellite and mask path lists must have the same length.")
        if not satellite_paths:
             raise ValueError("Satellite path list is empty.")
        print(f"Initialized dataset with {len(satellite_paths)} samples.") # Added print

    def __len__(self):
        return len(self.satellite_paths)

    def load_image(self, filename: str, is_mask: bool = False):
        try:
            img = tiff.imread(filename)
            if img is None:
                 raise IOError(f"tifffile returned None for {filename}")

            # ============================================================================
            # DATA LOADING & PREPROCESSING - NEEDS TO MATCH NORMALIZATION SCRIPT
            # ============================================================================
            # Option 1: Assuming data was already normalized (e.g., Z-score) and saved as float32
            if not is_mask:
                if img.dtype != np.float32:
                    print(f"Warning: Satellite image {Path(filename).name} dtype is {img.dtype}, expected float32 (normalized). Converting.")
                    img = img.astype(np.float32)
                # If Z-score was used, no division by 255 needed here.
                # Add any other necessary preprocessing *after* loading, if not done offline.

            # Option 2: Basic Normalization (if data is uint8/uint16 and NOT pre-normalized)
            # elif not is_mask:
            #     if img.dtype == np.uint16:
            #         img = img.astype('float32') / 65535.0
            #     elif img.dtype == np.uint8:
            #         img = img.astype('float32') / 255.0
            #     else:
            #         # Assume it might be pre-normalized float or handle other types
            #         img = img.astype('float32')

            # Masks should usually be float32 in range [0, 1] for BCEWithLogitsLoss
            elif is_mask:
                 if img.ndim == 3: # Handle masks saved with channel dim
                      img = img.squeeze()
                 if img.dtype != np.uint8 and img.max() > 1: # Check if it's not already 0/1
                      print(f"Warning: Mask {Path(filename).name} max value is {img.max()}. Converting to 0/1 uint8.")
                      img = (img > 0).astype(np.uint8) # Convert to binary 0 or 1

                 img = img.astype('float32') # Convert to float for loss calculation

            return img
        except Exception as e:
            print(f"CRITICAL ERROR loading {filename}: {str(e)}")
            # Optionally, return None or a placeholder, but raising is better to stop errors early
            raise

    def __getitem__(self, idx):
        sat_path = self.satellite_paths[idx]
        mask_path = self.mask_paths[idx]

        try:
            sat_img = self.load_image(sat_path, is_mask=False)
            # Expect (H, W, C) from tiff.imread usually
            if sat_img.ndim != 3 or sat_img.shape[-1] != 7:
                 raise ValueError(f"Satellite image {Path(sat_path).name} has unexpected shape: {sat_img.shape}. Expected (H, W, 7).")

            mask = self.load_image(mask_path, is_mask=True)
            if mask.ndim != 2:
                 # Handle masks potentially saved with a channel dim
                 if mask.ndim == 3 and mask.shape[0] == 1:
                     mask = mask.squeeze(0)
                 elif mask.ndim == 3 and mask.shape[-1] == 1:
                     mask = mask.squeeze(-1)
                 else:
                     raise ValueError(f"Mask {Path(mask_path).name} has unexpected shape: {mask.shape}. Expected (H, W) or squeezable to it.")

            # Ensure mask has same H, W as satellite image
            if sat_img.shape[0] != mask.shape[0] or sat_img.shape[1] != mask.shape[1]:
                raise ValueError(f"Satellite ({sat_img.shape[:2]}) and Mask ({mask.shape}) dimensions mismatch for index {idx}.")


            # Convert to tensors: Satellite (C, H, W), Mask (1, H, W)
            sat_tensor = torch.from_numpy(sat_img).permute(2, 0, 1) # H, W, C -> C, H, W
            mask_tensor = torch.from_numpy(mask).unsqueeze(0) # H, W -> 1, H, W

            return sat_tensor, mask_tensor

        except Exception as e:
            print(f"Error processing index {idx} (Sat: {Path(sat_path).name}, Mask: {Path(mask_path).name}): {e}")
            # Return None or re-raise; returning None requires collate_fn handling
            raise


def main():
    pl.seed_everything(42)

    # Prepare data filenames
    try:
        print("Preparing filenames...")
        # This function should now load paths from the pre-split directories
        filenames = prepare_filenames()
        train_sat, train_mask, val_sat, val_mask, test_sat, test_mask = filenames
        print("Filenames prepared.")
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR during data preparation: {e}")
        return

    # Create datasets
    print("Creating Datasets...")
    try:
        train_dataset = KelpDataset(train_sat, train_mask)
        val_dataset = KelpDataset(val_sat, val_mask)
        # We don't need the test_dataset here for training
        # test_dataset = KelpDataset(test_sat, test_mask)
    except ValueError as e:
         print(f"ERROR creating dataset: {e}")
         return

    # Create DataLoaders
    print("Creating DataLoaders...")
    # Reduce batch_size if OOM errors occur
    # Adjust num_workers based on your system
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 16 # Validation can often use larger batches
    NUM_WORKERS = psutil.cpu_count(logical=False)//2 # A common starting point

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False
    )

    # Initialize model
    print("Initializing Model...")
    model = KelpSegmentationModel(target_size=(350, 350), learning_rate=1e-4)

    # --- Checkpoint Callback ---
    # Define where checkpoints will be saved
    CHECKPOINT_DIR = Path("./350x350_200_resnet").resolve() # Save in current dir subfolder
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    print(f"Checkpoints will be saved in: {CHECKPOINT_DIR}")

    # Configure the checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',           # Metric to monitor
        dirpath=CHECKPOINT_DIR,       # Directory to save checkpoints
        filename='kelp-resnet-best-{epoch:02d}-{val_loss:.4f}', # Filename pattern
        save_top_k=1,                 # Save only the best model
        mode='min',                   # Mode for the monitored metric ('min' for loss/error)
        save_last=False,              # Optionally save the last epoch checkpoint too
        verbose=True                  # Print messages when checkpoints are saved
    )

    # --- Trainer ---
    print("Initializing Trainer...")
    MAX_EPOCHS = 200 # Adjust as needed
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, # Or specify multiple GPUs [0, 1]
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=20, # Log more frequently if steps are fast
        callbacks=[checkpoint_callback], # ***** ADD THE CALLBACK *****
        precision="16-mixed" if torch.cuda.is_available() else 32 # Use mixed precision on GPU
        # Add other trainer flags as needed (e.g., gradient clipping)
        # limit_train_batches=0.1, # For debugging: Use a fraction of data
        # limit_val_batches=0.1,
    )

    # --- Train the Model ---
    print("Starting Training...")
    try:
        trainer.fit(model, train_loader, val_loader)
        print("Training finished.")
    except Exception as e:
         print(f"ERROR during training: {e}")
         if torch.cuda.is_available() and "CUDA out of memory" in str(e):
             print("\n ***** CUDA Out of Memory! ***** ")
             print(f"    Try reducing TRAIN_BATCH_SIZE (currently {TRAIN_BATCH_SIZE})")
             print(f"    or VAL_BATCH_SIZE (currently {VAL_BATCH_SIZE}).")
             print(f"    Using '16-mixed' precision can also help.")
         return # Stop script on training error

    # --- Save the Best Model's Weights ---
    print("\nSaving the best model weights...")
    best_checkpoint_path = checkpoint_callback.best_model_path

    if best_checkpoint_path and Path(best_checkpoint_path).exists():
        print(f"Best checkpoint found at: {best_checkpoint_path}")
        try:
            # Load the model state from the best checkpoint
            # We need to instantiate the model class first to load state dict into it
            best_model = KelpSegmentationModel.load_from_checkpoint(best_checkpoint_path)

            # Define path for saving only the weights (.pth)
            WEIGHTS_SAVE_DIR = Path("./saved_weights").resolve()
            WEIGHTS_SAVE_DIR.mkdir(exist_ok=True)
            weights_save_path = WEIGHTS_SAVE_DIR / "kelp_resnet_350_best_weights.pth"

            # Save the state dictionary
            torch.save(best_model.state_dict(), weights_save_path)
            print(f"Successfully saved best model weights to: {weights_save_path}")

        except Exception as e:
            print(f"ERROR saving best model weights: {e}")
            print("Model weights might not have been saved. You can still use the full checkpoint:")
            print(f"  {best_checkpoint_path}")
    else:
        print("WARNING: No best model checkpoint path found. Weights not saved separately.")
        print("         This might happen if validation didn't run or improve.")

    # ============================================================================
    # REMOVED testing block - will be in test.py
    # ============================================================================
    # print("Starting Testing (predictions will be saved)...")
    # # ... (testing logic removed) ...
    # print("Testing finished. Predictions saved.")


if __name__ == "__main__":
    # Added psutil import for better num_workers default
    try:
        import psutil
    except ImportError:
        print("Warning: psutil not found. Defaulting num_workers. Install with 'pip install psutil'")
        psutil = None # Assign None if not available
        # Set a fallback number of workers if psutil is not installed
        NUM_WORKERS = 2 if os.cpu_count() > 2 else 0

    main()