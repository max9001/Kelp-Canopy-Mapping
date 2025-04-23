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
# Import specific backbones and weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from torchmetrics import Accuracy, JaccardIndex
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Configuration Constants ---
BACKBONE = "resnet50"
MAX_EPOCHS = 20 
RUN_NAME = "50bb_20_test"

# .pth file
WEIGHTS_FILENAME = "best_weights.pth" # Keep original name
# .ckpt file
CHECKPOINT_FILENAME = "checkpoint" # Keep original name

RUN_DIR = Path().resolve().parent / "runs" / RUN_NAME # Keep original name
# --- End Configuration Constants ---


# --- Imports for Utils and psutil ---
try:
    # Assumes script is in 'models' and utils is sibling to 'models' parent
    UTILS_DIR = Path().resolve().parent.parent / "utils"
    if not UTILS_DIR.exists():
        # Assumes script and utils are siblings
        UTILS_DIR = Path().resolve().parent / "utils"
        if not UTILS_DIR.exists(): raise FileNotFoundError("Could not find utils directory")
    sys.path.append(str(UTILS_DIR.parent)) # Add project root to path
    from utils.get_data import prepare_filenames
except ImportError:
    print("Error: Could not import prepare_filenames from utils. Ensure utils directory is accessible.")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"Error finding utils directory: {e}")
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("Warning: psutil not found. Defaulting num_workers. Install with 'pip install psutil'")
    psutil = None
# --- End Imports ---


# --- Model Definition (KelpSegmentationModel - Modified for Backbone Choice) ---
class KelpSegmentationModel(pl.LightningModule):
    def __init__(self,
                 target_size=(350, 350),
                 learning_rate=1e-4,
                 backbone_name="resnet18" # <-- ADD parameter, default to resnet18
                ):
        super().__init__()
        # Save hyperparameters (includes backbone_name, lr, etc.)
        self.save_hyperparameters()
        self.target_size = target_size

        # --- Select Backbone and Weights ---
        if self.hparams.backbone_name == "resnet18": # Access saved hyperparam
            base_encoder = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
            encoder_out_channels = 512
        elif self.hparams.backbone_name == "resnet34":
            base_encoder = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
            encoder_out_channels = 512
        elif self.hparams.backbone_name == "resnet50":
            base_encoder = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
            encoder_out_channels = 2048 # ResNet50 has different output size
        else:
            raise ValueError(f"Unsupported backbone: {self.hparams.backbone_name}.")

        print(f"Using backbone: {self.hparams.backbone_name} with {encoder_out_channels} output channels.")

        # --- Modify input layer for 7 channels ---
        base_encoder.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # --- Define the Encoder sequence ---
        self.encoder = nn.Sequential(
            base_encoder.conv1, base_encoder.bn1, base_encoder.relu, base_encoder.maxpool,
            base_encoder.layer1, base_encoder.layer2, base_encoder.layer3, base_encoder.layer4
        )

        # --- Adjust the FIRST layer of the Decoder based on encoder output ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_out_channels, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1) # Final output layer
        )

        # --- Loss and Metrics ---
        self.criterion = nn.BCEWithLogitsLoss()
        self.val_iou = JaccardIndex(task="binary")


    def forward(self, x):
        # Ensure input is (B, C, H, W) - Important check before encoder
        if x.ndim != 4 or x.shape[1] != 7:
             if x.ndim == 4 and x.shape[3] == 7:
                  # print(f"Warning: Input tensor shape {x.shape} -> permuting to (B, C, H, W).")
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
        # Access learning_rate via self.hparams
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


# --- Dataset Definition (Modified KelpDataset for Augmentations) ---
class KelpDataset(torch.utils.data.Dataset):
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

    def __len__(self):
        return len(self.satellite_paths)

    def load_image(self, filename: str, is_mask: bool = False):
        # Keep existing load_image logic
        try:
            img = tiff.imread(filename)
            if img is None: raise IOError(f"tifffile returned None for {filename}")
            if not is_mask: # Satellite Image
                if img.dtype != np.float32:
                    img = img.astype(np.float32)
                if img.ndim != 3 or img.shape[-1] != 7:
                    raise ValueError(f"Satellite shape: {img.shape}. Expected (H, W, 7)")
            else: # Mask
                 if img.ndim == 3: img = img.squeeze()
                 if img.dtype != np.uint8 or img.max() > 1:
                      img = (img > 0).astype(np.uint8)
            return img
        except Exception as e:
            print(f"CRITICAL ERROR loading {filename}: {str(e)}")
            raise

    def __getitem__(self, idx):
        sat_path = self.satellite_paths[idx]
        mask_path = self.mask_paths[idx]
        try:
            sat_img_np = self.load_image(sat_path, is_mask=False)
            mask_np = self.load_image(mask_path, is_mask=True)

            # --- Apply Albumentations ---
            if self.transforms:
                augmented = self.transforms(image=sat_img_np, mask=mask_np)
                sat_img_np = augmented['image']
                mask_np = augmented['mask']

            # --- Convert to Tensor ---
            sat_tensor = torch.from_numpy(sat_img_np.transpose(2, 0, 1))
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()

            # --- Apply Noise ---
            if self.apply_noise:
                noise = torch.randn_like(sat_tensor[:5, :, :]) * self.noise_level
                sat_tensor[:5, :, :] = sat_tensor[:5, :, :] + noise

            return sat_tensor, mask_tensor
        except Exception as e:
            print(f"Error processing index {idx} (Sat: {Path(sat_path).name}): {e}")
            raise


def main():
    pl.seed_everything(42)

    chosen_backbone = BACKBONE

    # --- Prepare Filenames ---
    try:
        print("Preparing filenames...")
        filenames = prepare_filenames()
        train_sat, train_mask, val_sat, val_mask, _, _ = filenames
        print("Filenames prepared.")
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR during data preparation: {e}"); return

    # --- Define Augmentation Pipeline ---
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])
    val_transforms = None

    # --- Create Datasets with Transforms ---
    print("Creating Datasets...")
    try:
        train_dataset = KelpDataset(
            train_sat, train_mask,
            transforms=train_transforms, apply_noise=True, noise_level=0.01
        )
        val_dataset = KelpDataset(
            val_sat, val_mask,
            transforms=val_transforms, apply_noise=False
        )
    except ValueError as e:
         print(f"ERROR creating dataset: {e}"); return

    # --- Create RUN_DIR (using constant name) ---
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Run artifacts will be saved in: {RUN_DIR}")

    # --- Create DataLoaders ---
    print("Creating DataLoaders...")
    TRAIN_BATCH_SIZE = 8  # Adjust based on GPU memory for chosen backbone
    VAL_BATCH_SIZE = 16
    NUM_WORKERS = psutil.cpu_count(logical=False)//2 if psutil else 0

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False
    )

    # --- Initialize Model (passing chosen backbone) ---
    print("Initializing Model...")
    model = KelpSegmentationModel(
        target_size=(350, 350),
        learning_rate=1e-4,
        backbone_name=chosen_backbone # <-- Pass the chosen name
    )

    # --- Checkpoint Callback (using constant filenames) ---
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=RUN_DIR,
        filename=f'{CHECKPOINT_FILENAME}-{{epoch:02d}}-{{val_loss:.4f}}', # Use constant base name
        save_top_k=1, mode='min', save_last=False, verbose=True
    )

    # --- Trainer ---
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
         print(f"ERROR during training: {e}")
         # Add OOM handling if needed
         return

    # --- Save the Best Model's Weights (using constant filename) ---
    print("\nSaving the best model weights...")
    best_checkpoint_path = checkpoint_callback.best_model_path
    if best_checkpoint_path and Path(best_checkpoint_path).exists():
        print(f"Best checkpoint found at: {best_checkpoint_path}")
        try:
            # Load model using the checkpoint method, which handles hyperparameters like backbone_name
            # No need to pass backbone_name explicitly if saved with save_hyperparameters
            best_model = KelpSegmentationModel.load_from_checkpoint(best_checkpoint_path)
            weights_save_path = RUN_DIR / WEIGHTS_FILENAME # Save weights in RUN_DIR
            torch.save(best_model.state_dict(), weights_save_path)
            print(f"Successfully saved best model weights to: {weights_save_path}")
        except Exception as e:
            print(f"ERROR saving best model weights: {e}")
            # ... (error handling) ...
    else:
        print("WARNING: No best model checkpoint path found.")
        # ... (warning handling) ...


if __name__ == "__main__":
    main()