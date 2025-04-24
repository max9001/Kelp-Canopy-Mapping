import pytorch_lightning as pl
import torch
import torch.nn as nn
import tifffile as tiff
import torchvision
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path
import os
import sys
# Import specific backbones and weights
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights
import torch.nn.functional as F
from torchmetrics import JaccardIndex
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
# Import Callback base class and specific callbacks used
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

# --- Configuration Constants ---
BACKBONE = "resnet50" # Options: "resnet18", "resnet34", "resnet50"
MAX_EPOCHS = 50     # Total epochs for cosine annealing cycle (can be stopped early)
RUN_NAME = "50_changelr_test" # Updated run name

WEIGHTS_FILENAME = "best_weights.pth"
CHECKPOINT_BASENAME = "checkpoint"

RUN_DIR = Path().resolve().parent / "runs" / RUN_NAME

# --- Scheduling and Stopping Parameters ---
EARLY_STOPPING_PATIENCE = 10 # Increase patience slightly for gradual decay?
EARLY_STOPPING_MIN_DELTA = 0.001 # Maybe loosen min_delta if expecting slower final improvements
COSINE_ETA_MIN = 1e-7 # Minimum LR for Cosine Annealing
# --- End Configuration Constants ---


# --- Imports ---
try:
    UTILS_DIR = Path().resolve().parent.parent / "utils"
    if not UTILS_DIR.exists():
        UTILS_DIR = Path().resolve().parent / "utils"
        if not UTILS_DIR.exists(): raise FileNotFoundError("Could not find utils directory")
    sys.path.append(str(UTILS_DIR.parent))
    from utils.get_data import prepare_filenames
except ImportError: print("Error: Could not import prepare_filenames."); sys.exit(1)
except FileNotFoundError as e: print(f"Error finding utils directory: {e}"); sys.exit(1)

try: import psutil
except ImportError: print("Warning: psutil not found."); psutil = None
# --- End Imports ---


# --- Learning Rate Monitor Callback Definition (Unchanged) ---
class LearningRateMonitor(Callback):
    """Logs the learning rate and prints a prominent message when it changes."""
    def __init__(self):
        super().__init__()
        self.last_lr = {}
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.optimizers: return
        for i, optimizer in enumerate(trainer.optimizers):
            if not optimizer.param_groups: continue
            current_lr = optimizer.param_groups[0]['lr']
            optimizer_key = f"optimizer_{i}_group_0"
            last_lr_group = self.last_lr.get(optimizer_key, None)
            pl_module.log(f"lr-{optimizer_key}", current_lr, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            if last_lr_group is not None and not np.isclose(current_lr, last_lr_group, rtol=1e-6, atol=1e-9):
                print("\n" + "="*60)
                print(f"Epoch {trainer.current_epoch}: Learning rate for optimizer {i} changed from {last_lr_group:.4e} to {current_lr:.4e}")
                print("="*60 + "\n")
            self.last_lr[optimizer_key] = current_lr
# --- End Callback Definition ---


# --- Model Definition (KelpSegmentationModel - Unchanged Internally) ---
class KelpSegmentationModel(pl.LightningModule):
    def __init__(self, target_size=(350, 350), learning_rate=1e-4, backbone_name="resnet18"):
        super().__init__()
        self.save_hyperparameters()
        self.target_size = target_size
        if self.hparams.backbone_name == "resnet18":
            base_encoder = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
            encoder_out_channels = 512
        elif self.hparams.backbone_name == "resnet34":
            base_encoder = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
            encoder_out_channels = 512
        elif self.hparams.backbone_name == "resnet50":
            base_encoder = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
            encoder_out_channels = 2048
        else: raise ValueError(f"Unsupported backbone: {self.hparams.backbone_name}.")
        print(f"Using backbone: {self.hparams.backbone_name} with {encoder_out_channels} output channels.")
        base_encoder.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(
            base_encoder.conv1, base_encoder.bn1, base_encoder.relu, base_encoder.maxpool,
            base_encoder.layer1, base_encoder.layer2, base_encoder.layer3, base_encoder.layer4
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_out_channels, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
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

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 7:
             if x.ndim == 4 and x.shape[3] == 7: x = x.permute(0, 3, 1, 2)
             else: raise ValueError(f"Invalid input tensor shape: {x.shape}. Expected (B, 7, H, W).")
        features = self.encoder(x)
        logits = self.decoder(features)
        output = F.interpolate(logits, size=self.target_size, mode='bilinear', align_corners=False)
        return output

    def training_step(self, batch, batch_idx):
        images, masks = batch; logits = self(images); loss = self.criterion(logits, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch; logits = self(images); loss = self.criterion(logits, masks)
        preds_prob = torch.sigmoid(logits); preds_binary = (preds_prob > 0.5)
        iou = self.val_iou(preds_binary, masks.int())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # ============================================================================
    # MODIFIED: Use CosineAnnealingLR scheduler
    # ============================================================================
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # Use CosineAnnealingLR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS,  # Number of epochs for one cycle (usually total epochs)
            eta_min=COSINE_ETA_MIN # Minimum learning rate
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # monitor is NOT needed for CosineAnnealingLR as it steps automatically
                "interval": "epoch", # Step the scheduler every epoch
                "frequency": 1,
            }
        }
    # ============================================================================


# --- Dataset Definition (KelpDataset - Unchanged) ---
class KelpDataset(torch.utils.data.Dataset):
    # ... (Keep KelpDataset class exactly as before) ...
    def __init__(self, satellite_paths: List[str], mask_paths: List[str],
                 transforms: Optional[A.Compose] = None, apply_noise: bool = False, noise_level: float = 0.01):
        self.satellite_paths = satellite_paths; self.mask_paths = mask_paths
        self.transforms = transforms; self.apply_noise = apply_noise; self.noise_level = noise_level
        if len(satellite_paths) != len(mask_paths): raise ValueError("Path lists length mismatch.")
        if not satellite_paths: raise ValueError("Satellite path list empty.")

    def __len__(self): return len(self.satellite_paths)

    def load_image(self, filename: str, is_mask: bool = False):
        try:
            img = tiff.imread(filename)
            if img is None: raise IOError(f"tifffile returned None: {filename}")
            if not is_mask:
                if img.dtype != np.float32: img = img.astype(np.float32)
                if img.ndim != 3 or img.shape[-1] != 7: raise ValueError(f"Sat shape: {img.shape}")
            else:
                 if img.ndim == 3: img = img.squeeze()
                 if img.dtype != np.uint8 or img.max() > 1: img = (img > 0).astype(np.uint8)
            return img
        except Exception as e: print(f"ERROR loading {filename}: {str(e)}"); raise

    def __getitem__(self, idx):
        sat_path = self.satellite_paths[idx]; mask_path = self.mask_paths[idx]
        try:
            sat_img_np = self.load_image(sat_path, is_mask=False)
            mask_np = self.load_image(mask_path, is_mask=True)
            if self.transforms:
                augmented = self.transforms(image=sat_img_np, mask=mask_np)
                sat_img_np, mask_np = augmented['image'], augmented['mask']
            sat_tensor = torch.from_numpy(sat_img_np.transpose(2, 0, 1))
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
            if self.apply_noise:
                noise = torch.randn_like(sat_tensor[:5, :, :]) * self.noise_level
                sat_tensor[:5, :, :] = sat_tensor[:5, :, :] + noise
            return sat_tensor, mask_tensor
        except Exception as e: print(f"Error processing idx {idx}: {e}"); raise


def main():
    pl.seed_everything(42)
    chosen_backbone = BACKBONE

    # --- Prepare Filenames ---
    try:
        print("Preparing filenames...")
        filenames = prepare_filenames()
        train_sat, train_mask, val_sat, val_mask, _, _ = filenames
        print("Filenames prepared.")
    except (ValueError, FileNotFoundError) as e: print(f"ERROR: {e}"); return

    # --- Define Augmentations ---
    train_transforms = A.Compose([ A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5) ])
    val_transforms = None

    # --- Create Datasets ---
    print("Creating Datasets...")
    try:
        train_dataset = KelpDataset(train_sat, train_mask, transforms=train_transforms, apply_noise=True, noise_level=0.01)
        val_dataset = KelpDataset(val_sat, val_mask, transforms=val_transforms, apply_noise=False)
    except ValueError as e: print(f"ERROR: {e}"); return

    # --- Checkpoint Resuming Logic (Unchanged) ---
    resume_checkpoint_path = None
    if RUN_DIR.exists():
        potential_checkpoints = list(RUN_DIR.glob('*.ckpt'))
        if potential_checkpoints:
            if len(potential_checkpoints) > 1: warnings.warn(f"Multiple checkpoints found. Using: {potential_checkpoints[0]}")
            resume_checkpoint_path = potential_checkpoints[0]
            print(f"Run directory exists. Resuming from: {resume_checkpoint_path}")
        else: print(f"Run directory {RUN_DIR} exists, but no checkpoint. Starting fresh.")
    else: print(f"Run directory {RUN_DIR} does not exist. Starting fresh."); RUN_DIR.mkdir(parents=True, exist_ok=True)

    # --- Create DataLoaders ---
    print("Creating DataLoaders...")
    TRAIN_BATCH_SIZE = 8; VAL_BATCH_SIZE = 16
    NUM_WORKERS = psutil.cpu_count(logical=False)//2 if psutil else 0
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False
    )

    # --- Initialize Model ---
    print("Initializing Model...")
    model = KelpSegmentationModel(
        target_size=(350, 350), learning_rate=1e-4, backbone_name=chosen_backbone
    )

    # --- Callbacks ---
    print("Configuring Callbacks...")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', dirpath=RUN_DIR,
        filename=f'{CHECKPOINT_BASENAME}-{{epoch:02d}}-{{val_loss:.4f}}',
        save_top_k=1, mode='min', save_last=False, verbose=True
    )
    # EarlyStopping setup remains the same
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', mode='min',
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        verbose=True
    )
    # LR Monitor callback remains useful for visualizing the cosine decay
    lr_monitor_callback = LearningRateMonitor()


    # --- Trainer ---
    print("Initializing Trainer...")
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, max_epochs=MAX_EPOCHS, log_every_n_steps=20,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor_callback], # Keep all callbacks
        precision="16-mixed" if torch.cuda.is_available() else 32
    )

    # --- Train the Model ---
    print(f"Starting Training (max_epochs={MAX_EPOCHS}, Cosine LR Decay, EarlyStopping patience={EARLY_STOPPING_PATIENCE})...")
    try:
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_checkpoint_path)
        print("Training finished.")
        # Check trainer state for stopped epoch
        stopped_epoch = trainer.state.epoch
        if stopped_epoch is not None and stopped_epoch < (trainer.max_epochs - 1):
             print(f"Training stopped before max_epochs. Early stopping likely triggered around epoch {stopped_epoch}.")

    except Exception as e: print(f"ERROR during training: {e}"); return

    # --- Save the Best Model's Weights (Unchanged) ---
    print("\nSaving the best model weights...")
    best_checkpoint_path_after_train = checkpoint_callback.best_model_path
    if best_checkpoint_path_after_train and Path(best_checkpoint_path_after_train).exists():
        print(f"Best checkpoint recorded: {best_checkpoint_path_after_train}")
        try:
            best_model = KelpSegmentationModel.load_from_checkpoint(best_checkpoint_path_after_train)
            weights_save_path = RUN_DIR / WEIGHTS_FILENAME
            torch.save(best_model.state_dict(), weights_save_path)
            print(f"Successfully saved best model weights to: {weights_save_path}")
        except Exception as e: print(f"ERROR saving weights: {e}")
    else: print("WARNING: No best checkpoint path found. Weights not saved.")

if __name__ == "__main__":
    main()