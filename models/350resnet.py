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
from utils.get_data import prepare_filenames # Assuming this points to your data prep script


class KelpSegmentationModel(pl.LightningModule):
    # ============================================================================
    # CHANGE 1: Update default target_size in __init__
    # ============================================================================
    def __init__(self, target_size=(350, 350), learning_rate=1e-4): # Changed default
        super().__init__()
        self.save_hyperparameters()
        self.target_size = target_size # Store the target size
        # Encoder (No change needed)
        base_encoder = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        base_encoder.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(
            base_encoder.conv1, base_encoder.bn1, base_encoder.relu, base_encoder.maxpool,
            base_encoder.layer1, base_encoder.layer2, base_encoder.layer3, base_encoder.layer4
        )
        # Decoder (No change needed - final size handled by interpolate)
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

        self.output_dir = Path().resolve().parent / "output" / "350_resnet_test" # Updated output dir name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Test predictions will be saved to: {self.output_dir}")


    def forward(self, x):
        if x.ndim == 4 and x.shape[1] != 7:
             x = x.permute(0, 3, 1, 2)
        features = self.encoder(x)
        logits = self.decoder(features)
        # ============================================================================
        # CHANGE 2: Interpolate uses self.target_size (which is now 350x350)
        # No code change needed here, as it already uses self.target_size
        # ============================================================================
        output = F.interpolate(logits, size=self.target_size, mode='bilinear', align_corners=False)
        return output

    # training_step, validation_step, test_step, configure_optimizers remain unchanged
    # They operate on the output of forward(), which is now correctly sized.

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images) # Output is already interpolated to target_size
        loss = self.criterion(logits, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images) # Output is already interpolated to target_size
        loss = self.criterion(logits, masks)
        preds_prob = torch.sigmoid(logits)
        preds_binary = (preds_prob > 0.5)
        iou = self.val_iou(preds_binary, masks.int())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images) # Output is already interpolated to target_size
        loss = self.criterion(logits, masks)
        preds_prob = torch.sigmoid(logits)
        preds_binary = (preds_prob > 0.5)
        iou = self.test_iou(preds_binary, masks.int())
        acc = self.test_accuracy(preds_binary, masks.int())
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('test_iou', iou, on_step=False, on_epoch=True, logger=True)
        self.log('test_accuracy', acc, on_step=False, on_epoch=True, logger=True)

        preds_to_save = preds_binary.cpu().numpy().astype(np.uint8)
        # TODO: Pass original filenames if needed for better saving names
        for i in range(preds_to_save.shape[0]):
            pred_array = preds_to_save[i].squeeze()
            filename = f"prediction_batch{batch_idx}_idx{i}.tif"
            save_path = self.output_dir / filename
            try:
                tiff.imwrite(save_path, pred_array)
            except Exception as e:
                print(f"Error saving prediction {filename}: {e}")
        return {'test_loss': loss, 'test_iou': iou, 'test_accuracy': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# KelpDataset remains unchanged - it loads images as they are.
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
        # Basic check - remove if causing issues, but good for sanity
        if sat_img.ndim != 3 or sat_img.shape[-1] != 7:
             raise ValueError(f"Satellite image {sat_path} has unexpected shape: {sat_img.shape}. Expected (H, W, 7).")
        sat_img = sat_img.astype('float32') / 255.0 # Basic normalization

        mask = self.load_image(mask_path)
        # Basic check
        if mask.ndim != 2:
             raise ValueError(f"Mask {mask_path} has unexpected shape: {mask.shape}. Expected (H, W).")
        mask = mask.astype('float32')

        sat_tensor = torch.from_numpy(sat_img).permute(2, 0, 1) # C, H, W
        mask_tensor = torch.from_numpy(mask).unsqueeze(0) # 1, H, W

        if mask_tensor.shape[0] != 1:
             raise ValueError(f"Mask {mask_path} has unexpected channel dimension: {mask_tensor.shape}")

        return sat_tensor, mask_tensor


def main():
    pl.seed_everything(42)

    # Prepare data - ENSURE prepare_filenames points to 350x350 data
    try:
        # ============================================================================
        # CHANGE 3: Ensure prepare_filenames is configured for 350x350 data
        # (This change is *inside* the prepare_filenames function itself)
        # ============================================================================
        filenames = prepare_filenames("original") # Or whatever type your 350x350 data is
        train_sat, train_mask, val_sat, val_mask, test_sat, test_mask = filenames
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR during data preparation: {e}")
        return

    # Create datasets and loaders (No changes needed here)
    print("Creating Datasets...")
    try:
        train_dataset = KelpDataset(train_sat, train_mask)
        val_dataset = KelpDataset(val_sat, val_mask)
        test_dataset = KelpDataset(test_sat, test_mask)
    except ValueError as e:
         print(f"ERROR creating dataset: {e}")
         return

    print("Creating DataLoaders...")
    # ============================================================================
    # POTENTIAL CHANGE 4: Reduce batch_size if you get Out-of-Memory errors
    # ============================================================================
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=4, # Reduced batch_size
        pin_memory=True, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=4, # Reduced batch_size
        pin_memory=True, persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4, # Reduced batch_size
        pin_memory=True, persistent_workers=True
    )

    # Initialize model and trainer
    print("Initializing Model and Trainer...")
    # ============================================================================
    # CHANGE 5: Explicitly pass target_size (or rely on new default)
    # ============================================================================
    model = KelpSegmentationModel(target_size=(350, 350), learning_rate=1e-4)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss', dirpath='checkpoints_350_test/', # Updated dir name
        filename='kelp-resnet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3, mode='min',
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=150, # Adjust as needed
        log_every_n_steps=10,
        # callbacks=[checkpoint_callback]
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        # limit_test_batches=0.1,
    )

    # Train
    print("Starting Training...")
    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
         print(f"ERROR during training: {e}")
         # Consider adding more specific error handling for CUDA OOM
         if "CUDA out of memory" in str(e):
             print("***** CUDA Out of Memory! Try reducing batch_size further. *****")
         return

    # Test
    print("Starting Testing (predictions will be saved)...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and Path(best_model_path).exists(): # Check path exists
        print(f"Loading best model from: {best_model_path}")
        trainer.test(model, dataloaders=test_loader, ckpt_path=best_model_path)
    else:
        print("No best model checkpoint found or path invalid. Testing with last model state.")
        trainer.test(model, dataloaders=test_loader)

    print("Testing finished. Predictions saved.")

if __name__ == "__main__":
    main()