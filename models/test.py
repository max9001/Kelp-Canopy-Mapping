import torch
import torch.nn as nn
import pytorch_lightning as pl
import tifffile
import numpy as np
from pathlib import Path
import sys
# import argparse # Removed argparse
from tqdm import tqdm
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
from typing import List, Tuple, Dict # Added Dict for config type hint

# --- Metrics ---
from torchmetrics import JaccardIndex, Precision, Recall, F1Score

# --- Configuration Constants ---
# *** SET THESE VALUES BEFORE RUNNING ***

# Path to the trained model weights (.pth file)
RUN = "normalized_20_test"
################################################


WEIGHTS_PATH_STR = Path().resolve().parent / "runs" / RUN / "best_weights.pth"

# Path to the base directory containing the pre-split data folders
# (test_sat, test_gt, etc.)
DATA_DIR_STR = Path().resolve().parent / "data" / "cleaned"

# Directory to save the predicted masks
OUTPUT_DIR_STR = Path().resolve().parent / "output" / RUN

# Target image size (height and width) the model was trained for
IMG_SIZE = 350

# Batch size for testing
BATCH_SIZE = 16

# Number of workers for DataLoader
NUM_WORKERS = 4 # Adjust based on your system

# Probability threshold for converting logits to binary predictions
THRESHOLD = 0.5

# Random seed for reproducibility
SEED = 42

# --- End Configuration ---


# --- Import necessary components (Ensure these paths are correct) ---
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


# --- Model and Dataset Class Definitions ---
# !! IMPORTANT !! These MUST match the definitions used during training
class KelpSegmentationModel(pl.LightningModule):
    def __init__(self, target_size=(350, 350), learning_rate=1e-4):
        super().__init__()
        self.target_size = target_size
        base_encoder = torchvision.models.resnet18(weights=None)
        base_encoder.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(
            base_encoder.conv1, base_encoder.bn1, base_encoder.relu, base_encoder.maxpool,
            base_encoder.layer1, base_encoder.layer2, base_encoder.layer3, base_encoder.layer4
        )
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

    def forward(self, x):
        if x.ndim == 4 and x.shape[1] != 7: x = x.permute(0, 3, 1, 2)
        elif x.ndim != 4 or x.shape[1] != 7: raise ValueError(f"Invalid input shape: {x.shape}")
        features = self.encoder(x)
        logits = self.decoder(features)
        output = F.interpolate(logits, size=self.target_size, mode='bilinear', align_corners=False)
        return output

class KelpDataset(torch.utils.data.Dataset):
    def __init__(self, satellite_paths: List[str], mask_paths: List[str], return_filename=False):
        self.satellite_paths = satellite_paths
        self.mask_paths = mask_paths
        self.return_filename = return_filename
        if len(satellite_paths) != len(mask_paths): raise ValueError("Path lists must have same length.")
        if not satellite_paths: raise ValueError("Satellite path list empty.")
        # print(f"Initialized dataset with {len(satellite_paths)} samples.") # Less verbose

    def __len__(self): return len(self.satellite_paths)

    def load_image(self, filename: str, is_mask: bool = False):
        try:
            img = tifffile.imread(filename)
            if img is None: raise IOError(f"tifffile returned None for {filename}")
            if not is_mask:
                if img.dtype != np.float32: img = img.astype(np.float32)
            elif is_mask:
                 if img.ndim == 3: img = img.squeeze()
                 if img.dtype != np.uint8 or img.max() > 1: img = (img > 0).astype(np.uint8)
                 img = img.astype('float32')
            return img
        except Exception as e:
            print(f"CRITICAL ERROR loading {filename}: {str(e)}"); raise

    def __getitem__(self, idx):
        sat_path = self.satellite_paths[idx]
        mask_path = self.mask_paths[idx]
        filename_stem = Path(sat_path).stem.replace("_satellite", "")
        try:
            sat_img = self.load_image(sat_path, is_mask=False)
            if sat_img.ndim != 3 or sat_img.shape[-1] != 7: raise ValueError(f"Sat shape {sat_img.shape}")
            mask = self.load_image(mask_path, is_mask=True)
            if mask.ndim != 2:
                 if mask.ndim == 3 and (mask.shape[0] == 1 or mask.shape[-1] == 1): mask = mask.squeeze()
                 else: raise ValueError(f"Mask shape {mask.shape}")
            if sat_img.shape[0] != mask.shape[0] or sat_img.shape[1] != mask.shape[1]:
                raise ValueError(f"Dims mismatch Sat ({sat_img.shape[:2]}) vs Mask ({mask.shape})")
            sat_tensor = torch.from_numpy(sat_img).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            if self.return_filename: return sat_tensor, mask_tensor, filename_stem
            else: return sat_tensor, mask_tensor
        except Exception as e:
            print(f"Error processing idx {idx} (Sat: {Path(sat_path).name}): {e}"); raise
# --- End Class Definitions ---


def run_testing(config: Dict):
    """Runs inference and evaluation using parameters from the config dict."""
    pl.seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert string paths from config to Path objects
    weights_path = Path(config["weights_path_str"]).resolve()
    data_dir = Path(config["data_dir_str"]).resolve()
    output_dir = Path(config["output_dir_str"]).resolve()

    # --- Load Data ---
    try:
        print(f"Loading test filenames from base directory: {data_dir}")
        filenames = prepare_filenames(base_dir=data_dir)
        test_sat_paths = filenames[4]
        test_gt_paths = filenames[5]
        if not test_sat_paths or not test_gt_paths:
             raise ValueError("Test set paths are empty.")
        print(f"Found {len(test_sat_paths)} test samples.")
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR during data preparation: {e}"); return

    # --- Create Dataset and DataLoader ---
    try:
        test_dataset = KelpDataset(test_sat_paths, test_gt_paths, return_filename=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True
        )
    except ValueError as e:
        print(f"ERROR creating dataset/loader: {e}"); return

    # --- Initialize Model ---
    print("Initializing model...")
    model = KelpSegmentationModel(target_size=(config["img_size"], config["img_size"]))

    # --- Load Weights ---
    print(f"Loading model weights from: {weights_path}")
    if not weights_path.exists():
        print(f"Error: Weights file not found at {weights_path}"); return
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        try:
             print("Attempting to load as full checkpoint...")
             model = KelpSegmentationModel.load_from_checkpoint(weights_path, map_location=device)
             print("Loaded as full checkpoint instead.")
        except Exception as e2:
             print(f"Failed to load as full checkpoint: {e2}"); return

    # --- Prepare for Evaluation ---
    model.to(device)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving predictions to: {output_dir}")

    # Initialize metrics
    iou_metric = JaccardIndex(task="binary").to(device)
    precision_metric = Precision(task="binary").to(device)
    recall_metric = Recall(task="binary").to(device)
    f1_metric = F1Score(task="binary").to(device)

    # --- Inference Loop ---
    print("Running inference on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images, masks, filename_stems = batch
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            preds_prob = torch.sigmoid(logits)
            preds_binary = (preds_prob > config["threshold"]).byte()

            # Update metrics
            iou_metric.update(preds_binary, masks.int())
            precision_metric.update(preds_binary, masks.int())
            recall_metric.update(preds_binary, masks.int())
            f1_metric.update(preds_binary, masks.int())

            # Save Predictions
            preds_to_save = preds_binary.cpu().numpy().astype(np.uint8)
            for i in range(preds_to_save.shape[0]):
                pred_array = preds_to_save[i].squeeze()
                stem = filename_stems[i]
                save_path = output_dir / f"{stem}_pred.tif"
                try:
                    if pred_array.ndim != 2: continue # Skip if shape is wrong
                    tifffile.imwrite(save_path, pred_array)
                except Exception as e:
                    print(f"Error saving prediction {save_path.name}: {e}")

    # --- Compute and Print Final Metrics ---
    print("\n--- Evaluation Metrics ---")
    final_iou = iou_metric.compute()
    final_precision = precision_metric.compute()
    final_recall = recall_metric.compute()
    final_f1 = f1_metric.compute()
    print(f"  Intersection over Union (IoU): {final_iou:.4f}")
    print(f"  Precision:                     {final_precision:.4f}")
    print(f"  Recall:                        {final_recall:.4f}")
    print(f"  F1-Score:                      {final_f1:.4f}")
    print("--------------------------")

    # Reset metrics
    iou_metric.reset(); precision_metric.reset(); recall_metric.reset(); f1_metric.reset()
    print("Testing script finished.")


if __name__ == "__main__":
    # Create a dictionary from the constants to pass to the function
    config = {
        "weights_path_str": WEIGHTS_PATH_STR,
        "data_dir_str": DATA_DIR_STR,
        "output_dir_str": OUTPUT_DIR_STR,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "threshold": THRESHOLD,
        "seed": SEED,
    }
    # Run the main testing function
    run_testing(config)