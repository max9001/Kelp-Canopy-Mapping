import torch
import torch.nn as nn
import pytorch_lightning as pl
import tifffile
import numpy as np
from pathlib import Path
import sys
# import argparse # No command-line args needed
from tqdm import tqdm
import torchvision
# Import specific backbones and weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from typing import List, Tuple, Dict, Type # Added Type

# --- Metrics ---
# Import base Metric class and specific metrics to optimize
from torchmetrics import Metric, JaccardIndex, F1Score

# --- Configuration Constants ---
# *** SET THESE VALUES BEFORE RUNNING ***

# --- Identify the Run ---
# Define the name of the training run whose results you want to evaluate
RUN_NAME = "18_og_noaug" # <<< CHANGE THIS TO MATCH THE DESIRED RUN

# --- Determine Backbone Used for the Run ---
# !! IMPORTANT !! Set this to match the backbone used for the RUN_NAME
# Options: "resnet18", "resnet34", "resnet50"
BACKBONE_NAME = "resnet18" # <<< SET THIS TO MATCH THE RUN
DATA_DIR_STR = str(Path().resolve().parent / "data" / "original")

# --- Construct Paths Based on Run Name ---
RUN_DIR = Path().resolve().parent / "runs" / RUN_NAME
WEIGHTS_PATH_STR = str(RUN_DIR / "best_weights.pth") # Assumes this naming convention

# Path to the base directory containing the pre-split data folders
# (val_satellite, val_kelp, etc.)


# --- Threshold Finding Parameters ---
# Choose Metric: JaccardIndex (IoU) or F1Score
OPTIMIZE_THRESHOLD_METRIC: Type[Metric] = JaccardIndex
# Define the range and granularity of thresholds to search
THRESHOLD_SEARCH_RANGE = np.linspace(0.1, 0.9, 81) # e.g., 0.1 to 0.9 in steps of 0.01
# Output file name for the threshold
THRESHOLD_FILENAME = "optimal_threshold.txt"

# --- Other Parameters (Should match model training/testing) ---
IMG_SIZE = 350
BATCH_SIZE = 16 # Adjust based on GPU memory for validation inference
NUM_WORKERS = 4 # Adjust based on your system
SEED = 42
# --- End Configuration ---


# --- Import necessary components (Ensure these paths are correct) ---
try:
    UTILS_DIR = Path().resolve().parent.parent / "utils"
    if not UTILS_DIR.exists(): UTILS_DIR = Path().resolve().parent / "utils"
    if not UTILS_DIR.exists(): raise FileNotFoundError("Could not find utils directory")
    sys.path.append(str(UTILS_DIR.parent)); from utils.get_data import prepare_filenames
except ImportError: print("Error: Could not import prepare_filenames."); sys.exit(1)
except FileNotFoundError as e: print(f"Error finding utils directory: {e}"); sys.exit(1)


# --- Model and Dataset Class Definitions ---
# !! IMPORTANT !! These MUST match the definitions used during training
class KelpSegmentationModel(pl.LightningModule):
    def __init__(self, target_size=(350, 350), learning_rate=1e-4, backbone_name="resnet18"):
        super().__init__(); self.target_size = target_size; self.backbone_name = backbone_name
        if self.backbone_name == "resnet18": base_encoder = torchvision.models.resnet18(weights=None); encoder_out_channels = 512
        elif self.backbone_name == "resnet34": base_encoder = torchvision.models.resnet34(weights=None); encoder_out_channels = 512
        elif self.backbone_name == "resnet50": base_encoder = torchvision.models.resnet50(weights=None); encoder_out_channels = 2048
        else: raise ValueError(f"Unsupported backbone: {self.backbone_name}.")
        base_encoder.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(base_encoder.conv1, base_encoder.bn1, base_encoder.relu, base_encoder.maxpool, base_encoder.layer1, base_encoder.layer2, base_encoder.layer3, base_encoder.layer4)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(encoder_out_channels, 256, 3, 2, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 1))
    def forward(self, x):
        if x.ndim == 4 and x.shape[1] != 7: x = x.permute(0, 3, 1, 2)
        elif x.ndim != 4 or x.shape[1] != 7: raise ValueError(f"Invalid input shape: {x.shape}")
        features = self.encoder(x); logits = self.decoder(features)
        output = F.interpolate(logits, size=self.target_size, mode='bilinear', align_corners=False); return output

class KelpDataset(torch.utils.data.Dataset):
    def __init__(self, satellite_paths: List[str], mask_paths: List[str], return_filename=False): # return_filename not needed here
        self.satellite_paths=satellite_paths; self.mask_paths=mask_paths; self.return_filename=return_filename # Keep arg for compatibility if needed elsewhere
        if len(satellite_paths) != len(mask_paths): raise ValueError("Path lists length mismatch.")
        if not satellite_paths: raise ValueError("Satellite path list empty.")
    def __len__(self): return len(self.satellite_paths)
    def load_image(self, filename: str, is_mask: bool = False):
        try:
            img = tifffile.imread(filename); assert img is not None
            if not is_mask: assert img.ndim == 3 and img.shape[-1] == 7; img = img.astype(np.float32) if img.dtype != np.float32 else img
            else: img = img.squeeze() if img.ndim == 3 else img; img = (img > 0).astype(np.uint8) if img.dtype != np.uint8 or img.max() > 1 else img
            return img.astype('float32') # Return float mask for potential direct use
        except Exception as e: print(f"ERROR loading {filename}: {str(e)}"); raise
    def __getitem__(self, idx):
        sat_path = self.satellite_paths[idx]; mask_path = self.mask_paths[idx]
        try:
            sat_img = self.load_image(sat_path, is_mask=False)
            mask = self.load_image(mask_path, is_mask=True)
            if sat_img.shape[0] != mask.shape[0] or sat_img.shape[1] != mask.shape[1]: raise ValueError(f"Dims mismatch Sat ({sat_img.shape[:2]}) vs Mask ({mask.shape})")
            sat_tensor = torch.from_numpy(sat_img).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0) # Keep as (1, H, W) float
            # No augmentations applied here
            # No filename needed for this script's purpose
            return sat_tensor, mask_tensor
        except Exception as e: print(f"Error processing idx {idx}: {e}"); raise
# --- End Class Definitions ---


def find_and_save_threshold(config: Dict):
    """Loads model, finds optimal threshold on validation set, saves it."""
    pl.seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert string paths from config to Path objects
    weights_path = Path(config["weights_path_str"]).resolve()
    data_dir = Path(config["data_dir_str"]).resolve()
    run_dir = Path(config["run_dir_str"]).resolve() # Get run dir for saving threshold
    metric_class = config["metric_class"]
    threshold_range = config["threshold_range"]

    # --- Load Validation Data ---
    try:
        print(f"Loading validation filenames from base directory: {data_dir}")
        filenames = prepare_filenames(base_dir=data_dir)
        # Indices: 0=train_sat, 1=train_gt, 2=val_sat, 3=val_gt, 4=test_sat, 5=test_gt
        val_sat_paths = filenames[2]
        val_gt_paths = filenames[3]
        if not val_sat_paths or not val_gt_paths:
             raise ValueError("Validation set paths are empty. Check data split and prepare_filenames.")
        print(f"Found {len(val_sat_paths)} validation samples.")
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR during data preparation: {e}"); return

    # --- Create Validation Dataset and DataLoader ---
    try:
        # No transforms/noise for validation threshold finding
        val_dataset = KelpDataset(val_sat_paths, val_gt_paths, return_filename=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True
        )
    except ValueError as e:
        print(f"ERROR creating dataset/loader: {e}"); return

    # --- Initialize Model ---
    print(f"Initializing model structure with backbone: {config['backbone_name']}...")
    model = KelpSegmentationModel(
        target_size=(config["img_size"], config["img_size"]),
        backbone_name=config["backbone_name"]
    )

    # --- Load Weights ---
    print(f"Loading model weights from: {weights_path}")
    if not weights_path.exists():
        print(f"Error: Weights file not found at {weights_path}"); return
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    except RuntimeError as e:
         print(f"ERROR loading state_dict: {e}"); print(f"Ensure BACKBONE_NAME ('{config['backbone_name']}') matches training."); return
    except Exception as e: print(f"Generic error loading weights: {e}"); return

    # --- Find Optimal Threshold ---
    model.to(device)
    model.eval()
    all_preds_probs = []
    all_gt_masks = []

    print("\nCollecting validation set predictions...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation Inference"):
            images, masks = batch
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_preds_probs.append(probs.cpu())
            all_gt_masks.append(masks.cpu().int()) # Ensure GT is int for metrics

    try:
        all_preds_probs_tensor = torch.cat(all_preds_probs, dim=0)
        all_gt_masks_tensor = torch.cat(all_gt_masks, dim=0)
    except RuntimeError as e: print(f"Error concatenating tensors: {e}"); return
    if all_preds_probs_tensor.numel() == 0: print("Error: No predictions collected."); return

    print(f"\nSearching for optimal threshold using {metric_class.__name__}...")
    best_threshold = 0.5
    best_score = -1.0
    metric = metric_class(task="binary").to(device)

    # Move tensors to device for calculations
    all_preds_probs_tensor = all_preds_probs_tensor.to(device)
    all_gt_masks_tensor = all_gt_masks_tensor.to(device)

    for threshold in tqdm(threshold_range, desc="Testing thresholds"):
        preds_binary = (all_preds_probs_tensor > threshold).byte()
        metric.update(preds_binary, all_gt_masks_tensor)
        current_score = metric.compute().item()
        metric.reset()
        if current_score > best_score:
            best_score = current_score
            best_threshold = threshold

    print("\n" + "="*60)
    print(f"Optimal Threshold Calculation Complete for Run: {config['run_name']}")
    print(f"  Optimized Metric: {metric_class.__name__}")
    print(f"  Best Threshold:   {best_threshold:.4f}")
    print(f"  Best Val Score:   {best_score:.4f}")
    print("="*60)

    # --- Save Threshold ---
    threshold_save_path = run_dir / config["threshold_filename"]
    try:
        run_dir.mkdir(parents=True, exist_ok=True) # Ensure run dir exists
        with open(threshold_save_path, 'w') as f:
            f.write(f"{best_threshold:.6f}\n") # Save with high precision
        print(f"Optimal threshold saved to: {threshold_save_path}")
    except Exception as e:
        print(f"Warning: Could not save optimal threshold to file: {e}")


if __name__ == "__main__":
    # Create config dictionary from constants
    config = {
        "run_name": RUN_NAME,
        "backbone_name": BACKBONE_NAME,
        "weights_path_str": WEIGHTS_PATH_STR,
        "data_dir_str": DATA_DIR_STR,
        "run_dir_str": str(RUN_DIR), # Pass run dir for saving threshold file
        "metric_class": OPTIMIZE_THRESHOLD_METRIC,
        "threshold_range": THRESHOLD_SEARCH_RANGE,
        "threshold_filename": THRESHOLD_FILENAME,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "seed": SEED,
    }
    # Run the threshold finding function
    find_and_save_threshold(config)