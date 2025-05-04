import torch
import torch.nn as nn
import pytorch_lightning as pl
import tifffile
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import torchvision
# Import specific backbones and weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from typing import List, Tuple, Dict, Type

# --- Metrics ---
from torchmetrics import Metric, JaccardIndex, Precision, Recall, F1Score

# --- Configuration Constants ---
# *** SET THESE VALUES BEFORE RUNNING ***

RUN_NAME = "18_og_noaug"
BACKBONE_NAME = "resnet18"
APPLY_LAND_MASK = True

# --- Construct Paths Based on Run Name ---
RUN_DIR = Path().resolve().parent / "runs" / RUN_NAME
WEIGHTS_PATH_STR = str(RUN_DIR / "best_weights.pth")
DATA_DIR_STR = str(Path().resolve().parent / "data" / "original")
OUTPUT_DIR_STR = str(Path().resolve().parent / "output" / RUN_NAME)
OPTIMAL_THRESHOLD_FILE = RUN_DIR / "optimal_threshold.txt"

# --- Other Testing Parameters ---
IMG_SIZE = 350
BATCH_SIZE = 16
NUM_WORKERS = 4
# THRESHOLD = 0.5 # Default, will be overwritten if optimal file exists
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

# === Use MODIFIED KelpDataset Definition from Step 1 ===
class KelpDataset(torch.utils.data.Dataset):
    def __init__(self, satellite_paths: List[str], mask_paths: List[str],
                 return_filename=False, return_dem=False): # Added return_dem flag
        self.satellite_paths = satellite_paths
        self.mask_paths = mask_paths
        self.return_filename = return_filename
        self.return_dem = return_dem # Store the flag
        self.dem_band_index = 6 # Assuming DEM is the 7th band (index 6)

        if len(satellite_paths) != len(mask_paths): raise ValueError("Path lists length mismatch.")
        if not satellite_paths: raise ValueError("Satellite path list empty.")

    def __len__(self): return len(self.satellite_paths)

    def load_image(self, filename: str, is_mask: bool = False):
        try:
            img = tifffile.imread(filename); assert img is not None
            if not is_mask: # Satellite Image (H, W, C)
                if img.dtype != np.float32: img = img.astype(np.float32)
                if img.ndim != 3 or img.shape[-1] != 7: raise ValueError(f"Sat shape: {img.shape}")
            else: # Mask (H, W)
                 img = img.squeeze() if img.ndim == 3 else img
                 if img.dtype != np.uint8 or img.max() > 1: img = (img > 0).astype(np.uint8)
                 img = img.astype('float32') # Return float mask
            return img
        except Exception as e: print(f"ERROR loading {filename}: {str(e)}"); raise

    def __getitem__(self, idx):
        sat_path = self.satellite_paths[idx]
        mask_path = self.mask_paths[idx]
        filename_stem = Path(sat_path).stem.replace("_satellite", "")
        try:
            sat_img_np = self.load_image(sat_path, is_mask=False) # Loads H, W, C
            mask_np = self.load_image(mask_path, is_mask=True)   # Loads H, W float

            if sat_img_np.shape[0]!= mask_np.shape[0] or sat_img_np.shape[1]!= mask_np.shape[1]:
                raise ValueError(f"Dims mismatch Sat ({sat_img_np.shape[:2]}) vs Mask ({mask_np.shape})")

            dem_band_np = None
            if self.return_dem:
                if sat_img_np.shape[2] > self.dem_band_index:
                    dem_band_np = sat_img_np[:, :, self.dem_band_index].copy() # H, W
                else:
                    warnings.warn(f"DEM band index {self.dem_band_index} out of bounds for {Path(sat_path).name}.")
                    dem_band_np = np.full(sat_img_np.shape[:2], -1, dtype=np.float32) # Dummy

            # Convert main data to tensors
            sat_tensor = torch.from_numpy(sat_img_np).permute(2, 0, 1) # C, H, W
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0) # 1, H, W

            # --- Prepare return tuple ---
            return_items = [sat_tensor, mask_tensor]
            if self.return_dem:
                 # Return DEM as a tensor (1, H, W) for easier batching
                 dem_tensor = torch.from_numpy(dem_band_np).unsqueeze(0)
                 return_items.append(dem_tensor)
            if self.return_filename:
                return_items.append(filename_stem)

            return tuple(return_items)

        except Exception as e: print(f"Error processing idx {idx}: {e}"); raise
# --- End Class Definitions ---


def run_testing(config: Dict):
    """Runs inference and evaluation using parameters from the config dict."""
    pl.seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    apply_land_mask = config["apply_land_mask"] # Get boolean flag from config
    if apply_land_mask:
        print("Land mask post-processing: ENABLED")
    else:
        print("Land mask post-processing: DISABLED")


    # --- Read Optimal Threshold ---
    threshold = 0.5 # Default
    optimal_threshold_file = Path(config["run_dir_str"]) / "optimal_threshold.txt"
    if optimal_threshold_file.exists():
        try:
            with open(optimal_threshold_file, 'r') as f:
                threshold = float(f.readline().strip())
            print(f"Using optimal threshold read from file: {threshold:.4f}")
        except Exception as e:
            print(f"Warning: Could not read threshold file {optimal_threshold_file}. Using default 0.5. Error: {e}")
            threshold = 0.5
    else:
         print(f"Warning: Optimal threshold file not found at {optimal_threshold_file}. Using default 0.5.")
         threshold = 0.5
    config["threshold"] = threshold # Update config with the threshold actually used

    # --- Convert Paths ---
    weights_path = Path(config["weights_path_str"]).resolve()
    data_dir = Path(config["data_dir_str"]).resolve()
    output_dir = Path(config["output_dir_str"]).resolve()

    # --- Load Data ---
    try:
        print(f"Loading test filenames from base directory: {data_dir}")
        filenames = prepare_filenames(base_dir=data_dir)
        test_sat_paths = filenames[4]; test_gt_paths = filenames[5]
        if not test_sat_paths or not test_gt_paths: raise ValueError("Test set paths are empty.")
        print(f"Found {len(test_sat_paths)} test samples.")
    except (ValueError, FileNotFoundError) as e: print(f"ERROR: {e}"); return

    # --- Create Dataset and DataLoader ---
    try:
        # ============================================================================
        # MODIFIED: Request DEM and filename from dataset
        # ============================================================================
        test_dataset = KelpDataset(test_sat_paths, test_gt_paths,
                                   return_filename=True,
                                   return_dem=apply_land_mask) # Only load DEM if needed
        # ============================================================================
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config["batch_size"], shuffle=False,
            num_workers=config["num_workers"], pin_memory=True
        )
    except ValueError as e: print(f"ERROR: {e}"); return

    # --- Initialize Model ---
    print(f"Initializing model structure with backbone: {config['backbone_name']}...")
    model = KelpSegmentationModel(target_size=(config["img_size"], config["img_size"]), backbone_name=config["backbone_name"])

    # --- Load Weights ---
    print(f"Loading model weights from: {weights_path}")
    if not weights_path.exists(): print(f"Error: Weights file not found: {weights_path}"); return
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict); print("Weights loaded successfully.")
    except RuntimeError as e: print(f"ERROR loading state_dict: {e}\nEnsure BACKBONE_NAME ('{config['backbone_name']}') matches training."); return
    except Exception as e: print(f"Generic error loading weights: {e}"); return

    # --- Prepare for Evaluation ---
    model.to(device); model.eval()
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
            # --- Unpack batch (might include DEM now) ---
            if apply_land_mask:
                images, masks, dems, filename_stems = batch # Unpack DEM
                dems = dems.to(device) # Move DEM to device for masking
            else:
                images, masks, filename_stems = batch # Original unpacking

            images = images.to(device)
            masks = masks.to(device) # Ground truth

            # --- Forward pass ---
            logits = model(images)
            preds_prob = torch.sigmoid(logits)
            preds_binary = (preds_prob > config["threshold"]).byte() # Initial binary preds

            # ============================================================================
            # ADDED: Apply Land Mask conditionally before metrics
            # ============================================================================
            if apply_land_mask:
                # Create land mask (DEM > 0 indicates land) on the same device
                # Ensure DEM has channel dim (1, H, W) like preds_binary
                land_mask_gpu = (dems > 0)
                # Zero out predictions where land_mask is True
                # Use bitwise NOT (~) on the boolean mask and multiply
                preds_binary_masked = preds_binary * (~land_mask_gpu)
            else:
                preds_binary_masked = preds_binary # Use original preds if not masking
            # ============================================================================

            # --- Update Metrics using potentially masked predictions ---
            iou_metric.update(preds_binary_masked, masks.int())
            precision_metric.update(preds_binary_masked, masks.int())
            recall_metric.update(preds_binary_masked, masks.int())
            f1_metric.update(preds_binary_masked, masks.int())

            # --- Save Predictions (Save the MASKED version) ---
            # Move the potentially masked predictions to CPU for saving
            preds_to_save = preds_binary_masked.cpu().numpy().astype(np.uint8)

            for i in range(preds_to_save.shape[0]): # Iterate through batch
                pred_array = preds_to_save[i].squeeze() # (H, W)
                stem = filename_stems[i]
                save_path = output_dir / f"{stem}_pred.tif"
                try:
                    if pred_array.ndim != 2: continue
                    tifffile.imwrite(save_path, pred_array)
                except Exception as e: print(f"Error saving prediction {save_path.name}: {e}")

    # --- Compute and Print Final Metrics ---
    print("\n--- Evaluation Metrics ---")
    final_iou = iou_metric.compute()
    final_precision = precision_metric.compute()
    final_recall = recall_metric.compute()
    final_f1 = f1_metric.compute()
    print(f"  Run:                           {config['run_name']}")
    print(f"  Backbone:                      {config['backbone_name']}")
    print(f"  Weights:                       {weights_path.name}")
    print(f"  Threshold Used:                {config['threshold']:.4f}")
    print(f"  Land Mask Applied:             {'Yes' if apply_land_mask else 'No'}") # Indicate if mask applied
    print(f"  Intersection over Union (IoU): {final_iou:.4f}")
    print(f"  Precision:                     {final_precision:.4f}")
    print(f"  Recall:                        {final_recall:.4f}")
    print(f"  F1-Score:                      {final_f1:.4f}")
    print("--------------------------")

    # Reset metrics
    iou_metric.reset(); precision_metric.reset(); recall_metric.reset(); f1_metric.reset()
    print("Testing script finished.")


if __name__ == "__main__":
    # Create config dictionary from constants
    config = {
        "run_name": RUN_NAME,
        "backbone_name": BACKBONE_NAME,
        "weights_path_str": WEIGHTS_PATH_STR,
        "data_dir_str": DATA_DIR_STR,
        "output_dir_str": OUTPUT_DIR_STR,
        "run_dir_str": str(RUN_DIR), # Pass run dir for reading threshold file
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        # "threshold": THRESHOLD, # Initial value, might be overwritten
        "seed": SEED,
        "apply_land_mask": APPLY_LAND_MASK # Pass the flag
    }
    # Run the main testing function
    run_testing(config)