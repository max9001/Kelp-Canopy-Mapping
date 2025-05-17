import torch
import torch.nn as nn
import pytorch_lightning as pl
import tifffile
import numpy as np
from pathlib import Path
# import sys # sys is no longer used for command-line arguments
from tqdm import tqdm
import torchvision
# Import specific backbones
from torchvision.models import resnet18, resnet34, resnet50
# Weights are not explicitly used from torchvision.models for pretrained, model uses its own trained weights.
import torch.nn.functional as F
from typing import List, Tuple, Dict, Type
import warnings # For warnings.warn

ROOT_DIR = Path().resolve().parent
if not (ROOT_DIR / "data").exists() and (Path().resolve() / "data").exists(): # If script is in root
    ROOT_DIR = Path().resolve()


# --- Configuration Constants ---
# *** SET THESE VALUES BEFORE RUNNING ***
SATELLITE_INPUT_DIR_STR = str(ROOT_DIR / "data" / "cleaned" / "train101") # Directory with satellite TIFs to process
ORIGINAL_RUN_DIR_FOR_WEIGHTS = ROOT_DIR / "runs" / "34_clean_aug" # Example: parent of your 'best_weights.pth'
RUN_NAME = "34_clean_aug_inference" # Name for this inference run (affects output dir)
BACKBONE_NAME = "resnet34"     # Must match the backbone used for training the weights
APPLY_LAND_MASK = True         # If True, applies DEM-based land masking post-prediction



# Path to the directory of the *original training run* to fetch weights and potentially optimal_threshold.txt
# This is NOT where this inference script saves its outputs, but where it loads from.
OUTPUT_DIR_STR = str(ROOT_DIR / "output" / "generated_masks" / RUN_NAME) # Where generated masks will be saved
WEIGHTS_FILENAME = "best_weights.pth" # Name of the weights file within ORIGINAL_RUN_DIR_FOR_WEIGHTS
WEIGHTS_PATH_STR = str(ORIGINAL_RUN_DIR_FOR_WEIGHTS / WEIGHTS_FILENAME)
OPTIMAL_THRESHOLD_FILE_PATH = ORIGINAL_RUN_DIR_FOR_WEIGHTS / "optimal_threshold.txt" # Optional

# --- Inference Parameters ---
IMG_SIZE = 350
BATCH_SIZE = 16
NUM_WORKERS = 4
SEED = 42
DEFAULT_THRESHOLD = 0.5 # Fallback if optimal_threshold.txt is not found or unreadable
# --- End Configuration ---


# --- Model and Dataset Class Definitions ---
# !! IMPORTANT !! These MUST match the definitions used during training the loaded weights
class KelpSegmentationModel(pl.LightningModule):
    def __init__(self, target_size=(350, 350), learning_rate=1e-4, backbone_name="resnet18"):
        super().__init__()
        self.target_size = target_size
        self.backbone_name = backbone_name
        if self.backbone_name == "resnet18":
            base_encoder = torchvision.models.resnet18(weights=None) # Load structure only
            encoder_out_channels = 512
        elif self.backbone_name == "resnet34":
            base_encoder = torchvision.models.resnet34(weights=None)
            encoder_out_channels = 512
        elif self.backbone_name == "resnet50":
            base_encoder = torchvision.models.resnet50(weights=None)
            encoder_out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}.")
        # Modify input layer for 7 channels
        base_encoder.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(
            base_encoder.conv1, base_encoder.bn1, base_encoder.relu, base_encoder.maxpool,
            base_encoder.layer1, base_encoder.layer2, base_encoder.layer3, base_encoder.layer4
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_out_channels, 256, 3, 2, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )
    def forward(self, x):
        if x.ndim == 4 and x.shape[1] != 7: x = x.permute(0, 3, 1, 2) # Ensure CxHxW
        elif x.ndim != 4 or x.shape[1] != 7: raise ValueError(f"Invalid input shape: {x.shape}, expected Bx7xHxW")
        features = self.encoder(x)
        logits = self.decoder(features)
        output = F.interpolate(logits, size=self.target_size, mode='bilinear', align_corners=False)
        return output

class KelpInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, satellite_paths: List[str], return_filename=True, return_dem=False):
        self.satellite_paths = satellite_paths
        self.return_filename = return_filename
        self.return_dem = return_dem
        self.dem_band_index = 6 # 0-indexed
        if not satellite_paths:
            raise ValueError("Satellite path list cannot be empty.")

    def __len__(self):
        return len(self.satellite_paths)

    def load_image(self, filename: str): # Simplified for satellite images only
        try:
            img = tifffile.imread(filename)
            assert img is not None, f"Image loaded as None: {filename}"
            assert img.ndim == 3 and img.shape[-1] == 7, f"Unexpected image shape {img.shape} for {filename}"
            img = img.astype(np.float32) if img.dtype != np.float32 else img
            return img
        except Exception as e:
            print(f"ERROR loading satellite image {filename}: {str(e)}")
            raise

    def __getitem__(self, idx):
        sat_path = self.satellite_paths[idx]
        filename_stem = Path(sat_path).stem.replace("_satellite", "")

        try:
            sat_img_np = self.load_image(sat_path)
            sat_tensor = torch.from_numpy(sat_img_np).permute(2, 0, 1) # To CxHxW

            items_to_return = [sat_tensor]
            dem_tensor_to_return = None

            if self.return_dem:
                if sat_img_np.shape[2] > self.dem_band_index:
                    dem_band_np = sat_img_np[:, :, self.dem_band_index].copy()
                    dem_tensor_to_return = torch.from_numpy(dem_band_np).unsqueeze(0) # Add channel dim
                else:
                    warnings.warn(f"DEM band index {self.dem_band_index} out of bounds for {Path(sat_path).name} (shape {sat_img_np.shape}). Returning a dummy DEM tensor of -1s.")
                    dummy_dem_np = np.full(sat_img_np.shape[:2], -1, dtype=np.float32)
                    dem_tensor_to_return = torch.from_numpy(dummy_dem_np).unsqueeze(0)
                items_to_return.append(dem_tensor_to_return)

            if self.return_filename:
                items_to_return.append(filename_stem)

            return tuple(items_to_return)
        except Exception as e:
            print(f"Error processing file at index {idx} ({filename_stem}): {e}")
            # To allow dataloader to skip, would need a custom collate_fn. For now, raising.
            raise
# --- End Class Definitions ---


def run_inference(config: Dict):
    """Runs inference to generate masks using parameters from the config dict."""
    pl.seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    apply_land_mask = config["apply_land_mask"]
    if apply_land_mask:
        print("Land mask post-processing: ENABLED (DEM band will be used)")
    else:
        print("Land mask post-processing: DISABLED")

    # --- Determine Threshold ---
    threshold = config["default_threshold"]
    optimal_threshold_file = Path(config["optimal_threshold_file_path"])
    if optimal_threshold_file.exists():
        try:
            with open(optimal_threshold_file, 'r') as f:
                threshold = float(f.readline().strip())
            print(f"Using optimal threshold read from {optimal_threshold_file}: {threshold:.4f}")
        except Exception as e:
            print(f"Warning: Could not read {optimal_threshold_file}. Using default {threshold:.4f}. Error: {e}")
    else:
        print(f"Optimal threshold file {optimal_threshold_file} not found. Using default threshold: {threshold:.4f}")
    config["threshold"] = threshold # Store actual used threshold in config for reference

    # --- Convert Paths ---
    weights_path = Path(config["weights_path_str"]).resolve()
    satellite_input_dir = Path(config["satellite_input_dir_str"]).resolve()
    output_dir = Path(config["output_dir_str"]).resolve()

    # --- Load Satellite Image Paths ---
    print(f"Scanning for satellite images in: {satellite_input_dir}")
    # Assuming satellite images end with "_satellite.tif"
    # Modify glob pattern if your naming is different.
    sat_image_paths = sorted(list(satellite_input_dir.glob("*_satellite.tif")))
    if not sat_image_paths:
        print(f"ERROR: No satellite TIFF files (ending with '_satellite.tif') found in {satellite_input_dir}")
        return
    print(f"Found {len(sat_image_paths)} satellite images to process.")

    # --- Create Dataset and DataLoader ---
    try:
        # Pass apply_land_mask to return_dem, as DEM is needed for land masking
        inference_dataset = KelpInferenceDataset(
            [str(p) for p in sat_image_paths], # Convert Path objects to strings
            return_filename=True,
            return_dem=apply_land_mask
        )
        inference_loader = torch.utils.data.DataLoader(
            inference_dataset,
            batch_size=config["batch_size"],
            shuffle=False, # Important for consistent output if needed, not critical for generation
            num_workers=config["num_workers"],
            pin_memory=True
        )
    except ValueError as e:
        print(f"ERROR creating dataset/dataloader: {e}")
        return

    # --- Initialize Model ---
    print(f"Initializing model structure with backbone: {config['backbone_name']}...")
    model = KelpSegmentationModel(
        target_size=(config["img_size"], config["img_size"]),
        backbone_name=config["backbone_name"]
    )

    # --- Load Weights ---
    print(f"Loading model weights from: {weights_path}")
    if not weights_path.exists():
        print(f"Error: Weights file not found: {weights_path}")
        return
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    except RuntimeError as e:
        print(f"ERROR loading state_dict: {e}\nEnsure BACKBONE_NAME in config matches the one used for training.")
        return
    except Exception as e:
        print(f"Generic error loading weights: {e}")
        return

    # --- Prepare for Inference ---
    model.to(device)
    model.eval() # Set model to evaluation mode

    output_dir.mkdir(parents=True, exist_ok=True) # Create output directory if it doesn't exist
    print(f"Saving predicted masks to: {output_dir}")

    # --- Inference Loop ---
    print("Running inference to generate masks...")
    with torch.no_grad(): # Disable gradient calculations for inference
        for batch in tqdm(inference_loader, desc="Generating Masks"):
            # Unpack batch based on what KelpInferenceDataset returns
            if apply_land_mask:
                images, dems, filename_stems = batch
                dems = dems.to(device)
            else:
                images, filename_stems = batch
                # dems is not available or needed

            images = images.to(device)

            # Forward pass
            logits = model(images)
            preds_prob = torch.sigmoid(logits) # Apply sigmoid to get probabilities

            # Apply threshold to get binary predictions
            preds_binary = (preds_prob > config["threshold"]).byte() # .byte() or .bool().int()

            # Apply land mask if enabled
            if apply_land_mask:
                # DEM > 0 is considered land. Invert to get water mask.
                # Predictions should only be where water is.
                water_mask_gpu = (dems <= 0) # Assuming DEM <=0 is water
                preds_binary_final = preds_binary * water_mask_gpu
            else:
                preds_binary_final = preds_binary

            # Save predictions
            preds_to_save = preds_binary_final.cpu().numpy().astype(np.uint8)
            for i in range(preds_to_save.shape[0]): # Iterate over batch items
                pred_array = preds_to_save[i].squeeze() # Remove channel dim (1, H, W) -> (H, W)
                stem = filename_stems[i]
                # Save with a descriptive name, e.g., originalstem_predmask.tif
                save_path = output_dir / f"{stem}_predmask.tif"
                try:
                    if pred_array.ndim != 2:
                        warnings.warn(f"Prediction for {stem} has unexpected ndim {pred_array.ndim} after squeeze. Skipping save.")
                        continue
                    tifffile.imwrite(save_path, pred_array)
                except Exception as e:
                    print(f"Error saving prediction for {save_path.name}: {e}")

    print("\nMask generation complete.")
    print(f"All generated masks have been saved to: {output_dir}")


if __name__ == "__main__":
    # Create config dictionary from constants
    config = {
        "run_name": RUN_NAME,
        "backbone_name": BACKBONE_NAME,
        "weights_path_str": WEIGHTS_PATH_STR,
        "satellite_input_dir_str": SATELLITE_INPUT_DIR_STR, # New input dir
        "output_dir_str": OUTPUT_DIR_STR,
        "optimal_threshold_file_path": str(OPTIMAL_THRESHOLD_FILE_PATH), # Path to optional threshold file
        "default_threshold": DEFAULT_THRESHOLD,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "seed": SEED,
        "apply_land_mask": APPLY_LAND_MASK
    }
    # Run the main inference function
    run_inference(config)