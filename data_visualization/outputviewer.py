import matplotlib.pyplot as plt
import numpy as np
import random
import tifffile as tiff
from PIL import Image
import os
# from pyprojroot import here # Using Path().resolve() instead
from pathlib import Path
import warnings

'''
Used to view a random output of the test script. 
compares RGB representation of a sattelite image, the model's prediction, and its ground truth Kelp Mask
'''

OUTPUT_RUN = "34_clean_aug"

# --- Helper Functions ---

def get_overlay(image_GT):
    rgba_overlay = np.zeros((image_GT.shape[0], image_GT.shape[1], 4), dtype=np.float32)

    cmap = plt.get_cmap('Wistia')
    kelp_color = cmap(1.0)  # Get the color for the 'highest' value in Wistia (fully opaque kelp)

    # Set the RGBA values where kelp is present
    rgba_overlay[image_GT == 1, :3] = kelp_color[:3]  # Set RGB from colormap
    rgba_overlay[image_GT == 1, 3] = 1.0  # Set alpha to 1 (fully opaque) for kelp
    rgba_overlay[image_GT == 0, 3] = 0.0 #Set alpha to 0 where there is no kelp.

    return rgba_overlay

def normalize_band(band):
    """Normalizes a band to the range [0, 1], handling potential division by zero."""
    min_val = band.min()
    max_val = band.max()
    if max_val == min_val:
        return np.zeros_like(band)  # Return zeros if the band is constant
    return (band - min_val) / (max_val - min_val)

# --- Main Script ---

def main():
    # Define project root relative to the script location
    # Assumes script is in 'scripts' or 'models' folder, adjust if needed
    root_dir = Path().resolve().parent
    # --- Define Directories ---
    # Use the output directory specified in test.py constants
    pred_mask_dir = root_dir / "output" / OUTPUT_RUN
    satellite_dir = root_dir / "data" / "cleaned" / "test_satellite"
    gt_mask_dir = root_dir / "data" / "cleaned" / "test_kelp"

    # --- Validate Directories ---
    if not pred_mask_dir.exists():
        print(f"Error: Prediction directory not found: {pred_mask_dir}")
        return
    if not satellite_dir.exists():
        print(f"Error: Satellite directory not found: {satellite_dir}")
        return
    if not gt_mask_dir.exists():
        print(f"Error: Ground truth directory not found: {gt_mask_dir}")
        return

    # --- Select Random Prediction and Find Corresponding Files ---
    max_attempts = 20 # Increased attempts slightly
    attempt = 0
    found_files = False
    while attempt < max_attempts:
        attempt += 1
        try:
            # ============================================================
            # CHANGE 1: Update glob pattern for prediction files
            # ============================================================
            prediction_files = list(pred_mask_dir.glob('*_pred.tif'))
            if not prediction_files:
                print(f"Error: No prediction files matching '*_pred.tif' found in {pred_mask_dir}")
                return

            random_pred_path = random.choice(prediction_files)
            pred_filename = random_pred_path.name

            # ============================================================
            # CHANGE 2: Update base filename extraction
            # ============================================================
            # Remove the '_pred.tif' suffix to get the Tile ID
            base_filename = pred_filename.replace("_pred.tif", "")

            # Construct corresponding satellite and GT filenames
            satellite_filename = f"{base_filename}_satellite.tif"
            gt_mask_filename = f"{base_filename}_kelp.tif" # Assuming GT filenames are TILEID_kelp.tif
            satellite_path = satellite_dir / satellite_filename
            gt_mask_path = gt_mask_dir / gt_mask_filename

            # Check if corresponding files exist
            if not satellite_path.exists():
                warnings.warn(f"Satellite file not found ({satellite_filename}) for prediction {pred_filename}. Trying another...")
                continue
            if not gt_mask_path.exists():
                warnings.warn(f"Ground truth mask not found ({gt_mask_filename}) for prediction {pred_filename}. Trying another...")
                continue

            found_files = True # Found a complete set
            break # Exit the loop

        except Exception as e:
            print(f"An error occurred during file selection (Attempt {attempt}/{max_attempts}): {e}")
            # Optionally continue or return based on error severity
            continue # Try again

    if not found_files:
        print(f"Error: Could not find a complete set of corresponding files after {max_attempts} attempts.")
        return

    print(f"Selected Prediction: {pred_filename}")
    print(f"Corresponding Satellite: {satellite_filename}")
    print(f"Corresponding Ground Truth: {gt_mask_filename}")

    # --- Load Images ---
    try:
        # Use tifffile for consistency, as PIL might handle multi-channel differently
        pred_mask_array = tiff.imread(random_pred_path)
        # If mask has channel dim, remove it
        if pred_mask_array.ndim == 3:
             pred_mask_array = pred_mask_array.squeeze()
        if pred_mask_array.ndim != 2:
             raise ValueError(f"Prediction mask {pred_filename} has unexpected shape after squeeze: {pred_mask_array.shape}")


        # Load satellite image (assuming it was saved as H, W, C)
        image_ST = tiff.imread(satellite_path)
        if image_ST.ndim != 3 or image_ST.shape[-1] != 7:
             raise ValueError(f"Satellite image {satellite_filename} has unexpected shape: {image_ST.shape}. Expected (H, W, 7)")

        # Load ground truth mask
        gt_mask_array = tiff.imread(gt_mask_path)
        # If mask has channel dim, remove it
        if gt_mask_array.ndim == 3:
             gt_mask_array = gt_mask_array.squeeze()
        if gt_mask_array.ndim != 2:
            raise ValueError(f"Ground truth mask {gt_mask_filename} has unexpected shape after squeeze: {gt_mask_array.shape}")


    except Exception as e:
        print(f"Error loading image files: {e}")
        return

    # --- Process Satellite Image for RGB Display ---
    # Indices 2, 3, 4 correspond to Red, Green, Blue in a (H, W, 7) array
    # where channels are SWIR, NIR, Red, Green, Blue, Cloud, DEM
    try:
        if image_ST.shape[-1] < 5: # Need at least up to Blue band
             raise ValueError(f"Satellite image {satellite_filename} has fewer than 5 bands.")

        # Check if data looks normalized (e.g., float32 and potentially negative)
        # or scaled (e.g., uint8/16 or float with max > 1)
        # Apply normalization only if needed
        r_band = image_ST[:, :, 2].astype(float)
        g_band = image_ST[:, :, 3].astype(float)
        b_band = image_ST[:, :, 4].astype(float)

        # Simple contrast stretch / scaling if data is not already [0,1]
        # Adjust this logic based on how your satellite data is stored/normalized
        if r_band.max() > 1.0 or r_band.min() < 0.0: r_band = normalize_band(r_band)
        if g_band.max() > 1.0 or g_band.min() < 0.0: g_band = normalize_band(g_band)
        if b_band.max() > 1.0 or b_band.min() < 0.0: b_band = normalize_band(b_band)

        rgb_image = np.dstack((r_band, g_band, b_band))
        # Clip just in case normalization produced slightly out-of-range values
        rgb_image = np.clip(rgb_image, 0, 1)

    except Exception as e:
        print(f"Error processing satellite image for RGB: {e}")
        return

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    # 1) Original Satellite Image
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Satellite (RGB)")
    axes[0].axis('off')

    # 2) Predicted Mask (Directly)
    # Use vmin/vmax for binary images to ensure correct color mapping
    axes[1].imshow(rgb_image)
    predicted_overlay = get_overlay(pred_mask_array)
    axes[1].imshow(predicted_overlay)
    axes[1].set_title("Predicted Kelp Mask")
    axes[1].axis('off')

    # 3) Ground Truth Mask (Directly)
    axes[2].imshow(rgb_image)
    gt_overlay = get_overlay(gt_mask_array)
    axes[2].imshow(gt_overlay)
    axes[2].set_title("Ground Truth Kelp Mask")
    axes[2].axis('off')

    plt.suptitle(f"Comparison for: {base_filename}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

if __name__ == "__main__":
    main()