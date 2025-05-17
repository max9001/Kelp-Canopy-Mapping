import matplotlib.pyplot as plt
import numpy as np
import random
import tifffile as tiff
from pathlib import Path
import warnings

'''
Used to view a random output of a mask generation script.
Compares an RGB representation of a satellite image with the model's predicted mask.
'''

# --- Configuration ---
# Name of the inference run whose predictions you want to view
# This should match the RUN_NAME used in the inference script,
# or at least the subfolder name under "output/generated_masks/"
INFERENCE_RUN_NAME = "34_clean_aug_inference" 
SATELLITE_INPUT_DIR_NAME = "train101" 
SATELLITE_DATA_SUBFOLDER = "cleaned" # Example: "cleaned", "original"

# Suffix used for predicted mask files by the inference script
PREDICTION_SUFFIX = "_predmask.tif"
SATELLITE_SUFFIX = "_satellite.tif"
# --- End Configuration ---


# --- Helper Functions ---

def get_overlay(image_mask_binary, cmap_name='Wistia', alpha_kelp=0.7, alpha_no_kelp=0.0):
    """
    Creates an RGBA overlay for binary masks.
    Pixels with value 1 are colored and made partially transparent.
    Pixels with value 0 are made fully transparent (or as specified).
    """
    if image_mask_binary.ndim != 2:
        raise ValueError(f"Input mask must be 2D, got shape {image_mask_binary.shape}")

    rgba_overlay = np.zeros((image_mask_binary.shape[0], image_mask_binary.shape[1], 4), dtype=np.float32)
    cmap = plt.get_cmap(cmap_name)
    kelp_color = cmap(1.0)  # Get the color for the 'highest' value (e.g., bright yellow for Wistia)

    # Apply color and alpha where mask is 1 (kelp predicted)
    kelp_pixels = (image_mask_binary == 1)
    rgba_overlay[kelp_pixels, :3] = kelp_color[:3]  # Set RGB
    rgba_overlay[kelp_pixels, 3] = alpha_kelp      # Set alpha for kelp

    # Set alpha for non-kelp pixels (mask is 0)
    no_kelp_pixels = (image_mask_binary == 0)
    rgba_overlay[no_kelp_pixels, 3] = alpha_no_kelp

    return rgba_overlay

def normalize_band(band):
    """Normalizes a band to the range [0, 1], handling potential division by zero."""
    min_val = band.min()
    max_val = band.max()
    if max_val == min_val:
        # If band is constant, return zeros or a constant value (e.g., 0.5 if it's non-zero)
        return np.zeros_like(band, dtype=np.float32) if min_val == 0 else np.full_like(band, 0.5, dtype=np.float32)
    return (band.astype(np.float32) - min_val) / (max_val - min_val)

# --- Main Script ---

def main():
    # --- Define Project Root and Directories ---
    # Assumes script is in project_root/scripts/ or similar,
    # and 'data' & 'output' are children of project_root.
    root_dir = Path().resolve()
    if not (root_dir / "data").exists() and not (root_dir / "output").exists():
        root_dir = root_dir.parent # Try one level up if not found

    pred_mask_dir = root_dir / "output" / "generated_masks" / INFERENCE_RUN_NAME
    satellite_dir = root_dir / "data" / SATELLITE_DATA_SUBFOLDER / SATELLITE_INPUT_DIR_NAME

    # --- Validate Directories ---
    if not pred_mask_dir.exists():
        print(f"Error: Prediction directory not found: {pred_mask_dir}")
        print(f"Please check INFERENCE_RUN_NAME ('{INFERENCE_RUN_NAME}') and script's relative location.")
        return
    if not satellite_dir.exists():
        print(f"Error: Satellite directory not found: {satellite_dir}")
        print(f"Please check SATELLITE_INPUT_DIR_NAME ('{SATELLITE_INPUT_DIR_NAME}') and SATELLITE_DATA_SUBFOLDER ('{SATELLITE_DATA_SUBFOLDER}').")
        return

    # --- Select Random Prediction and Find Corresponding Satellite File ---
    max_attempts = 30
    attempt = 0
    found_pair = False
    pred_path = None
    satellite_path = None
    base_filename = None

    prediction_files = list(pred_mask_dir.glob(f"*{PREDICTION_SUFFIX}"))
    if not prediction_files:
        print(f"Error: No prediction files matching '*{PREDICTION_SUFFIX}' found in {pred_mask_dir}")
        return

    print(f"Found {len(prediction_files)} prediction files in {pred_mask_dir}.")

    while attempt < max_attempts and not found_pair:
        attempt += 1
        try:
            random_pred_path = random.choice(prediction_files)
            pred_filename_full = random_pred_path.name

            # Extract base filename (e.g., TILEID from TILEID_predmask.tif)
            current_base_filename = pred_filename_full.replace(PREDICTION_SUFFIX, "")

            # Construct corresponding satellite filename
            satellite_filename_full = f"{current_base_filename}{SATELLITE_SUFFIX}"
            current_satellite_path = satellite_dir / satellite_filename_full

            if current_satellite_path.exists():
                pred_path = random_pred_path
                satellite_path = current_satellite_path
                base_filename = current_base_filename
                found_pair = True
                break
            else:
                if attempt % 5 == 0: # Print warning less frequently
                    warnings.warn(f"Attempt {attempt}: Satellite file '{satellite_filename_full}' not found in {satellite_dir} "
                                  f"for prediction '{pred_filename_full}'. Trying another...")

        except Exception as e:
            print(f"An error occurred during file selection (Attempt {attempt}/{max_attempts}): {e}")
            continue

    if not found_pair:
        print(f"Error: Could not find a corresponding satellite image for any of the randomly selected "
              f"prediction files after {max_attempts} attempts.")
        print(f"Checked for satellite files like 'BASE_FILENAME{SATELLITE_SUFFIX}' in {satellite_dir}")
        print(f"And prediction files like 'BASE_FILENAME{PREDICTION_SUFFIX}' in {pred_mask_dir}")
        return

    print(f"\nSelected Prediction: {pred_path.name}")
    print(f"Corresponding Satellite: {satellite_path.name}")

    # --- Load Images ---
    try:
        pred_mask_array = tiff.imread(pred_path)
        if pred_mask_array.ndim == 3 and pred_mask_array.shape[0] == 1: # Handle (1, H, W)
            pred_mask_array = pred_mask_array.squeeze(0)
        elif pred_mask_array.ndim == 3 and pred_mask_array.shape[-1] == 1: # Handle (H, W, 1)
             pred_mask_array = pred_mask_array.squeeze(-1)
        if pred_mask_array.ndim != 2:
             raise ValueError(f"Prediction mask {pred_path.name} has unexpected shape after squeeze: {pred_mask_array.shape}. Expected 2D.")

        image_ST = tiff.imread(satellite_path)
        if image_ST.ndim != 3 or image_ST.shape[-1] != 7: # Assuming H, W, C with 7 channels
             raise ValueError(f"Satellite image {satellite_path.name} has unexpected shape: {image_ST.shape}. Expected (H, W, 7)")

    except Exception as e:
        print(f"Error loading image files: {e}")
        return

    # --- Process Satellite Image for RGB Display ---
    # Indices 2, 3, 4 correspond to Red, Green, Blue if channels are [SWIR, NIR, Red, Green, Blue, Cloud, DEM]
    try:
        if image_ST.shape[-1] < 5: # Need at least Red, Green, Blue bands
             raise ValueError(f"Satellite image {satellite_path.name} has fewer than 5 bands (found {image_ST.shape[-1]}). Cannot extract RGB.")

        r_band = image_ST[:, :, 2]
        g_band = image_ST[:, :, 3]
        b_band = image_ST[:, :, 4]

        # Normalize each band independently for visualization
        rgb_image = np.dstack((
            normalize_band(r_band),
            normalize_band(g_band),
            normalize_band(b_band)
        ))
        # Clip to ensure values are strictly within [0, 1] after normalization
        rgb_image = np.clip(rgb_image, 0, 1)

    except Exception as e:
        print(f"Error processing satellite image for RGB display: {e}")
        # Fallback: create a black image of the same spatial dimensions if RGB fails
        if pred_mask_array is not None:
            rgb_image = np.zeros((pred_mask_array.shape[0], pred_mask_array.shape[1], 3), dtype=np.float32)
        else:
            rgb_image = np.zeros((256, 256, 3), dtype=np.float32) # Absolute fallback

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

    # 1) Original Satellite Image (RGB)
    axes[0].imshow(rgb_image)
    axes[0].set_title("Satellite Image (RGB)")
    axes[0].axis('off')

    # 2) Predicted Mask Overlay on Satellite Image
    axes[1].imshow(rgb_image) # Show satellite as background
    predicted_overlay = get_overlay(pred_mask_array, alpha_kelp=0.6) # Make overlay semi-transparent
    axes[1].imshow(predicted_overlay)
    axes[1].set_title("Predicted Kelp Mask (Overlay)")
    axes[1].axis('off')

    plt.suptitle(f"Prediction Visualization for: {base_filename}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

if __name__ == "__main__":
    main()