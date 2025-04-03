# import matplotlib.pyplot as plt
# import numpy as np
# import random
# import tifffile as tiff
# from pathlib import Path

# # Get the output directory
# directory = Path().resolve().parent / "output" / "predictions_resnet_test"

# # Get a list of all image filenames
# filenames = np.array([f.name for f in directory.iterdir() if f.is_file()])

# # Keep searching for an image with at least one nonzero pixel
# valid_image = None
# max_attempts = len(filenames)  # Prevent infinite loops

# for _ in range(max_attempts):
#     filename = random.choice(filenames)
#     output_mask = tiff.imread(directory / filename)

#     if np.any(output_mask):  # Check if there's at least one nonzero pixel
#         valid_image = output_mask
#         break

# output_mask = tiff.imread(directory / filename)
# valid_image = output_mask


# # print(valid_image)
# if valid_image is not None:
#     plt.figure(figsize=(12, 6))
#     plt.title(f"Predicted Image: {filename}", fontsize=25)
#     plt.imshow(valid_image)  # Ensure grayscale visualization
#     plt.show()
# else:
#     print("No nonzero masks found in the dataset.")




import matplotlib.pyplot as plt
import numpy as np
import random
import tifffile as tiff
from PIL import Image
import os
from pyprojroot import here
from pathlib import Path

# --- Helper Functions ---

def normalize_band(band):
    """Normalizes a band to the range [0, 1], handling potential division by zero."""
    min_val = band.min()
    max_val = band.max()
    if max_val == min_val:
        return np.zeros_like(band)  # Return zeros if the band is constant
    return (band - min_val) / (max_val - min_val)

# get_overlay function is no longer needed for the main plot,
# but might be useful elsewhere, so keeping it is fine.
def get_overlay(mask_array, cmap_name='Wistia', alpha=0.6):
    """Creates an RGBA overlay for a binary mask."""
    if mask_array.dtype != bool:
        mask_array = mask_array.astype(bool) # Ensure boolean for indexing

    rgba_overlay = np.zeros((mask_array.shape[0], mask_array.shape[1], 4), dtype=np.float32)
    cmap = plt.get_cmap(cmap_name)
    kelp_color = cmap(0.8)

    rgba_overlay[mask_array, :3] = kelp_color[:3]
    rgba_overlay[mask_array, 3] = alpha
    return rgba_overlay

# --- Main Script ---

def main():
    root_dir = here()

    # --- Define Directories ---
    pred_mask_dir = root_dir / "output" / "predictions_resnet_test"
    satellite_dir = root_dir / "data" / "balanced_tiled_40_60" / "train_satellite"
    gt_mask_dir = root_dir / "data" / "balanced_tiled_40_60" / "train_kelp"

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
    max_attempts = 10
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            prediction_files = list(pred_mask_dir.glob('prediction_*.tif'))
            if not prediction_files:
                print(f"Error: No prediction files found in {pred_mask_dir}")
                return

            random_pred_path = random.choice(prediction_files)
            pred_filename = random_pred_path.name
            base_filename = pred_filename.replace("prediction_", "").replace(".tif", "")
            satellite_filename = f"{base_filename}_satellite.tif"
            gt_mask_filename = f"{base_filename}_kelp.tif"
            satellite_path = satellite_dir / satellite_filename
            gt_mask_path = gt_mask_dir / gt_mask_filename

            if not satellite_path.exists():
                print(f"Warning: Satellite file not found for {pred_filename}. Trying another...")
                continue
            if not gt_mask_path.exists():
                print(f"Warning: Ground truth mask not found for {pred_filename}. Trying another...")
                continue
            break
        except Exception as e:
            print(f"An error occurred during file selection: {e}")
            return

    if attempt >= max_attempts:
        print(f"Error: Could not find a complete set of corresponding files after {max_attempts} attempts.")
        return

    print(f"Selected Prediction: {pred_filename}")
    print(f"Corresponding Satellite: {satellite_filename}")
    print(f"Corresponding Ground Truth: {gt_mask_filename}")

    # --- Load Images ---
    try:
        pred_mask_img = Image.open(random_pred_path)
        pred_mask_array = np.array(pred_mask_img)

        image_ST = tiff.imread(satellite_path)

        gt_mask_img = Image.open(gt_mask_path)
        gt_mask_array = np.array(gt_mask_img)

    except Exception as e:
        print(f"Error loading image files: {e}")
        return

    # --- Process Satellite Image for RGB Display ---
    try:
        if image_ST.shape[-1] < 5:
             raise ValueError(f"Satellite image {satellite_filename} has fewer than 5 bands.")

        rgb_image = np.dstack((
            normalize_band(image_ST[:, :, 2]),
            normalize_band(image_ST[:, :, 3]),
            normalize_band(image_ST[:, :, 4])
        ))
    except Exception as e:
        print(f"Error processing satellite image for RGB: {e}")
        return

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1) Original Satellite Image
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Satellite (RGB)")
    axes[0].axis('off')

    # 2) Predicted Mask (Directly)
    axes[1].imshow(pred_mask_array, cmap='gray') # Display mask directly with grayscale colormap
    axes[1].set_title("Predicted Kelp Mask")
    axes[1].axis('off')

    # 3) Ground Truth Mask (Directly)
    axes[2].imshow(gt_mask_array, cmap='gray') # Display mask directly with grayscale colormap
    axes[2].set_title("Ground Truth Kelp Mask")
    axes[2].axis('off')

    plt.suptitle(f"Comparison for: {base_filename}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()