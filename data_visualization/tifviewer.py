import matplotlib.pyplot as plt
import numpy as np
import random
import tifffile as tiff
from PIL import Image
# import os # os is not directly used
# from pyprojroot import here # here is not used
from pathlib import Path
# import sys # sys.argv is commented out, can be removed if not intended for use

'''
Used to visualize an RGB representation of a sattelite image, and its ground truth Kelp Mask
'''


# --- Configuration Constants ---
ROOT_DIR = Path().resolve().parent
BASE_DATA_DIR = ROOT_DIR / "data" / "cleaned"
# FILENAME_BASE_TO_VISUALIZE = "AA498489" # Example: "AA498489"
FILENAME_BASE_TO_VISUALIZE = "" # If empty, a random file will be chosen




KELP_DATA_DIR = BASE_DATA_DIR / "train_kelp"
SATELLITE_DATA_DIR = BASE_DATA_DIR / "train_satellite"
def normalize_band(band):
    """Normalizes a single band of an image to the range [0, 1]."""
    min_val = band.min()
    max_val = band.max()
    if max_val == min_val: # Avoid division by zero if the band is flat
        return np.zeros_like(band, dtype=np.float32)
    return (band - min_val) / (max_val - min_val)

def gamma_correction(image, gamma=1.0):
    """Applies gamma correction to an image."""
    return np.power(image, gamma)

def get_overlay(image_GT):
    """
    Creates an RGBA overlay for kelp pixels.
    Kelp pixels (value 1) are colored using the 'Wistia' colormap and made opaque.
    Non-kelp pixels are made transparent.
    """
    rgba_overlay = np.zeros((image_GT.shape[0], image_GT.shape[1], 4), dtype=np.float32)

    cmap = plt.get_cmap('Wistia')
    # Get the color for the 'highest' value in Wistia (bright yellow, fully opaque for kelp)
    kelp_color = cmap(1.0)

    # Set the RGBA values where kelp is present
    kelp_mask = (image_GT == 1)
    rgba_overlay[kelp_mask, :3] = kelp_color[:3]  # Set RGB from colormap
    rgba_overlay[kelp_mask, 3] = 1.0             # Set alpha to 1 (fully opaque) for kelp

    # Non-kelp pixels (image_GT == 0) will have alpha = 0.0 by default from np.zeros initialization.
    # Explicitly: rgba_overlay[image_GT == 0, 3] = 0.0

    return rgba_overlay

# --- Main script execution ---

# Get list of available kelp mask filenames
try:
    filenames_in_kelp_dir = np.array([f.name for f in KELP_DATA_DIR.iterdir() if f.is_file() and f.name.endswith('_kelp.tif')])
    if filenames_in_kelp_dir.size == 0:
        print(f"Error: No '_kelp.tif' files found in {KELP_DATA_DIR}")
        exit()
except FileNotFoundError:
    print(f"Error: Kelp data directory not found: {KELP_DATA_DIR}")
    exit()


# Select a filename
if FILENAME_BASE_TO_VISUALIZE:
    selected_filename_base = FILENAME_BASE_TO_VISUALIZE
    # Verify if the chosen filename exists
    if f"{selected_filename_base}_kelp.tif" not in filenames_in_kelp_dir:
        print(f"Warning: Specified filename '{selected_filename_base}_kelp.tif' not found in {KELP_DATA_DIR}.")
        print(f"Choosing a random file instead.")
        selected_filename_full = random.choice(filenames_in_kelp_dir)
        selected_filename_base = selected_filename_full[:-9] # Remove '_kelp.tif'
else:
    selected_filename_full = random.choice(filenames_in_kelp_dir)
    selected_filename_base = selected_filename_full[:-9] # Remove '_kelp.tif'

print(f"Visualizing data for: {selected_filename_base}")

# Construct full paths to the image files
gt_img_path = KELP_DATA_DIR / f"{selected_filename_base}_kelp.tif"
st_img_path = SATELLITE_DATA_DIR / f"{selected_filename_base}_satellite.tif"

# Load Ground Truth (Kelp Mask)
try:
    image_GT = Image.open(gt_img_path)
    image_GT = np.array(image_GT)
except FileNotFoundError:
    print(f"Error: Ground truth image not found: {gt_img_path}")
    exit()

# Load Satellite Image
try:
    image_ST = tiff.imread(st_img_path)
    image_ST = np.array(image_ST) # Ensure it's a NumPy array, tiff.imread usually returns one
except FileNotFoundError:
    print(f"Error: Satellite image not found: {st_img_path}")
    exit()

# Band description:
# 	0: Short-wave infrared (SWIR)
# 	1: Near infrared (NIR)
# 	2: Red
# 	3: Green
# 	4: Blue
# 	5: Cloud Mask (binary - is there cloud or not)
# 	6: Digital Elevation Model (meters above sea-level)

# Create RGB image from satellite bands (Red, Green, Blue)
# Assuming bands 2, 3, 4 are Red, Green, Blue respectively.
if image_ST.ndim == 3 and image_ST.shape[-1] >= 5: # Check if image has enough bands
    rgb_image = np.dstack((
        normalize_band(image_ST[:, :, 2]),  # Red
        normalize_band(image_ST[:, :, 3]),  # Green
        normalize_band(image_ST[:, :, 4])   # Blue
    ))
else:
    print(f"Error: Satellite image {st_img_path.name} does not have expected dimensions or enough bands. Shape: {image_ST.shape}")
    # Create a dummy black image to prevent further errors if plotting is still attempted
    if image_GT is not None:
        rgb_image = np.zeros((image_GT.shape[0], image_GT.shape[1], 3), dtype=np.float32)
    else: # Fallback if GT also failed to load
        rgb_image = np.zeros((256, 256, 3), dtype=np.float32)


# --- Plotting ---
plt.figure(figsize=(14, 7)) # Increased figure size for better layout

plt.subplot(1, 2, 1)
plt.title(f"Satellite Image\n{selected_filename_base}", fontsize=16)
plt.imshow(rgb_image)
plt.axis('off') # Hide axes for cleaner image display

plt.subplot(1, 2, 2)
plt.title(f"Labeled Kelp Overlay\n{selected_filename_base}", fontsize=16)
plt.imshow(rgb_image) # Show satellite image as background
rgba_overlay = get_overlay(image_GT)
plt.imshow(rgba_overlay)  # Overlay kelp mask
plt.axis('off') # Hide axes

plt.tight_layout(pad=2.0) # Add some padding
plt.show()