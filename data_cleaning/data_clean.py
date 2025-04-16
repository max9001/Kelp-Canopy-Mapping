import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
import warnings
import gc # Garbage collector

# --- Configuration ---
BASE_DIR = Path().resolve().parent
# *** VERY IMPORTANT: This script will modify files in this directory! ***
# *** Make sure this points to the COPY of your data.             ***
CLEANED_DATA_DIR = BASE_DIR / "data" / "cleaned"
SAT_DIR = CLEANED_DATA_DIR / "train_satellite"

# Check if the target directory exists
if not SAT_DIR.is_dir():
    raise FileNotFoundError(
        f"Specified satellite directory does not exist: {SAT_DIR}\n"
        "Make sure you have copied your original data to this 'cleaned' location."
    )

# Band definitions (0-based index)
BANDS_TO_NORMALIZE = [0, 1, 2, 3, 4, 6] # SWIR, NIR, Red, Green, Blue, DEM
CLOUD_MASK_BAND = 5
DEM_BAND = 6

# Data markers and masking thresholds
BAD_DATA_MARKER = -32768
# Pixels < 0 in spectral bands (0-4) are also often bad/missing data
SPECTRAL_BANDS = [0, 1, 2, 3, 4]
NEGATIVE_THRESHOLD = 0 # Treat values < 0 in spectral bands as bad

CLOUD_MASK_THRESHOLD = 1 # Pixels in band 5 equal to this are clouds
WATER_MASK_THRESHOLD = 0 # Pixels in band 6 less than or equal to this are water

# Clipping configuration (Optional)
APPLY_CLIPPING = True # Set to False to disable clipping before stat calculation
CLIP_LOW_PERCENTILE = 1.0 # e.g., 1st percentile
CLIP_HIGH_PERCENTILE = 99.0 # e.g., 99th percentile

# Epsilon to prevent division by zero during standardization
EPSILON = 1e-8

# --- Helper Functions ---

def create_masks(img_data):
    """Creates masks for bad data, clouds, and water."""
    # 1. Bad data mask (specific marker and negatives in spectral)
    bad_data_mask = (img_data == BAD_DATA_MARKER)
    for band_idx in SPECTRAL_BANDS:
         if band_idx < img_data.shape[0]: # Check if band exists
             bad_data_mask[band_idx, :, :] |= (img_data[band_idx, :, :] < NEGATIVE_THRESHOLD)

    # 2. Cloud mask
    cloud_mask = np.zeros_like(img_data[0, :, :], dtype=bool)
    if CLOUD_MASK_BAND < img_data.shape[0]:
        cloud_mask = (img_data[CLOUD_MASK_BAND, :, :] == CLOUD_MASK_THRESHOLD)

    # 3. Water mask (based on DEM)
    water_mask = np.zeros_like(img_data[0, :, :], dtype=bool)
    if DEM_BAND < img_data.shape[0] and np.issubdtype(img_data[DEM_BAND, :, :].dtype, np.number):
        # Ensure DEM band isn't itself entirely bad data before masking
        dem_valid = ~bad_data_mask[DEM_BAND, :, :]
        water_mask[dem_valid] = (img_data[DEM_BAND, :, :][dem_valid] <= WATER_MASK_THRESHOLD)
    
    # Combine masks: Mask pixels that are bad OR cloud OR water
    # Apply cloud/water mask only to bands we intend to normalize stats for
    combined_mask_for_stats = np.zeros_like(img_data, dtype=bool)
    for band_idx in BANDS_TO_NORMALIZE:
         if band_idx < img_data.shape[0]:
             # Mask if bad data OR (cloud AND it's not the cloud mask band itself) OR (water AND it's not the DEM band itself - though DEM is usually normalized)
             # Simplified: mask if bad OR cloud OR water for the purpose of stat calculation on normalized bands
              combined_mask_for_stats[band_idx, :, :] = (
                  bad_data_mask[band_idx, :, :] | cloud_mask | water_mask
              )
              
    # Return the bad data mask separately for initial NaN replacement
    # and the combined mask for statistical calculations
    return bad_data_mask, combined_mask_for_stats, cloud_mask, water_mask


# --- Main Script ---

print(f"Starting standardization process for data in: {SAT_DIR}")
print("*** This script will modify files in place! ***")
tif_files = sorted(list(SAT_DIR.glob("*_satellite.tif")))

if not tif_files:
    print(f"Error: No satellite TIFF files found in {SAT_DIR}")
    exit()

print(f"Found {len(tif_files)} files.")

# --- Phase 1: Calculate Statistics (Mean, Std Dev, and Clipping Thresholds) ---
print("\n--- Phase 1: Calculating statistics across the dataset ---")

# Initialize lists to store valid, masked pixel values for each band
# Warning: This can be memory intensive for large datasets!
# Consider iterative calculation (sum, sum_sq, count) for very large datasets.
all_band_data = {band_idx: [] for band_idx in BANDS_TO_NORMALIZE}
total_pixels_processed = 0
pixels_per_image = None

# First pass to gather data for stats
for file_path in tqdm(tif_files, desc="Pass 1/2: Reading data for stats"):
    try:
        img = tifffile.imread(file_path)
        # Ensure data is float for NaN compatibility
        img_float = img.astype(np.float32)
        if pixels_per_image is None and len(img_float.shape) == 3:
             pixels_per_image = img_float.shape[1] * img_float.shape[2]


        # Handle potential 2D images or unexpected shapes
        if len(img_float.shape) != 3 or img_float.shape[0] < max(BANDS_TO_NORMALIZE):
             warnings.warn(f"Skipping {file_path.name}: Unexpected shape {img_float.shape}")
             continue
             
        # Create masks
        bad_data_mask, combined_mask_for_stats, _, _ = create_masks(img_float)

        # Temporarily replace *all* bad data markers with NaN for calculations
        img_float[bad_data_mask] = np.nan

        for band_idx in BANDS_TO_NORMALIZE:
            if band_idx < img_float.shape[0]:
                band_pixels = img_float[band_idx, :, :]
                # Apply the combined mask (bad data, clouds, water)
                mask = combined_mask_for_stats[band_idx, :, :]
                valid_pixels = band_pixels[~mask] # Select only pixels NOT masked
                all_band_data[band_idx].extend(valid_pixels.tolist()) # Append valid pixels

        total_pixels_processed += pixels_per_image if pixels_per_image else 0
        
        # Clear memory
        del img, img_float, band_pixels, valid_pixels, bad_data_mask, combined_mask_for_stats
        gc.collect()


    except Exception as e:
        warnings.warn(f"Skipping {file_path.name} due to error during stat calculation pass: {e}")

# Calculate clipping thresholds (if enabled) and stats
print("\nCalculating final statistics...")
band_stats = {}
clip_thresholds = {}

for band_idx in BANDS_TO_NORMALIZE:
    data_array = np.array(all_band_data[band_idx], dtype=np.float32)
    
    if data_array.size == 0:
         warnings.warn(f"No valid data found for band {band_idx} after masking. Setting stats to mean=0, std=1.")
         mean = 0.0
         std = 1.0
         low_clip, high_clip = -np.inf, np.inf # No effective clipping
    else:
        # Calculate clipping thresholds on the gathered data *before* calculating mean/std
        if APPLY_CLIPPING:
            low_clip = np.nanpercentile(data_array, CLIP_LOW_PERCENTILE)
            high_clip = np.nanpercentile(data_array, CLIP_HIGH_PERCENTILE)
            clip_thresholds[band_idx] = (low_clip, high_clip)
            # Apply clipping to the data before calculating mean/std
            data_array = np.clip(data_array, low_clip, high_clip)
            print(f"Band {band_idx}: Applied clipping [{low_clip:.2f}, {high_clip:.2f}]")
        else:
            low_clip, high_clip = -np.inf, np.inf # Store ineffective thresholds if clipping off
            clip_thresholds[band_idx] = (low_clip, high_clip)


        # Calculate mean and std dev on the (potentially clipped) valid data
        mean = np.nanmean(data_array)
        std = np.nanstd(data_array)

    # Store stats
    band_stats[band_idx] = {'mean': mean, 'std': std if std > EPSILON else 1.0} # Avoid std=0

    print(f"Band {band_idx}: Mean={band_stats[band_idx]['mean']:.4f}, StdDev={band_stats[band_idx]['std']:.4f}")
    
    # Clear memory for the band
    all_band_data[band_idx] = [] # Free list memory
    del data_array
    gc.collect()
    
# Clear the large data structure
del all_band_data
gc.collect()

# --- Phase 2: Apply Standardization and Save ---
print("\n--- Phase 2: Applying standardization and saving modified files ---")

for file_path in tqdm(tif_files, desc="Pass 2/2: Applying normalization"):
    try:
        # Read the image again
        img = tifffile.imread(file_path)
        original_dtype = img.dtype
        img_standardized = img.astype(np.float32) # Work with float32

        # Handle potential 2D images or unexpected shapes encountered earlier
        if len(img_standardized.shape) != 3 or img_standardized.shape[0] < max(BANDS_TO_NORMALIZE):
             # This file should have been skipped in phase 1, but double-check
             warnings.warn(f"Skipping {file_path.name} in phase 2: Unexpected shape {img_standardized.shape}")
             continue
             
        # Re-create bad data mask for NaN replacement
        bad_data_mask, _, _, _ = create_masks(img_standardized)

        # Keep the original cloud mask band untouched if it exists
        original_cloud_mask_data = None
        if CLOUD_MASK_BAND < img_standardized.shape[0]:
            original_cloud_mask_data = img_standardized[CLOUD_MASK_BAND, :, :].copy()

        # Apply standardization band by band
        for band_idx in BANDS_TO_NORMALIZE:
             if band_idx < img_standardized.shape[0]:
                 band_data = img_standardized[band_idx, :, :]
                 band_bad_mask = bad_data_mask[band_idx, :, :]

                 # Replace bad data with NaN for clipping/standardization
                 band_data[band_bad_mask] = np.nan

                 # Apply clipping (using pre-calculated thresholds)
                 if APPLY_CLIPPING:
                      low_clip, high_clip = clip_thresholds[band_idx]
                      band_data = np.clip(band_data, low_clip, high_clip) # np.clip handles NaNs correctly (keeps them NaN)

                 # Apply standardization: (value - mean) / std
                 mean = band_stats[band_idx]['mean']
                 std = band_stats[band_idx]['std']
                 standardized_band = (band_data - mean) / (std + EPSILON) # Add epsilon for safety

                 # Replace NaNs (originally bad data) with 0 after standardization
                 standardized_band[np.isnan(standardized_band)] = 0.0

                 img_standardized[band_idx, :, :] = standardized_band

        # Restore the original cloud mask data
        if original_cloud_mask_data is not None:
            img_standardized[CLOUD_MASK_BAND, :, :] = original_cloud_mask_data

        # Save the modified image back to the *same file path*
        # Ensure output dtype is float32 for standardized data
        tifffile.imwrite(file_path, img_standardized.astype(np.float32), imagej=True) # imagej=True often helps compatibility

    except Exception as e:
        warnings.warn(f"Failed to process and save {file_path.name}: {e}")
        # Optionally, decide if you want to stop or continue on error

print("\nStandardization process complete.")
print(f"Files in {SAT_DIR} have been modified.")


'''

Calculating final statistics...
Band 0: Applied clipping [6.00, 19689.00]
Band 0: Mean=8800.5283, StdDev=3631.0242
Band 1: Applied clipping [6.00, 19673.00]
Band 1: Mean=8798.9248, StdDev=3628.6870
Band 2: Applied clipping [6.00, 19661.00]
Band 2: Mean=8797.4678, StdDev=3627.4983
Band 3: Applied clipping [6.00, 19647.00]
Band 3: Mean=8795.8926, StdDev=3626.0796
Band 4: Applied clipping [6.00, 19647.00]
Band 4: Mean=8794.6309, StdDev=3625.4009
Band 6: Applied clipping [7.00, 19603.00]
Band 6: Mean=8791.6172, StdDev=3620.2842

'''