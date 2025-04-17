import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
import warnings
import gc
from joblib import Parallel, delayed
import multiprocessing

# --- Configuration ---
BASE_DIR = Path().resolve().parent
if not (BASE_DIR / "data").exists():
     BASE_DIR = Path().resolve()
     if not (BASE_DIR / "data").exists():
         raise FileNotFoundError("Could not automatically find the 'data' directory relative to the script.")

CLEANED_DATA_DIR = BASE_DIR / "data" / "cleaned"
SAT_DIR = CLEANED_DATA_DIR / "train_satellite" # MODIFY THIS DIRECTORY

if not SAT_DIR.is_dir():
    raise FileNotFoundError(f"Directory not found: {SAT_DIR}")

# --- Constants ---
NUM_BANDS = 7
EXPECTED_DIM = 350
BANDS_TO_NORMALIZE = [0, 1, 2, 3, 4, 6] # SWIR, NIR, Red, Green, Blue, DEM (Indices for C,H,W)
CLOUD_MASK_BAND = 5
DEM_BAND = 6
BAD_DATA_MARKER = -32768
SPECTRAL_BANDS = [0, 1, 2, 3, 4]
NEGATIVE_THRESHOLD = 0
CLOUD_MASK_THRESHOLD = 1
WATER_MASK_THRESHOLD = 0
EPSILON = 1e-8
APPLY_CLIPPING = True # Control clipping in Phase 2

# *** IMPORTANT: Define Global Clipping Thresholds ***
# These need to be calculated beforehand (e.g., from your previous run or
# a separate script analyzing a sample/full dataset more slowly).
# Using the values you provided as an example:
GLOBAL_CLIP_THRESHOLDS = {
    # band_idx: (low_percentile_value, high_percentile_value)
    0: (6.00, 19689.00),
    1: (6.00, 19673.00),
    2: (6.00, 19661.00),
    3: (6.00, 19647.00),
    4: (6.00, 19647.00),
    6: (7.00, 19603.00),
}


# --- Helper Functions (fix_image_shape, create_masks - keep as before) ---
def fix_image_shape(img, filename):
    """Attempts to reshape/transpose image to (C, H, W)."""
    if img.ndim != 3:
        warnings.warn(f"Skipping {filename}: Unexpected dimensions {img.ndim}, expected 3.")
        return None
    shape = img.shape
    dims = list(shape)
    try:
        c_idx = dims.index(NUM_BANDS)
        # Check if other dimensions match EXPECTED_DIM
        other_dims = [d for i, d in enumerate(dims) if i != c_idx]
        if len(other_dims) == 2 and all(d == EXPECTED_DIM for d in other_dims):
             if shape == (NUM_BANDS, EXPECTED_DIM, EXPECTED_DIM): return img # C, H, W
             if shape == (EXPECTED_DIM, EXPECTED_DIM, NUM_BANDS): return np.transpose(img, (2, 0, 1)) # H, W, C
             if shape == (EXPECTED_DIM, NUM_BANDS, EXPECTED_DIM): return np.transpose(img, (1, 0, 2)) # H, C, W
             # Add other permutations if necessary, e.g., W, H, C -> transpose(img, (2, 1, 0)) etc.
             else:
                 warnings.warn(f"Skipping {filename}: Ambiguous shape {shape} - cannot reliably determine C, H, W order.")
                 return None
        else:
             warnings.warn(f"Skipping {filename}: Shape {shape} has {NUM_BANDS} bands but other dims are not ({EXPECTED_DIM}, {EXPECTED_DIM}).")
             return None
    except ValueError:
        warnings.warn(f"Skipping {filename}: Shape {shape} does not contain expected number of bands ({NUM_BANDS}).")
        return None

def create_masks(img_data):
    """Creates masks. Assumes input shape (C, H, W)."""
    bad_data_mask = (img_data == BAD_DATA_MARKER)
    for band_idx in SPECTRAL_BANDS:
         if band_idx < img_data.shape[0]:
             bad_data_mask[band_idx, :, :] |= (img_data[band_idx, :, :] < NEGATIVE_THRESHOLD)
    cloud_mask = np.zeros_like(img_data[0, :, :], dtype=bool)
    if CLOUD_MASK_BAND < img_data.shape[0]:
        cloud_mask = (img_data[CLOUD_MASK_BAND, :, :] == CLOUD_MASK_THRESHOLD)
    water_mask = np.zeros_like(img_data[0, :, :], dtype=bool)
    if DEM_BAND < img_data.shape[0] and np.issubdtype(img_data[DEM_BAND, :, :].dtype, np.number):
        # Ensure DEM isn't all bad data before masking
        dem_not_bad = ~bad_data_mask[DEM_BAND, :, :]
        # Apply water mask only where DEM is not bad data
        water_mask[dem_not_bad] = (img_data[DEM_BAND, :, :][dem_not_bad] <= WATER_MASK_THRESHOLD)

    # Combined mask for STATS calculation (ignore bad data, clouds, water)
    combined_mask_for_stats = np.zeros_like(img_data, dtype=bool)
    for band_idx in BANDS_TO_NORMALIZE:
         if band_idx < img_data.shape[0]:
              combined_mask_for_stats[band_idx, :, :] = (
                  bad_data_mask[band_idx, :, :] | cloud_mask | water_mask
              )
    return bad_data_mask, combined_mask_for_stats


# --- Phase 1: Calculate Mean/Std Iteratively ---
print("\n--- Phase 1: Calculating statistics iteratively ---")

accumulators = {
    band_idx: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0}
    for band_idx in BANDS_TO_NORMALIZE
}

tif_files = sorted(list(SAT_DIR.glob("*_satellite.tif")))
if not tif_files:
    print(f"Error: No satellite TIFF files found in {SAT_DIR}")
    exit()
print(f"Found {len(tif_files)} files.")

for file_path in tqdm(tif_files, desc="Pass 1/1: Calculating stats"):
    try:
        img_raw = tifffile.imread(file_path)
        img = fix_image_shape(img_raw, file_path.name)
        if img is None: continue

        img_float = img.astype(np.float64) # Use float64 for accumulators to avoid precision issues

        _, combined_mask_for_stats = create_masks(img_float)
        img_float[combined_mask_for_stats] = np.nan # Mask bad/cloud/water for stats

        for band_idx in BANDS_TO_NORMALIZE:
            if band_idx < img_float.shape[0]:
                band_data = img_float[band_idx, :, :]
                valid_pixels = band_data[~np.isnan(band_data)]

                accumulators[band_idx]['sum'] += np.sum(valid_pixels)
                accumulators[band_idx]['sum_sq'] += np.sum(valid_pixels**2)
                accumulators[band_idx]['count'] += valid_pixels.size

        del img_raw, img, img_float, band_data, valid_pixels, combined_mask_for_stats
        gc.collect()

    except Exception as e:
        warnings.warn(f"Skipping {file_path.name} during stats calculation: {e}")

# Calculate final mean and std
print("\nCalculating final statistics...")
band_stats = {}
for band_idx in BANDS_TO_NORMALIZE:
    total_sum = accumulators[band_idx]['sum']
    total_sum_sq = accumulators[band_idx]['sum_sq']
    total_count = accumulators[band_idx]['count']

    if total_count == 0:
        warnings.warn(f"No valid pixels found for band {band_idx}. Setting mean=0, std=1.")
        mean, std = 0.0, 1.0
    else:
        mean = total_sum / total_count
        # Variance = E[X^2] - (E[X])^2
        variance = (total_sum_sq / total_count) - (mean**2)
        if variance < 0: # Handle potential floating point inaccuracies near zero
             variance = 0
        std = np.sqrt(variance)
        if std < EPSILON:
            warnings.warn(f"Std dev for band {band_idx} is near zero ({std}). Setting to 1.0 to avoid division by zero.")
            std = 1.0

    band_stats[band_idx] = {'mean': mean, 'std': std}
    print(f"Band {band_idx}: Mean={mean:.4f}, StdDev={std:.4f} (Calculated BEFORE clipping)")

del accumulators # Free accumulator memory
gc.collect()


# --- Phase 2: Apply Clipping, Normalization, Save (Parallel) ---
print("\n--- Phase 2: Applying clipping, normalization, and saving (Parallel) ---")

def process_file(file_path, global_stats, global_clips, apply_clip):
    """Processes a single file: clips, normalizes, saves."""
    try:
        img_raw = tifffile.imread(file_path)
        img = fix_image_shape(img_raw, file_path.name)
        if img is None: return f"Skipped {file_path.name}: Invalid shape"

        img_normalized = img.astype(np.float32) # Use float32 for final output

        # Create bad data mask (only need this one now)
        bad_data_mask, _ = create_masks(img_normalized) # Don't need combined mask here

        # Store original cloud mask
        original_cloud_mask_data = None
        if CLOUD_MASK_BAND < img_normalized.shape[0]:
            original_cloud_mask_data = img_normalized[CLOUD_MASK_BAND, :, :].copy()

        for band_idx in BANDS_TO_NORMALIZE:
            if band_idx < img_normalized.shape[0]:
                band_data = img_normalized[band_idx, :, :]
                band_bad_mask = bad_data_mask[band_idx, :, :]

                # Replace bad data with NaN for clipping/math
                band_data[band_bad_mask] = np.nan

                # --- Apply Clipping (using GLOBAL thresholds) ---
                if apply_clip and band_idx in global_clips:
                    low_clip, high_clip = global_clips[band_idx]
                    # Use nan_to_num before clip might be safer if clip doesn't handle NaN well
                    # band_data = np.nan_to_num(band_data, nan=some_value_outside_clip_range)
                    band_data = np.clip(band_data, low_clip, high_clip) # np.clip handles NaNs correctly

                # --- Apply Standardization ---
                mean = global_stats[band_idx]['mean']
                std = global_stats[band_idx]['std']
                standardized_band = (band_data - mean) / std # std already checked for > EPSILON

                # Replace final NaNs (orig bad data) with 0
                standardized_band[np.isnan(standardized_band)] = 0.0
                img_normalized[band_idx, :, :] = standardized_band

        # Restore original cloud mask
        if original_cloud_mask_data is not None:
            img_normalized[CLOUD_MASK_BAND, :, :] = original_cloud_mask_data

        # Transpose back to H, W, C for saving
        img_to_save = np.transpose(img_normalized, (1, 2, 0))

        # Final shape check
        if img_to_save.shape != (EXPECTED_DIM, EXPECTED_DIM, NUM_BANDS):
             return f"Failed {file_path.name}: Final shape incorrect {img_to_save.shape}"

        # Overwrite the file
        tifffile.imwrite(file_path, img_to_save.astype(np.float32), imagej=True)
        return f"Processed {file_path.name}"

    except Exception as e:
        return f"Failed {file_path.name}: {e}"

# Determine number of workers (use N-1 cores or as appropriate)
num_cores = multiprocessing.cpu_count()
workers = max(1, num_cores - 1)
print(f"Using {workers} workers for parallel processing...")

# Run in parallel
results = Parallel(n_jobs=workers, backend='loky')( # 'loky' is often more robust
    delayed(process_file)(fp, band_stats, GLOBAL_CLIP_THRESHOLDS, APPLY_CLIPPING) for fp in tqdm(tif_files, desc="Pass 2/2: Normalizing")
)

# Optional: Print results/errors from parallel processing
processed_count = 0
error_count = 0
for res in results:
    if "Processed" in res:
        processed_count += 1
    else:
        error_count += 1
        print(res) # Print failures

print(f"\nParallel processing complete. Processed: {processed_count}, Failed/Skipped: {error_count}")
print(f"Files in {SAT_DIR} have been modified and saved in (Height, Width, Bands) format.")
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


Calculating final statistics...
Band 0: Mean=13735.6915, StdDev=2968.9758 (Calculated BEFORE clipping)
Band 1: Mean=15268.2139, StdDev=3569.4751 (Calculated BEFORE clipping)
Band 2: Mean=9861.2322, StdDev=1682.6679 (Calculated BEFORE clipping)
Band 3: Mean=9736.3202, StdDev=1663.2848 (Calculated BEFORE clipping)
Band 4: Mean=8932.5510, StdDev=1367.7162 (Calculated BEFORE clipping)
Band 6: Mean=43.6187, StdDev=57.8011 (Calculated BEFORE clipping)
'''