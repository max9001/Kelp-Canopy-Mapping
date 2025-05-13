import tifffile as tiff
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import warnings
from collections import defaultdict

# --- Configuration ---
# Assume the script is run from the project root or adjust as needed
base_dir = Path().resolve()
if not (base_dir / "data").exists():
     # A common structure might be src/scripts, data/
     base_dir = base_dir.parent
     if not (base_dir / "data").exists():
         # Try one level higher for cases like project_root/notebooks
         base_dir = base_dir.parent
         if not (base_dir / "data").exists():
            raise FileNotFoundError("Could not automatically find the 'data' directory relative to the script/notebook. Please adjust 'base_dir'.")


# Adjust these paths if your structure is different
data_dir = base_dir / "data" / "original"
sat_dir = data_dir / "train_satellite"
# kelp_dir = data_dir / "train_kelp" # Not strictly needed for this script

# Check if satellite directory exists
if not sat_dir.is_dir():
    raise FileNotFoundError(f"Satellite image directory not found: {sat_dir}")

# Band names for easier reference (Matching the description and previous script)
BAND_NAMES = [
    "0_SWIR",
    "1_NIR",
    "2_Red",
    "3_Green",
    "4_Blue",
    "5_CloudMask",
    "6_DEM"
]
NUM_BANDS = len(BAND_NAMES)

# --- Main Analysis ---

print("Starting Analysis: Finding images with max counts of extreme pixel values.")

# Get list of satellite images
sat_files = list(sat_dir.glob("*_satellite.tif"))
if not sat_files:
    print(f"Error: No satellite TIFF files found in {sat_dir}")
    sys.exit(1)

print(f"Found {len(sat_files)} satellite images.")

# --- Step 1: Find Global Min/Max for Each Band ---
print("\n--- Pass 1: Determining global min/max values for each band ---")

# Initialize dictionaries to store global min/max
global_min_max = {band_name: {'min': np.inf, 'max': -np.inf} for band_name in BAND_NAMES}
processed_files_pass1 = 0
error_files_pass1 = []

for f_path in tqdm(sat_files, desc="Pass 1/2: Finding Min/Max"):
    try:
        img = tiff.imread(f_path)

        # Basic validation
        if img.ndim != 3 or img.shape[2] != NUM_BANDS:
            warnings.warn(f"Skipping {f_path.name}: Incorrect shape {img.shape}")
            error_files_pass1.append(f"{f_path.name} (Bad Shape: {img.shape})")
            continue
        # Add dtype check if necessary, e.g., if you expect uint16
        # if img.dtype != np.uint16:
        #     warnings.warn(f"Skipping {f_path.name}: Unexpected dtype {img.dtype}")
        #     error_files_pass1.append(f"{f_path.name} (Bad Dtype: {img.dtype})")
        #     continue

        for i, band_name in enumerate(BAND_NAMES):
            band_data = img[:, :, i]
            current_min = np.min(band_data)
            current_max = np.max(band_data)

            if current_min < global_min_max[band_name]['min']:
                global_min_max[band_name]['min'] = current_min
            if current_max > global_min_max[band_name]['max']:
                global_min_max[band_name]['max'] = current_max
        processed_files_pass1 += 1

    except FileNotFoundError:
        warnings.warn(f"File not found during pass 1: {f_path.name}")
        error_files_pass1.append(f"{f_path.name} (File Not Found)")
    except Exception as e:
        warnings.warn(f"Error reading/processing file {f_path.name} during pass 1: {e}")
        error_files_pass1.append(f"{f_path.name} (Read Error: {e})")

print(f"\nPass 1 Complete. Processed {processed_files_pass1} files.")
if error_files_pass1:
    print(f"Encountered errors in {len(error_files_pass1)} files during Pass 1.")

print("\nGlobal Min/Max Values Found:")
for band_name, values in global_min_max.items():
    print(f"  {band_name}: Min = {values['min']}, Max = {values['max']}")


# --- Step 2: Count Pixels Matching Global Min/Max in Each Image ---
print("\n--- Pass 2: Counting pixels matching global extremes in each image ---")

# Structure to store the filename and count for the image with the most extreme pixels
# Format: results[band_name]['min'/'max'] = {'filename': '...', 'count': N}
results = {
    band_name: {
        'min': {'filename': None, 'count': -1},
        'max': {'filename': None, 'count': -1}
    } for band_name in BAND_NAMES
}

processed_files_pass2 = 0
error_files_pass2 = []

for f_path in tqdm(sat_files, desc="Pass 2/2: Counting Extremes"):
    base_filename = f_path.stem.replace('_satellite', '')
    try:
        img = tiff.imread(f_path)

        # Basic validation (repeat check in case some files were readable but weird)
        if img.ndim != 3 or img.shape[2] != NUM_BANDS:
            # Warning issued in pass 1, just skip here
             if f"{f_path.name} (Bad Shape: {img.shape})" not in error_files_pass1 and f"{f_path.name} (Read Error: {e})" not in error_files_pass1:
                warnings.warn(f"Skipping {f_path.name} in pass 2 due to unexpected shape: {img.shape}")
                error_files_pass2.append(f"{f_path.name} (Bad Shape: {img.shape})")
             continue
        # Add dtype check if needed

        for i, band_name in enumerate(BAND_NAMES):
            band_data = img[:, :, i]
            global_min = global_min_max[band_name]['min']
            global_max = global_min_max[band_name]['max']

            # Count pixels matching global min
            min_count = np.sum(band_data == global_min)
            # Count pixels matching global max
            max_count = np.sum(band_data == global_max)

            # Check if this image has more min-value pixels than the current leader
            if min_count > results[band_name]['min']['count']:
                results[band_name]['min']['filename'] = base_filename
                results[band_name]['min']['count'] = min_count

            # Check if this image has more max-value pixels than the current leader
            if max_count > results[band_name]['max']['count']:
                results[band_name]['max']['filename'] = base_filename
                results[band_name]['max']['count'] = max_count
        processed_files_pass2 += 1

    except FileNotFoundError:
         # Should not happen if pass 1 succeeded, but handle defensively
        if f"{f_path.name} (File Not Found)" not in error_files_pass1:
             warnings.warn(f"File not found during pass 2: {f_path.name}")
             error_files_pass2.append(f"{f_path.name} (File Not Found)")
    except Exception as e:
         if f"{f_path.name} (Read Error: {e})" not in error_files_pass1:
             warnings.warn(f"Error reading/processing file {f_path.name} during pass 2: {e}")
             error_files_pass2.append(f"{f_path.name} (Read Error: {e})")


print(f"\nPass 2 Complete. Processed {processed_files_pass2} files.")
if error_files_pass2:
    print(f"Encountered additional errors in {len(error_files_pass2)} files during Pass 2.")

# --- Step 3: Report Results ---
print("\n--- Files to Inspect (Highest Count of Extreme Pixels) ---")

for band_name in BAND_NAMES:
    print(f"\nBand: {band_name}")

    # Min Value Report
    min_info = results[band_name]['min']
    global_min_val = global_min_max[band_name]['min']
    if min_info['filename']:
        print(f"  Max Count of MIN Value ({global_min_val}):")
        print(f"    Filename: {min_info['filename']}")
        print(f"    Count:    {min_info['count']} pixels")
    else:
        print(f"  Max Count of MIN Value ({global_min_val}): No valid file found.")

    # Max Value Report
    max_info = results[band_name]['max']
    global_max_val = global_min_max[band_name]['max']
    if max_info['filename']:
        print(f"  Max Count of MAX Value ({global_max_val}):")
        print(f"    Filename: {max_info['filename']}")
        print(f"    Count:    {max_info['count']} pixels")
    else:
        print(f"  Max Count of MAX Value ({global_max_val}): No valid file found.")

print("\nAnalysis complete.")




'''
Band: 0_SWIR
  Max Count of MIN Value (-32768):
    Filename: WK510339
    Count:    103601 pixels
  Max Count of MAX Value (65535)=
    Filename: MG668025
    Count:    1357 pixels

Band: 1_NIR
  Max Count of MIN Value (-32768):
    Filename: WK510339
    Count:    103601 pixels
  Max Count of MAX Value (65535):
    Filename: PK637108
    Count:    350 pixels

Band: 2_Red
  Max Count of MIN Value (-32768):
    Filename: WK510339
    Count:    103601 pixels
  Max Count of MAX Value (65535):
    Filename: PK637108
    Count:    4404 pixels

Band: 3_Green
  Max Count of MIN Value (-32768):
    Filename: WK510339
    Count:    103601 pixels
  Max Count of MAX Value (65535):
    Filename: HH695604
    Count:    806 pixels

Band: 4_Blue
  Max Count of MIN Value (-32768):
    Filename: WK510339
    Count:    103601 pixels
  Max Count of MAX Value (65535):
    Filename: UX385857
    Count:    13412 pixels

Band: 5_CloudMask
  Max Count of MIN Value (0):
    Filename: AB602248
    Count:    122500 pixels
  Max Count of MAX Value (1):
    Filename: HG365959
    Count:    119915 pixels

Band: 6_DEM
  Max Count of MIN Value (-32768):
    Filename: AB440131
    Count:    122500 pixels
  Max Count of MAX Value (706):
    Filename: UF158815
    Count:    1 pixels



analysis:
    nothing needs to be done to band 5 or 6
        maybe cloud mask later if time permits
'''