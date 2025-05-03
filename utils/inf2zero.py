import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
import warnings

# --- Configuration ---
# Define the base directory containing the 'original' data splits
try:
    # Assumes the script is run from a directory where '../data/original' exists
    BASE_DIR = Path().resolve().parent / "data" / "original"
    if not BASE_DIR.is_dir():
         BASE_DIR = Path().resolve() / "data" / "original"
         if not BASE_DIR.is_dir():
              raise FileNotFoundError("Could not automatically find the 'data/original' directory.")
except Exception as e:
    raise FileNotFoundError(f"Error setting base directory: {e}")

# Define the subdirectories to process
SUBDIRS = [
    "train_satellite",
    "train_kelp",
    "val_satellite",
    "val_kelp",
    "test_satellite",
    "test_kelp",
]

# Data type for processing and saving (float32 is required for NaN/inf)
PROCESSING_DTYPE = np.float32

# Value to replace NaN/Inf with
REPLACEMENT_VALUE = 0.0

# --- Main Processing Logic ---

def clean_image_file(file_path: Path):
    """Loads a TIFF image, replaces NaN and Inf values with 0, and saves if modified."""
    modified = False
    try:
        # Load the image
        original_image = tifffile.imread(file_path)

        # Work with a floating-point copy
        image_data = original_image.astype(PROCESSING_DTYPE, copy=True)

        # ============================================================
        # ADDED: Check for both NaN and Inf
        # ============================================================
        # Find infinite values (positive and negative)
        inf_mask = np.isinf(image_data)
        # Find NaN values
        nan_mask = np.isnan(image_data)
        # Combine masks: Find where either NaN or Inf is present
        bad_values_mask = inf_mask | nan_mask
        # ============================================================

        # Check if any bad values were found
        if np.any(bad_values_mask):
            num_inf = np.sum(inf_mask)
            num_nan = np.sum(nan_mask)
            print(f"  Found {num_nan} NaN / {num_inf} Inf values in {file_path.name}. Replacing with {REPLACEMENT_VALUE}...")

            # Replace bad values with the specified replacement value
            image_data[bad_values_mask] = REPLACEMENT_VALUE
            modified = True # Mark as modified

            # Save the modified image back to the original path
            try:
                tifffile.imwrite(file_path, image_data, imagej=True)
            except Exception as write_e:
                warnings.warn(f"  Error saving modified file {file_path.name}: {write_e}")
                modified = False # Revert modified status if save failed
        # else:
            # Optional: print if no bad values found
            # print(f"  No NaN or Inf values found in {file_path.name}.")

    except FileNotFoundError:
        warnings.warn(f"  File not found during processing: {file_path}. Skipping.")
    except Exception as read_e:
        warnings.warn(f"  Error reading/processing file {file_path.name}: {read_e}. Skipping.")

    return modified # Return whether the file was successfully modified and saved

# --- Script Execution ---
if __name__ == "__main__":
    print("="*50)
    print("WARNING: This script modifies image files in place.")
    print(f"Target base directory: {BASE_DIR}")
    print(f"It will replace NaN and Inf values with: {REPLACEMENT_VALUE}")
    print("Please ensure you have a backup before proceeding.")
    print("="*50)
    # Uncomment the following lines for a confirmation prompt
    # confirmation = input("Type 'yes' to continue: ")
    # if confirmation.lower() != 'yes':
    #     print("Operation cancelled.")
    #     exit()

    total_files_scanned = 0
    total_files_modified = 0

    print("\nStarting NaN/Inf replacement process...")

    for subdir_name in SUBDIRS:
        subdir_path = BASE_DIR / subdir_name
        print(f"\nProcessing directory: {subdir_path}...")

        if not subdir_path.is_dir():
            warnings.warn(f"  Directory not found. Skipping.")
            continue

        tif_files = list(subdir_path.glob('*.tif'))
        if not tif_files:
            print("  No .tif files found in this directory.")
            continue

        print(f"  Found {len(tif_files)} .tif files.")
        files_modified_in_dir = 0
        total_files_scanned += len(tif_files)

        for file_path in tqdm(tif_files, desc=f"  {subdir_name}", unit="file"):
            if clean_image_file(file_path):
                files_modified_in_dir += 1

        if files_modified_in_dir > 0:
            print(f"  Modified {files_modified_in_dir} file(s) in this directory.")
        total_files_modified += files_modified_in_dir

    print("\n" + "="*50)
    print("Processing complete.")
    print(f"Total files scanned: {total_files_scanned}")
    print(f"Total files modified (NaN/Inf replaced): {total_files_modified}")
    print("="*50)