# import os
# import shutil
# from pathlib import Path
# from pyprojroot import here
# from tqdm import tqdm  # Import tqdm for progress bar
# import numpy as np
# from PIL import Image #for reading in files
# import heapq  # Import for efficient top-k selection

# def copy_balanced_data(source_satellite_dir, source_kelp_dir, dest_satellite_dir, dest_kelp_dir, top_k_dict):
#     """
#     Copies satellite images and corresponding kelp masks to new directories,
#     using a dictionary of filenames to select the files.

#     Args:
#         source_satellite_dir (str or Path): Path to the source satellite image directory.
#         source_kelp_dir (str or Path): Path to the source kelp mask directory.
#         dest_satellite_dir (str or Path): Path to the destination satellite image directory.
#         dest_kelp_dir (str or Path): Path to the destination kelp mask directory.
#         top_k_dict (dict): Dictionary where keys are filenames (without extensions)
#                            and values are pixel counts. This determines which files are copied.
#     """
#     source_satellite_dir = Path(source_satellite_dir)
#     source_kelp_dir = Path(source_kelp_dir)
#     dest_satellite_dir = Path(dest_satellite_dir)
#     dest_kelp_dir = Path(dest_kelp_dir)

#     os.makedirs(dest_satellite_dir, exist_ok=True)
#     os.makedirs(dest_kelp_dir, exist_ok=True)

#     # Iterate through the filenames in the dictionary
#     for filename in tqdm(top_k_dict, desc="Copying Files"):
#         # Construct full paths for source and destination
#         base_filename = filename[:-9]  # Remove "_kelp.tif" to get base
#         satellite_src = source_satellite_dir / f"{base_filename}_satellite.tif"
#         kelp_src = source_kelp_dir / f"{base_filename}_kelp.tif"
#         satellite_dest = dest_satellite_dir / f"{base_filename}_satellite.tif"
#         kelp_dest = dest_kelp_dir / f"{base_filename}_kelp.tif"

#         # Copy satellite image
#         if satellite_src.exists():
#             try:
#                 shutil.copy2(satellite_src, satellite_dest)
#             except Exception as e:
#                 print(f"Error copying {satellite_src}: {e}")
#         else:
#             print(f"Warning: Satellite file not found: {satellite_src}")

#         # Copy corresponding kelp mask
#         if kelp_src.exists():
#             try:
#                 shutil.copy2(kelp_src, kelp_dest)
#             except Exception as e:
#                 print(f"Error copying {kelp_src}: {e}")
#         else:
#             print(f"Warning: Kelp file not found: {kelp_src}")


# def calculate_kelp_pixel_counts(directory):
#     """Calculates kelp pixel counts, returning (count, filename) tuples."""
#     kelp_counts = []
#     filenames = [f for f in directory.iterdir() if f.is_file() and f.name.endswith('_kelp.tif')]

#     for filename in tqdm(filenames, desc="Processing Images"):
#         try:
#             image_GT = Image.open(filename)
#             image_GT = np.array(image_GT)
#             kelp_count = np.sum(image_GT == 1)
#             kelp_counts.append((kelp_count, filename.name))
#         except Exception as e:
#             print(f"Error processing {filename}: {e}")

#     return kelp_counts

# def get_top_k_counts(kelp_counts_with_filenames, k=1000):
#     """Efficiently gets the top k (count, filename) pairs."""
#     if not kelp_counts_with_filenames or k <= 0:
#         return {}
#     if k > len(kelp_counts_with_filenames):
#         k = len(kelp_counts_with_filenames)

#     top_k_list = heapq.nlargest(k, kelp_counts_with_filenames, key=lambda x: x[0])
#     top_k_dict = {filename: count for count, filename in top_k_list}
#     return top_k_dict

# def main():
#     root_dir = here()
#     source_satellite_dir = root_dir / "data" / "train_satellite"
#     source_kelp_dir = root_dir / "data" / "train_kelp"
#     dest_satellite_dir = root_dir / "data" / "train_balanced_satellite"  # New destination
#     dest_kelp_dir = root_dir / "data" / "train_balanced_kelp"       # New destination

#     if not source_satellite_dir.exists():
#         raise FileNotFoundError(f"Source satellite directory not found: {source_satellite_dir}")
#     if not source_kelp_dir.exists():
#         raise FileNotFoundError(f"Source kelp directory not found: {source_kelp_dir}")

#     kelp_counts_with_filenames = calculate_kelp_pixel_counts(source_kelp_dir)
#     top_1000_counts = get_top_k_counts(kelp_counts_with_filenames, k=1000)


#     # Copy the selected files to the new directories
#     copy_balanced_data(source_satellite_dir, source_kelp_dir, dest_satellite_dir, dest_kelp_dir, top_1000_counts)
#     print(f"Copied balanced data to {dest_satellite_dir} and {dest_kelp_dir}")

# if __name__ == "__main__":
#     main()



import os
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np
from pyprojroot import here
from tqdm import tqdm

def scan_kelp_files(source_kelp_dir):
    """
    Scans the source directory to classify files based on kelp presence.

    Args:
        source_kelp_dir (Path): Path to the directory containing kelp mask files.

    Returns:
        tuple: (list of Path objects for zero-kelp files,
                list of Path objects for non-zero-kelp files)
    """
    zero_kelp_files = []
    non_zero_kelp_files = []

    # Use tqdm for scanning progress
    file_iterator = list(source_kelp_dir.glob('*_kelp.tif')) # Get all potential files first
    for kelp_path in tqdm(file_iterator, desc="Scanning kelp files"):
        if not kelp_path.is_file():
            continue # Skip if not a file (though glob should handle this)
        try:
            img = Image.open(kelp_path)
            img_array = np.array(img)
            kelp_count = np.sum(img_array == 1)

            if kelp_count == 0:
                zero_kelp_files.append(kelp_path)
            else:
                non_zero_kelp_files.append(kelp_path)
        except Exception as e:
            print(f"Warning: Error processing {kelp_path.name}: {e}")

    return zero_kelp_files, non_zero_kelp_files

def balance_and_copy_data(source_kelp_dir, source_satellite_dir, dest_dir,
                          target_total_samples, target_zero_percentage=0.4):
    """
    Creates a new balanced dataset by copying files.

    Args:
        source_kelp_dir (Path): Source directory for kelp masks.
        source_satellite_dir (Path): Source directory for satellite images.
        dest_dir (Path): Destination directory for the new balanced dataset.
        target_total_samples (int): The total number of samples desired in the new dataset.
        target_zero_percentage (float): The desired proportion of zero-kelp images (0.0 to 1.0).
    """
    print("--- Starting Data Balancing and Copying ---")

    # 1. Scan source directory
    zero_kelp_files, non_zero_kelp_files = scan_kelp_files(source_kelp_dir)
    print(f"Found {len(zero_kelp_files)} images with zero kelp.")
    print(f"Found {len(non_zero_kelp_files)} images with non-zero kelp.")

    if not zero_kelp_files and not non_zero_kelp_files:
        print("Error: No kelp files found in the source directory.")
        return

    # 2. Calculate target counts
    target_zero_count = int(target_total_samples * target_zero_percentage)
    target_non_zero_count = target_total_samples - target_zero_count
    print(f"Targeting {target_zero_count} zero-kelp images ({target_zero_percentage*100:.1f}%).")
    print(f"Targeting {target_non_zero_count} non-zero-kelp images ({(1-target_zero_percentage)*100:.1f}%).")

    # 3. Check availability
    if target_zero_count > len(zero_kelp_files):
        print(f"Warning: Requested {target_zero_count} zero-kelp images, but only {len(zero_kelp_files)} available. Using all available.")
        target_zero_count = len(zero_kelp_files)
        # Recalculate non-zero count if needed to maintain total (optional, depends on priority)
        # target_non_zero_count = target_total_samples - target_zero_count

    if target_non_zero_count > len(non_zero_kelp_files):
        print(f"Warning: Requested {target_non_zero_count} non-zero-kelp images, but only {len(non_zero_kelp_files)} available. Using all available.")
        target_non_zero_count = len(non_zero_kelp_files)
        # Recalculate zero count if needed (optional)
        # target_zero_count = target_total_samples - target_non_zero_count

    actual_total = target_zero_count + target_non_zero_count
    if actual_total != target_total_samples:
         print(f"Note: Actual total samples will be {actual_total} due to availability.")


    # 4. Random Sampling
    selected_zero_files = random.sample(zero_kelp_files, target_zero_count)
    selected_non_zero_files = random.sample(non_zero_kelp_files, target_non_zero_count)
    selected_files = selected_zero_files + selected_non_zero_files
    random.shuffle(selected_files) # Shuffle the combined list

    print(f"Selected {len(selected_files)} files for the new dataset.")

    # 5. Create Destination Directories
    dest_kelp_dir = dest_dir / "train_kelp"
    dest_satellite_dir = dest_dir / "train_satellite"
    os.makedirs(dest_kelp_dir, exist_ok=True)
    os.makedirs(dest_satellite_dir, exist_ok=True)
    print(f"Created destination directories: {dest_kelp_dir} and {dest_satellite_dir}")

    # 6. Copy Files
    copied_count = 0
    for kelp_path in tqdm(selected_files, desc="Copying files"):
        try:
            # Copy kelp file
            dest_kelp_path = dest_kelp_dir / kelp_path.name
            shutil.copy2(kelp_path, dest_kelp_path)

            # Construct and copy corresponding satellite file
            satellite_name = kelp_path.name.replace("_kelp.tif", "_satellite.tif")
            source_satellite_path = source_satellite_dir / satellite_name
            dest_satellite_path = dest_satellite_dir / satellite_name

            if source_satellite_path.exists():
                shutil.copy2(source_satellite_path, dest_satellite_path)
                copied_count += 1
            else:
                print(f"Warning: Satellite file not found for {kelp_path.name}. Kelp mask copied, but satellite image missing.")
                # Optionally remove the copied kelp mask if satellite is required
                # os.remove(dest_kelp_path)

        except Exception as e:
            print(f"Error copying files for {kelp_path.name}: {e}")

    print(f"--- Finished Copying ---")
    print(f"Successfully copied {copied_count} pairs of kelp and satellite images to {dest_dir}")
    if copied_count != len(selected_files):
        print(f"Note: {len(selected_files) - copied_count} satellite images were missing.")


def main():
    """
    Main function to set up directories and run the balancing process.
    """
    root_dir = here()
    source_kelp_dir = root_dir / "data" / "tiled_kelp"
    source_satellite_dir = root_dir / "data" / "tiled_satellite"
    dest_dir = root_dir / "data" / "balanced_tiled_40_60" # Descriptive name

    # --- Configuration ---
    TARGET_TOTAL_SAMPLES = 100000  # Adjust this to your desired dataset size
    TARGET_ZERO_PERCENTAGE = 0.50 # 40% zero kelp

    # --- Run the process ---
    if not source_kelp_dir.exists():
        raise FileNotFoundError(f"Source kelp directory not found: {source_kelp_dir}")
    if not source_satellite_dir.exists():
        raise FileNotFoundError(f"Source satellite directory not found: {source_satellite_dir}")

    balance_and_copy_data(source_kelp_dir, source_satellite_dir, dest_dir,
                          TARGET_TOTAL_SAMPLES, TARGET_ZERO_PERCENTAGE)

if __name__ == "__main__":
    main()