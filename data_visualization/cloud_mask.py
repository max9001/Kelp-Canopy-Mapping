import numpy as np
import tifffile as tiff
from pathlib import Path
import os
from tqdm import tqdm
import heapq  # For efficient top-k selection
import sys
from pyprojroot import here # Optional, if you use it for root finding

def calculate_cloud_counts(directory):
    """
    Calculates the number of cloud pixels (value 1 in band 5) for each satellite image.

    Args:
        directory (Path): Path to the directory containing satellite image files
                          (e.g., ending in '_satellite.tif').

    Returns:
        list: A list of (count, filename) tuples, where count is the number
              of cloud pixels and filename is the name of the satellite image file.
              Returns an empty list if no suitable files are found or errors occur.
    """
    cloud_counts = []
    # Find all satellite image files
    satellite_files = list(directory.glob("*_satellite.tif"))

    if not satellite_files:
        print(f"Warning: No '*_satellite.tif' files found in {directory}")
        return []

    print(f"Found {len(satellite_files)} satellite files. Calculating cloud counts...")

    for file_path in tqdm(satellite_files, desc="Processing Satellite Images"):
        try:
            # Load the multi-band satellite image
            image_st = tiff.imread(file_path)
            image_st = np.array(image_st) # Ensure it's a NumPy array

            # Check dimensions - expecting (Height, Width, Channels)
            if image_st.ndim != 3 or image_st.shape[-1] < 6:
                print(f"Warning: Skipping {file_path.name} due to unexpected shape: {image_st.shape}")
                continue

            # Extract the cloud mask band (index 5)
            cloud_mask_band = image_st[:, :, 5]

            # Count the number of pixels where the cloud mask is 1
            cloud_pixel_count = np.sum(cloud_mask_band == 1)

            # Store the count and the filename
            cloud_counts.append((cloud_pixel_count, file_path.name))

        except Exception as e:
            print(f"Error processing satellite image {file_path.name}: {e}")

    return cloud_counts

def get_kelp_count(kelp_mask_path):
    """Loads a kelp mask and returns the count of kelp pixels (value 1)."""
    try:
        kelp_mask = tiff.imread(kelp_mask_path)
        kelp_mask = np.array(kelp_mask)
        # Ensure mask is binary (handle potential non-binary values if necessary)
        kelp_mask = (kelp_mask == 1).astype(np.uint8)
        kelp_pixel_count = np.sum(kelp_mask)
        return kelp_pixel_count
    except FileNotFoundError:
        return "Not Found"
    except Exception as e:
        print(f"Error reading kelp mask {kelp_mask_path.name}: {e}")
        return "Error"


def main():
    """
    Finds the top 5 satellite images with the most clouds and prints their
    cloud and corresponding kelp pixel counts.
    """
    try:
        # --- Define Directories ---
        # root_dir = here()
        root_dir = Path().resolve().parent

        # Directory containing the satellite images
        satellite_dir = root_dir / "data" / "train_satellite" # Example: Original data
        # satellite_dir = root_dir / "data" / "tiled_satellite"
        # satellite_dir = root_dir / "data" / "balanced_tiled_40_60" / "train_satellite"

        # Directory containing the corresponding kelp masks
        kelp_mask_dir = root_dir / "data" / "train_kelp" # Example: Original data
        # kelp_mask_dir = root_dir / "data" / "tiled_kelp"
        # kelp_mask_dir = root_dir / "data" / "balanced_tiled_40_60" / "train_kelp"


        if not satellite_dir.exists():
            raise FileNotFoundError(f"Satellite image directory not found: {satellite_dir}")
        if not kelp_mask_dir.exists():
            raise FileNotFoundError(f"Kelp mask directory not found: {kelp_mask_dir}")


        # --- Calculate Cloud Counts ---
        cloud_counts_with_filenames = calculate_cloud_counts(satellite_dir)

        if not cloud_counts_with_filenames:
            print("No cloud counts were calculated. Exiting.")
            return

        # --- Find Top 5 Cloudy Images ---
        k = 20
        top_k_cloudy_files = heapq.nlargest(k, cloud_counts_with_filenames, key=lambda item: item[0])

        print(f"\n--- Top {k} Files with Most Cloud Pixels (and their Kelp Counts) ---")
        if not top_k_cloudy_files:
            print(f"No files found to determine top {k}.")
        else:
            num_to_print = min(k, len(top_k_cloudy_files))
            for i in range(num_to_print):
                cloud_count, satellite_filename = top_k_cloudy_files[i]

                # --- Find Corresponding Kelp Mask and Get Kelp Count ---
                base_filename = satellite_filename.replace("_satellite.tif", "")
                kelp_filename = f"{base_filename}_kelp.tif"
                kelp_path = kelp_mask_dir / kelp_filename

                kelp_count = get_kelp_count(kelp_path)
                # --- End Kelp Count ---

                print(f"{i+1}. {satellite_filename} (Cloud Pixels: {cloud_count}, Kelp Pixels: {kelp_count})")

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()