import os
from pathlib import Path
import tifffile
import numpy as np
from tqdm import tqdm
from pyprojroot import here

# --- Configuration ---
TILE_SIZE = 25
# Assuming original images are 350x350
ORIG_H, ORIG_W = 350, 350
# ---------------------

def tile_image(image_path, output_dir):
    """
    Reads an image, tiles it into TILE_SIZE x TILE_SIZE patches,
    and saves each tile with a descriptive filename.

    Args:
        image_path (Path): Path to the input image file (.tif).
        output_dir (Path): Directory to save the tiled images.
    """
    try:
        image = tifffile.imread(image_path)

        # Basic check for expected dimensions
        if image.shape[0] != ORIG_H or image.shape[1] != ORIG_W:
            print(f"Warning: Skipping {image_path.name} - Unexpected dimensions {image.shape[:2]}, expected {ORIG_H}x{ORIG_W}")
            return

        # Extract original base name (e.g., AA498489) and type (satellite/kelp)
        parts = image_path.stem.split('_')
        if len(parts) < 2:
             print(f"Warning: Skipping {image_path.name} - Unexpected filename format.")
             return
        original_base_name = parts[0]
        file_type = parts[-1] # Assumes type is the last part before .tif

        num_tiles_h = ORIG_H // TILE_SIZE
        num_tiles_w = ORIG_W // TILE_SIZE

        for r in range(num_tiles_h):
            for c in range(num_tiles_w):
                start_row = r * TILE_SIZE
                end_row = start_row + TILE_SIZE
                start_col = c * TILE_SIZE
                end_col = start_col + TILE_SIZE

                # Extract the tile using numpy slicing
                if image.ndim == 3:  # Satellite image (H, W, C)
                    tile = image[start_row:end_row, start_col:end_col, :]
                elif image.ndim == 2:  # Mask image (H, W)
                    tile = image[start_row:end_row, start_col:end_col]
                else:
                    print(f"Warning: Unexpected image dimensions {image.ndim} for {image_path.name}")
                    continue # Skip this image if dimensions are weird

                # Construct the output filename
                # Format: {OriginalName}_r{RowIndex}_c{ColIndex}_{Type}.tif
                # Using 2 digits for row/col index ensures correct sorting later
                tile_filename = f"{original_base_name}_r{r:02d}_c{c:02d}_{file_type}.tif"
                output_path = output_dir / tile_filename

                # Save the tile
                tifffile.imwrite(output_path, tile)

    except FileNotFoundError:
        print(f"Error: File not found {image_path}")
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")


def main():
    """
    Main function to find images, create directories, and initiate tiling.
    """
    root_dir = here()

    # --- Define Source and Destination Directories ---
    source_satellite_dir = root_dir / "data" / "train_satellite"
    source_mask_dir = root_dir / "data" / "train_kelp"

    dest_satellite_dir = root_dir / "data" / "tiled_satellite"
    dest_mask_dir = root_dir / "data" / "tiled_kelp"
    # -------------------------------------------------

    # Create destination directories if they don't exist
    print(f"Creating output directories (if needed):")
    print(f"  - {dest_satellite_dir}")
    print(f"  - {dest_mask_dir}")
    os.makedirs(dest_satellite_dir, exist_ok=True)
    os.makedirs(dest_mask_dir, exist_ok=True)

    # Check if source directories exist
    if not source_satellite_dir.exists():
        raise FileNotFoundError(f"Source satellite directory not found: {source_satellite_dir}")
    if not source_mask_dir.exists():
        raise FileNotFoundError(f"Source mask directory not found: {source_mask_dir}")

    # Get a list of satellite images to process
    # Iterating through satellite images ensures we process pairs
    satellite_files = list(source_satellite_dir.glob("*_satellite.tif"))

    if not satellite_files:
        print(f"No satellite images found in {source_satellite_dir}")
        return

    print(f"\nFound {len(satellite_files)} satellite images to tile.")

    # Process each satellite image and its corresponding mask
    for sat_path in tqdm(satellite_files, desc="Tiling Images"):
        # Derive the corresponding mask filename
        base_name = sat_path.stem.split('_')[0]
        mask_filename = f"{base_name}_kelp.tif"
        mask_path = source_mask_dir / mask_filename

        # Check if the mask file actually exists before proceeding
        if not mask_path.exists():
            print(f"Warning: Mask file not found for {sat_path.name}, skipping.")
            continue

        # Tile the satellite image
        tile_image(sat_path, dest_satellite_dir)

        # Tile the mask image
        tile_image(mask_path, dest_mask_dir)

    print("\nTiling complete.")
    print(f"Tiled satellite images saved to: {dest_satellite_dir}")
    print(f"Tiled mask images saved to: {dest_mask_dir}")


if __name__ == "__main__":
    main()