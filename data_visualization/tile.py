import os
from pathlib import Path
import tifffile
import numpy as np
from tqdm import tqdm
# from pyprojroot import here # Removed as per request

'''
used to tile the dataset
was going to explore how deleting meaningless tiles (all land, no kelp) would affect training
never ended up exploring this avenue
'''

# --- Configuration Constants ---
ROOT_DIR = Path().resolve().parent
SOURCE_SATELLITE_DIR = ROOT_DIR / "data" / "original" / "train_satellite"
SOURCE_MASK_DIR = ROOT_DIR / "data" / "original" / "train_kelp"

DEST_SATELLITE_DIR = ROOT_DIR / "data" / "tiled_satellite"
DEST_MASK_DIR = ROOT_DIR / "data" / "tiled_kelp"

# --- Tiling Parameters ---
TILE_SIZE = 175
# Assuming original images are 350x350, adjust if necessary
ORIG_H, ORIG_W = 350, 350
# --- End Configuration Constants ---


def tile_image(image_path: Path, output_dir: Path):
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
        # Example: AA498489_satellite.tif -> parts = ["AA498489", "satellite"]
        parts = image_path.stem.split('_')
        if len(parts) < 2:
             print(f"Warning: Skipping {image_path.name} - Unexpected filename format (expected 'basename_type').")
             return
        original_base_name = parts[0] # The part before the first underscore
        file_type = parts[-1] # The part right before ".tif" (e.g., "satellite" or "kelp")

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
                    print(f"Warning: Unexpected image dimensions {image.ndim} for {image_path.name}. Skipping this image.")
                    continue

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
    Uses globally defined directory paths.
    """
    # Create destination directories if they don't exist
    # os.makedirs can handle Path objects directly
    print(f"Creating output directories (if needed):")
    print(f"  - {DEST_SATELLITE_DIR}")
    print(f"  - {DEST_MASK_DIR}")
    os.makedirs(DEST_SATELLITE_DIR, exist_ok=True)
    os.makedirs(DEST_MASK_DIR, exist_ok=True)
    # Alternatively, using Path.mkdir:
    # DEST_SATELLITE_DIR.mkdir(parents=True, exist_ok=True)
    # DEST_MASK_DIR.mkdir(parents=True, exist_ok=True)

    # Check if source directories exist
    if not SOURCE_SATELLITE_DIR.exists():
        raise FileNotFoundError(f"Source satellite directory not found: {SOURCE_SATELLITE_DIR}")
    if not SOURCE_MASK_DIR.exists():
        raise FileNotFoundError(f"Source mask directory not found: {SOURCE_MASK_DIR}")

    # Get a list of satellite images to process
    # Iterating through satellite images ensures we process pairs
    satellite_files = list(SOURCE_SATELLITE_DIR.glob("*_satellite.tif"))

    if not satellite_files:
        print(f"No satellite images found in {SOURCE_SATELLITE_DIR}")
        return

    print(f"\nFound {len(satellite_files)} satellite images to tile.")

    # Process each satellite image and its corresponding mask
    for sat_path in tqdm(satellite_files, desc="Tiling Images"):
        # Derive the corresponding mask filename
        # Assumes base_name is the part before the first underscore
        base_name = sat_path.stem.split('_')[0]
        mask_filename = f"{base_name}_kelp.tif"
        mask_path = SOURCE_MASK_DIR / mask_filename

        # Check if the mask file actually exists before proceeding
        if not mask_path.exists():
            print(f"Warning: Mask file not found for {sat_path.name} (expected at {mask_path}), skipping.")
            continue

        # Tile the satellite image
        tile_image(sat_path, DEST_SATELLITE_DIR)

        # Tile the mask image
        tile_image(mask_path, DEST_MASK_DIR)

    print("\nTiling complete.")
    print(f"Tiled satellite images saved to: {DEST_SATELLITE_DIR}")
    print(f"Tiled mask images saved to: {DEST_MASK_DIR}")


if __name__ == "__main__":
    main()