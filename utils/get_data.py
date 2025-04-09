from pathlib import Path
import os
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

def prepare_filenames(option="tile"):

    if option == "original":
        print("Loading Original Dataset...")
        return prepare_filenames_original()

    if option == "tile":
        print("Loading Tiled Dataset...")
        return prepare_filenames_tiled()

def prepare_filenames_original():
    train, val, test = split_dataset_original()

    train_masks = get_gt_filenames(train)
    train_data = get_st_filenames(train)

    val_masks = get_gt_filenames(val)
    val_data = get_st_filenames(val)

    test_masks = get_gt_filenames(test)
    test_data = get_st_filenames(test)

    return [train_data, train_masks, val_data, val_masks, test_data, test_masks]


def split_dataset_original():

    # dir_root = here()
    # directory = str(dir_root)
    # filenames = np.array(os.listdir(directory + "/data/train_kelp/"))

    dir_root = Path().resolve().parent  # Equivalent to here()
    directory = dir_root / "data" / "train_kelp"
    filenames = np.array([f.name for f in directory.iterdir() if f.is_file()])
    filenames = [f[:8] for f in filenames]

    # 60% of the data for training, 20% for validation, and 20% for testing.
    train_val_files, test_files = train_test_split(filenames, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_val_files, test_size=0.25, random_state=42)

    return train_files, val_files, test_files
    
def get_gt_filenames(filenames):
    # dir_root = here()
    # directory = str(dir_root)
    # return [directory + "/data/train_kelp/" + f + "_kelp.tif" for f in filenames]

    dir_root = Path().resolve().parent
    directory = dir_root / "data" / "train_kelp"     
    return [str(directory / f"{f}_kelp.tif") for f in filenames]

def get_st_filenames(filenames):
    # dir_root = here()
    # directory = str(dir_root)
    # return [directory + "/data/train_satellite/" + f + "_satellite.tif" for f in filenames]

    dir_root = Path().resolve().parent
    directory = dir_root / "data" / "train_satellite"  
    return [str(directory / f"{f}_satellite.tif") for f in filenames]

# -------------------------------------------------------------------------------------------



def prepare_filenames_tiled():
    """
    Prepares lists of corresponding satellite and mask file paths for
    train, validation, and test sets based on the files present in the
    balanced dataset directory.
    """
    # Get the lists of actual kelp mask file paths for each split
    train_kelp_paths, val_kelp_paths, test_kelp_paths = split_dataset_tiled()

    # Define the base directory for the balanced data
    dir_root = Path().resolve().parent
    balanced_data_dir = dir_root / "data" / "balanced_tiled_40_60"
    satellite_dir = balanced_data_dir / "train_satellite"

    # --- Generate corresponding satellite file paths ---
    def get_corresponding_satellite_paths(kelp_paths):
        satellite_paths = []
        for kelp_path in kelp_paths:
            # kelp_path is already a Path object from split_dataset
            satellite_name = kelp_path.name.replace("_kelp.tif", "_satellite.tif")
            satellite_path = satellite_dir / satellite_name
            # Important: Convert Path object to string for functions expecting strings
            satellite_paths.append(str(satellite_path))
        return satellite_paths

    train_satellite_paths = get_corresponding_satellite_paths(train_kelp_paths)
    val_satellite_paths = get_corresponding_satellite_paths(val_kelp_paths)
    test_satellite_paths = get_corresponding_satellite_paths(test_kelp_paths)

    # Convert kelp Path objects to strings for consistency
    train_mask_paths = [str(p) for p in train_kelp_paths]
    val_mask_paths = [str(p) for p in val_kelp_paths]
    test_mask_paths = [str(p) for p in test_kelp_paths]


    print(f"Train samples: {len(train_satellite_paths)}")
    print(f"Validation samples: {len(val_satellite_paths)}")
    print(f"Test samples: {len(test_satellite_paths)}")

    # Example check for the first file pair (optional debugging)
    if train_satellite_paths:
        print(f"Example Train Pair:\n  Sat: {train_satellite_paths[0]}\n  Mask: {train_mask_paths[0]}")


    return [train_satellite_paths, train_mask_paths,
            val_satellite_paths, val_mask_paths,
            test_satellite_paths, test_mask_paths]


def split_dataset_tiled():
    """
    Scans the balanced kelp directory and splits the *existing* file paths
    into train, validation, and test sets.
    """
    dir_root = Path().resolve().parent
    # --- Point directly to the kelp masks in the BALANCED directory ---
    kelp_directory = dir_root / "data" / "balanced_tiled_40_60" / "train_kelp"

    if not kelp_directory.exists():
        raise FileNotFoundError(f"Balanced kelp directory not found: {kelp_directory}")

    # --- Get a list of Path objects for all existing kelp files ---
    print(f"Scanning for existing files in: {kelp_directory}")
    all_kelp_files = list(kelp_directory.glob('*_kelp.tif'))
    print(f"Found {len(all_kelp_files)} kelp files in the balanced directory.")

    if not all_kelp_files:
        raise ValueError(f"No '*_kelp.tif' files found in {kelp_directory}. Did the balancing script run correctly?")

    # --- Split the list of existing Path objects ---
    # 60% train, 20% validation, 20% test (0.25 of the remaining 80% is 20%)
    train_val_files, test_files = train_test_split(all_kelp_files, test_size=0.2, random_state=42, shuffle=True)
    train_files, val_files = train_test_split(train_val_files, test_size=0.25, random_state=42, shuffle=True) # 0.25 * 0.8 = 0.2

    # Return lists of Path objects
    return train_files, val_files, test_files