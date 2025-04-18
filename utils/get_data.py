from pathlib import Path
import numpy as np
import warnings

# --- Configuration ---
# Define the base directory where the split folders are located.
# Adjust this path as needed. Assumes script is in the project root
# or one level below (e.g., in a 'scripts' folder).
try:
    # Assumes script is in a 'scripts' subdir relative to 'data'
    BASE_DIR = Path().resolve().parent / "data" / "cleaned"
    if not BASE_DIR.exists():
        # If not found, assume script is in project root
        BASE_DIR = Path().resolve() / "data" / "cleaned"
        if not BASE_DIR.exists():
            raise FileNotFoundError("Could not automatically find the 'data/cleaned' directory.")
except Exception as e:
     raise FileNotFoundError(f"Error determining base directory: {e}")


# --- Main Function ---

def prepare_filenames(base_dir: Path = BASE_DIR):
    """
    Loads lists of satellite and ground truth file paths from pre-split
    directories (train_sat, train_gt, val_sat, val_gt, test_sat, test_gt).

    Args:
        base_dir: The Path object pointing to the directory containing
                  the six split folders (e.g., 'data/cleaned').

    Returns:
        A list containing six lists of file paths (absolute paths as strings):
        [train_sat_paths, train_gt_paths,
         val_sat_paths,   val_gt_paths,
         test_sat_paths,  test_gt_paths]
    """
    print(f"Loading filenames from pre-split directories in: {base_dir}")

    # Define the expected input directories based on the base_dir
    split_dirs = {
        "train_sat": base_dir / "train_satellite",
        "train_gt": base_dir / "train_kelp", # gt = ground truth
        "val_sat": base_dir / "val_satellite",
        "val_gt": base_dir / "val_kelp",
        "test_sat": base_dir / "test_satellite",
        "test_gt": base_dir / "test_kelp",
    }

    # --- Validate Directories ---
    all_dirs_exist = True
    for name, dir_path in split_dirs.items():
        if not dir_path.is_dir():
            warnings.warn(f"Required directory not found: {dir_path}")
            all_dirs_exist = False
    if not all_dirs_exist:
        raise FileNotFoundError("One or more required split directories are missing.")

    # --- Load File Paths ---
    def get_paths_from_dir(dir_path: Path, suffix: str):
        """Helper to get sorted list of absolute file paths as strings."""
        # Use glob to find files, convert Paths to absolute strings, and sort
        paths = sorted([str(p.resolve()) for p in dir_path.glob(f"*{suffix}")])
        return paths

    print("Reading file paths...")
    train_sat_paths = get_paths_from_dir(split_dirs["train_sat"], "_satellite.tif")
    train_gt_paths = get_paths_from_dir(split_dirs["train_gt"], "_kelp.tif")
    val_sat_paths = get_paths_from_dir(split_dirs["val_sat"], "_satellite.tif")
    val_gt_paths = get_paths_from_dir(split_dirs["val_gt"], "_kelp.tif")
    test_sat_paths = get_paths_from_dir(split_dirs["test_sat"], "_satellite.tif")
    test_gt_paths = get_paths_from_dir(split_dirs["test_gt"], "_kelp.tif")

    # --- Perform Sanity Checks (Optional but Recommended) ---
    if len(train_sat_paths) != len(train_gt_paths):
        warnings.warn(f"Mismatch in train set counts: {len(train_sat_paths)} satellite vs {len(train_gt_paths)} ground truth files.")
    if len(val_sat_paths) != len(val_gt_paths):
        warnings.warn(f"Mismatch in validation set counts: {len(val_sat_paths)} satellite vs {len(val_gt_paths)} ground truth files.")
    if len(test_sat_paths) != len(test_gt_paths):
        warnings.warn(f"Mismatch in test set counts: {len(test_sat_paths)} satellite vs {len(test_gt_paths)} ground truth files.")

    print(f"Found {len(train_sat_paths)} training samples.")
    print(f"Found {len(val_sat_paths)} validation samples.")
    print(f"Found {len(test_sat_paths)} test samples.")

    # Example check for the first file pair (optional debugging)
    if train_sat_paths:
        # Extract just the filename for easier comparison
        sat_name = Path(train_sat_paths[0]).name
        gt_name = Path(train_gt_paths[0]).name
        print(f"Example Train Pair Check:\n  Sat File: {sat_name}\n  GT File:  {gt_name}")
        # Basic check if base names match
        if sat_name.replace("_satellite.tif", "") != gt_name.replace("_kelp.tif", ""):
             warnings.warn("First train satellite and GT filenames do not seem to correspond!")

    return [train_sat_paths, train_gt_paths,
            val_sat_paths, val_gt_paths,
            test_sat_paths, test_gt_paths]


# --- Example Usage ---
if __name__ == "__main__":
    try:
        # Call the main function using the configured BASE_DIR
        all_filenames = prepare_filenames()

        # Unpack the results (optional)
        train_data, train_masks, val_data, val_masks, test_data, test_masks = all_filenames

        print("\nSuccessfully loaded filenames.")
        # print(f"\nFirst 5 train satellite files:\n{train_data[:5]}")
        # print(f"\nFirst 5 train mask files:\n{train_masks[:5]}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")