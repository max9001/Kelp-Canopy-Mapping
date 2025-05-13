import random
import shutil
from pathlib import Path
from tqdm import tqdm
import math
import warnings # Added for potential warnings

# --- Configuration Constants ---
# *** SET THESE VALUES BEFORE RUNNING ***

# Path to the base directory containing 'train_satellite' and 'train_kelp' folders.
# Example: BASE_DIR_PATH_STR = "/path/to/your/data/project"
# Or assuming the script is run from the project root:
BASE_DIR_PATH_STR = Path().resolve().parent / "data" / "cleaned"

TRAIN_RATIO = 0.7    # Proportion for the training set (e.g., 0.7 for 70%)
VAL_RATIO = 0.15     # Proportion for the validation set (e.g., 0.15 for 15%)
# Test ratio is automatically calculated as 1.0 - TRAIN_RATIO - VAL_RATIO

SEED = 42           # Random seed for shuffling (for reproducibility)
MOVE_FILES = False  # Set to True to move files instead of copying (use with caution!)

# --- End Configuration ---


def create_splits(base_dir: Path, train_ratio: float, val_ratio: float, seed: int, move_files: bool = False):
    """
    Splits satellite and kelp data into train, validation, and test sets.

    Args:
        base_dir: The directory containing 'train_satellite' and 'train_kelp'.
        train_ratio: Proportion of data for the training set (e.g., 0.7).
        val_ratio: Proportion of data for the validation set (e.g., 0.15).
        seed: Random seed for shuffling to ensure reproducibility.
        move_files: If True, move files instead of copying. Defaults to False.
    """
    random.seed(seed)
    operation = shutil.move if move_files else shutil.copy2 # copy2 preserves metadata

    # --- Define Input and Output Paths ---
    sat_in_dir = base_dir / "train_satellite1"
    kelp_in_dir = base_dir / "train_kelp1"

    # Output directories (will be created within base_dir)
    # base_dir = Path().resolve().parent / "data" / "cleaned"
    out_dirs = {
        "train_sat": base_dir / "train_satellite",
        "train_gt": base_dir / "train_kelp", # gt = ground truth
        "val_sat": base_dir / "val_satellite",
        "val_gt": base_dir / "val_kelp",
        "test_sat": base_dir / "test_satellite",
        "test_gt": base_dir / "test_kelp",
    }

    # --- Validate Inputs ---
    if not sat_in_dir.is_dir():
        print(f"Error: Input satellite directory not found: {sat_in_dir}")
        return
    if not kelp_in_dir.is_dir():
        print(f"Error: Input kelp directory not found: {kelp_in_dir}")
        return
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and (train_ratio + val_ratio) < 1):
         print(f"Error: Invalid ratios. train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be less than 1.")
         return

    # --- Create Output Directories ---
    print("Creating output directories...")
    for dir_path in out_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # --- Find Paired Files ---
    print("Finding paired satellite and kelp files...")
    sat_files = {p.stem.replace("_satellite", ""): p for p in sat_in_dir.glob("*_satellite.tif")}
    kelp_files = {p.stem.replace("_kelp", ""): p for p in kelp_in_dir.glob("*_kelp.tif")}

    # Find the common base filenames (tile IDs) that have both files
    common_ids = sorted(list(sat_files.keys() & kelp_files.keys())) # Use set intersection

    if not common_ids:
        print("Error: No matching pairs of satellite and kelp files found.")
        return

    num_total = len(common_ids)
    print(f"Found {num_total} paired files.")
    if len(sat_files) != num_total or len(kelp_files) != num_total:
        warnings.warn(f"{len(sat_files) - num_total} satellite files had no matching kelp file.")
        warnings.warn(f"{len(kelp_files) - num_total} kelp files had no matching satellite file.")
        warnings.warn("Only paired files will be split.")

    # --- Shuffle and Split IDs ---
    print(f"Shuffling and splitting {num_total} IDs (seed={seed})...")
    random.shuffle(common_ids)

    test_ratio = 1.0 - train_ratio - val_ratio
    num_train = math.floor(num_total * train_ratio)
    num_val = math.floor(num_total * val_ratio)
    # Assign remaining to test set to ensure all files are used
    num_test = num_total - num_train - num_val

    train_ids = common_ids[:num_train]
    val_ids = common_ids[num_train : num_train + num_val]
    test_ids = common_ids[num_train + num_val :]

    print(f"  Train: {len(train_ids)} pairs ({len(train_ids)/num_total:.2%})")
    print(f"  Validation: {len(val_ids)} pairs ({len(val_ids)/num_total:.2%})")
    print(f"  Test: {len(test_ids)} pairs ({len(test_ids)/num_total:.2%})")
    assert len(train_ids) + len(val_ids) + len(test_ids) == num_total

    # --- Define Transfer Function ---
    def transfer_files(ids, sat_dest_key, gt_dest_key):
        """Copies or moves files for a given list of IDs to destination folders."""
        sat_dest_dir = out_dirs[sat_dest_key]
        gt_dest_dir = out_dirs[gt_dest_key]
        op_name = "Moving" if move_files else "Copying"
        print(f"\n{op_name} {len(ids)} pairs to {sat_dest_dir.name} and {gt_dest_dir.name}...")

        for tile_id in tqdm(ids, desc=f"{op_name} {sat_dest_key.split('_')[0]}"):
            try:
                # Source paths
                src_sat_path = sat_files[tile_id]
                src_kelp_path = kelp_files[tile_id]

                # Destination paths
                dest_sat_path = sat_dest_dir / src_sat_path.name
                dest_kelp_path = gt_dest_dir / src_kelp_path.name

                # Perform operation
                if src_sat_path.exists():
                     operation(src_sat_path, dest_sat_path)
                else:
                     warnings.warn(f"Source file missing during transfer: {src_sat_path}")
                if src_kelp_path.exists():
                     operation(src_kelp_path, dest_kelp_path)
                else:
                    warnings.warn(f"Source file missing during transfer: {src_kelp_path}")

            except KeyError:
                warnings.warn(f"ID {tile_id} found during initial scan but missing during transfer.")
            except Exception as e:
                print(f"Error transferring files for ID {tile_id}: {e}")

    # --- Execute File Transfers ---
    transfer_files(train_ids, "train_sat", "train_gt")
    transfer_files(val_ids, "val_sat", "val_gt")
    transfer_files(test_ids, "test_sat", "test_gt")

    action = "moved" if move_files else "copied"
    print(f"\nSplit complete. Files {action} to:")
    for name, path in out_dirs.items():
        print(f"  - {name}: {path}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Convert the string path to a Path object
    base_directory = Path(BASE_DIR_PATH_STR)

    # Call the function with the constants defined above
    create_splits(base_directory, TRAIN_RATIO, VAL_RATIO, SEED, MOVE_FILES)