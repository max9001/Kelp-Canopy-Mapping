import os
import shutil
import random
from pathlib import Path
from pyprojroot import here

def copy_random_samples(source_dir, dest_dir, num_samples=100):
    """
    Copies a specified number of random samples from the source directory to the
    destination directory.

    Args:
        source_dir (str or Path): Path to the source directory.
        dest_dir (str or Path): Path to the destination directory.
        num_samples (int): Number of samples to copy.
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Get all .tif files
    all_files = [f for f in source_dir.glob('*.tif')]

    if not all_files:
        print(f"No .tif files found in {source_dir}")
        return

    if len(all_files) < num_samples:
        raise ValueError(f"Requested {num_samples} samples, but only {len(all_files)} files are available.")

    # Randomly sample filenames without replacement
    sampled_files = random.sample(all_files, num_samples)

    for file_path in sampled_files:
        try:
            shutil.copy2(file_path, dest_dir / file_path.name)  # Use copy2 to preserve metadata
        except Exception as e:
            print(f"Error copying {file_path}: {e}")


def main():
    """
    Main function to copy 100 random training samples to data/train100/.
    """
    root_dir = here()
    source_kelp_dir = root_dir / "data" / "train_kelp"  # Corrected source directory
    dest_dir = root_dir / "data" / "train100"

    # Make sure source directory exists
    if not source_kelp_dir.exists():
        raise FileNotFoundError(f"Source kelp directory not found: {source_kelp_dir}")


    # Copy kelp files
    copy_random_samples(source_kelp_dir, dest_dir)
    print(f"Copied {100} random samples to {dest_dir}")

if __name__ == "__main__":
    main()