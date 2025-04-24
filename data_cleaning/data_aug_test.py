import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from pyprojroot import here
from tqdm import tqdm
import random
import string

def generate_random_string(length=8):
    """Generates a random string of specified length."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def flip_and_save(image_path, dest_dir, hflip=False, vflip=False):
    """
    Flips an image and saves it with a random 8-character filename.
    """
    try:
        img = Image.open(image_path)
        if hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if vflip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        random_string = generate_random_string()
        new_filename = f"{random_string}_kelp.tif"
        new_filepath = dest_dir / new_filename

        img.save(new_filepath)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def augment_images(source_dir, dest_dir, threshold):
    """
    Augments images based on kelp pixel count, and copies others unchanged.
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    for image_path in tqdm(list(source_dir.glob("*.tif")), desc="Processing Images"):
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            kelp_count = np.sum(img_array == 1)

            if kelp_count > threshold:
                # Augment and save (flipped versions)
                flip_and_save(image_path, dest_dir, hflip=True, vflip=False)
                flip_and_save(image_path, dest_dir, hflip=False, vflip=True)
                flip_and_save(image_path, dest_dir, hflip=True, vflip=True)
            else:
                # Copy original image with a new random name
                random_string = generate_random_string()
                new_filename = f"{random_string}_kelp.tif"
                new_filepath = dest_dir / new_filename
                shutil.copy2(image_path, new_filepath)  # Copy original

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def main():
    """
    Main function.
    """
    root_dir = here()
    source_dir = root_dir / "data" / "train100"
    dest_dir = root_dir / "data" / "train100_augmented"
    threshold = 1  # Example threshold

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    augment_images(source_dir, dest_dir, threshold)
    print(f"Augmented and copied images saved to {dest_dir}")

if __name__ == "__main__":
    main()