import matplotlib.pyplot as plt
import numpy as np
import random
import tifffile as tiff
from pathlib import Path

# Get the output directory
directory = Path().resolve().parent / "output" / "predictions"

# Get a list of all image filenames
filenames = np.array([f.name for f in directory.iterdir() if f.is_file()])

# Keep searching for an image with at least one nonzero pixel
valid_image = None
max_attempts = len(filenames)  # Prevent infinite loops

for _ in range(max_attempts):
    filename = random.choice(filenames)
    output_mask = tiff.imread(directory / filename)

    if np.any(output_mask):  # Check if there's at least one nonzero pixel
        valid_image = output_mask
        break

# print(valid_image)
if valid_image is not None:
    plt.figure(figsize=(12, 6))
    plt.title(f"Predicted Image: {filename}", fontsize=25)
    plt.imshow(valid_image)  # Ensure grayscale visualization
    plt.show()
else:
    print("No nonzero masks found in the dataset.")
