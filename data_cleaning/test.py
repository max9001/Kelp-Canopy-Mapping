from pathlib import Path
from PIL import Image
import numpy as np


directory = Path().resolve().parent

filename = "AA498489"

original_img_dir = str(directory / "data"/ "original" / "train_kelp" / f"{filename}_kelp.tif")
cleaned_img_dir = str(directory / "data"/ "cleaned" / "train_kelp" / f"{filename}_kelp.tif")

original_img = np.array(Image.open(original_img_dir))
cleaned_img = np.array(Image.open(cleaned_img_dir))

print()
print(original_img.shape)
print(cleaned_img.shape)