import matplotlib.pyplot as plt
import numpy as np
import random
import tifffile as tiff
from PIL import Image
import os

def normalize_band(band):
    return (band - band.min()) / (band.max() - band.min())

def gamma_correction(image, gamma=1.0):
    return np.power(image, gamma)

directory = "data/train_kelp/"
filenames = os.listdir(directory)

filename = random.choice(filenames)
filename = filename[:-9]

# filename = "AA498489"

GT_img = "kaggle_dataset/train_kelp/" + filename + "_kelp.tif"
ST_img = "kaggle_dataset/train_satellite/" + filename + "_satellite.tif"

image_GT = Image.open(GT_img)
image_GT = np.array(image_GT)

image_ST = tiff.imread(ST_img)
image_ST = np.array(image_ST)

# Band description:
# 	0: Short-wave infrared (SWIR)
# 	1: Near infrared (NIR)
# 	2: Red 
# 	3: Green 
# 	4: Blue 
# 	5: Cloud Mask (binary - is there cloud or not)
# 	6: Digital Elevation Model (meters above sea-level)

rgb_image = np.dstack((
    normalize_band(image_ST[:, :, 2]),  
    normalize_band(image_ST[:, :, 3]),  
    normalize_band(image_ST[:, :, 4])   
))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Satellite Image", fontsize=25)
plt.imshow(rgb_image)

plt.subplot(1, 2, 2)
plt.title("Labeled Kelp", fontsize=25)
plt.imshow(rgb_image)
plt.imshow(image_GT, cmap='Wistia', alpha=image_GT)

# plt.tight_layout()
plt.show()


