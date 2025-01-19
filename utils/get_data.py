from pyprojroot import here
import os
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_filenames():
    train, val, test = split_dataset()

    train_masks = get_gt_filenames(train)
    train_data = get_st_filenames(train)

    val_masks = get_gt_filenames(val)
    val_data = get_st_filenames(val)

    test_masks = get_gt_filenames(test)
    test_data = get_st_filenames(test)

    return [train_data, train_masks, val_data, val_masks, test_data, test_masks]


def split_dataset():

    dir_root = here()
    directory = str(dir_root)
    filenames = np.array(os.listdir(directory + "/data/train_kelp/"))
    filenames = [f[:8] for f in filenames]

    # 60% of the data for training, 20% for validation, and 20% for testing.
    train_val_files, test_files = train_test_split(filenames, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_val_files, test_size=0.25, random_state=42)

    return train_files, val_files, test_files
    
def get_gt_filenames(filenames):
    dir_root = here()
    directory = str(dir_root)
    return [directory + "/data/train_kelp/" + f + "_kelp.tif" for f in filenames]

def get_st_filenames(filenames):
    dir_root = here()
    directory = str(dir_root)
    return [directory + "/data/train_satellite/" + f + "_satellite.tif" for f in filenames]
