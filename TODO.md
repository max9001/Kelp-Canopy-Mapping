- [ ] delete old "custom" datasets. 

- [ ] clean data. VC433864 should definitely be deleted. see if there are outliers for standard deviation, or cases where min/max is unusual. do this per band.

- [ ] for each band, Calculate the mean and standard deviation  across all images in the training set only. Apply the normalization (pixel_value - channel_mean) / channel_std to all images (train, validation, and test) using the means and standard deviations calculated from the training set.

- [ ] paritition cleaned 350x350 set - need to reserve images for test regenerate tiles, balanced tiles, etc

- [ ] create `test.py` that takes data and some kind of model weights as input, generates masks from input data, calculates the 4 metrics (IOU, precision, recall, f1)

- [ ] retrain on 3 datasets. save model weights or something similar

- [ ] test unmodified data on model trained on unmodified data, get 4 metrics

- [ ] test data on model trained on data thats only been tiled (no deletion). tile test set, generate masks, reconstruct and get 4 metrics

- [ ] test data on model trained on tiled + deleted data. tile test set, generate masks, reconstruct and get 4 metrics

***now caught up to where I am now, just with things done correctly***

- [ ] implement some kind of augmentation. 50% chance that a given augmentation will be applied (H flip, V flip, 90 rotation clockwise, counter clockwise, color jitter)

- [ ] expirament with other loss functions.

    
 