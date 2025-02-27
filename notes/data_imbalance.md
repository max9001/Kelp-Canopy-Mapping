# Why is current implementations generating a mask of all 0s everytime?

- There is a major data imbalance:
    - 37.4% of all images don't even have kelp
    - images have an average of 830 kelp pixels per image
        - 830/(350x350) = 0.0067755102 
        - model will get 99.34% accuracy by just guessing 0 every time

# Options to fix

### loss functions

    - Binary Cross Entropy Loss is probably not the best choice for this data imabalance

    - Dice Loss (or Tversky Loss)
        - Directly optimizes the Dice coefficient (similar to Intersection over Union, IoU)
        - Dice loss is less sensitive to class imbalance than cross-entropy. 


### data augmentation

    - augment examples with kelp pixels to supplement data 
        - horizontal flip
        - vertical flip
        - both

        lets see if this does anything:

            before processing:

                Total number of images processed: 100
                Number of images with zero kelp pixels: 38
                Maximum kelp pixel count: 11481
                Minimum kelp pixel count: 0
                Average kelp pixel count: 730.46
                Median kelp pixel count: 138.50

            after doing 3 flips to imgs with > 100 kelp pixels

                Total number of images processed: 206
                Number of images with zero kelp pixels: 38
                Maximum kelp pixel count: 11481
                Minimum kelp pixel count: 0
                Average kelp pixel count: 1061.03
                Median kelp pixel count: 507.00



