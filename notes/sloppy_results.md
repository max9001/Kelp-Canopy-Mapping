Curruent standings

UNET Trained from scratch, no data preprocessing

    0.9932244898% accuracy
    0.0 IOU

    model just guesses 0 every time. massive dataset imbalance

        below methods were attempted ONLY on the untrained UNET. no improvements seen

            tiling
            dice loss
            weighted bce loss
            augmentations
            weighted augmentations (high kelp pixels augmented, others not)

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

Train with pretrained resnet18 weights, no data preprocessing ~ 150epochs

    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃        Test metric        ┃       DataLoader 0        ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │       test_accuracy       │    0.9945759177207947     │
    │         test_iou          │    0.26340940594673157    │
    │         test_loss         │    0.0451936274766922     │
    └───────────────────────────┴───────────────────────────┘


Tile 350x350 images into 49 25x25 tiles. tile with 0 kelp pixels to predict were erased. resulting dataset was a 50/50 split - half of the images had 0 kelp pixels to predict, half had at least one kelp pixel in the tile. trained on resnet18 weights for 200 epochs. no other preprocessing

    ? accuracy
    ~0.49 IOU

    TESTED on the unbalanced dataset... probably should test on the original data somehow.

---------------------------------------------------------------------

beginning of explatory data analysis

    VC433864 - weird stripes -> likely corrupted

    Lots of images where practically the whole image is covered by clouds... but most have 0 kelp pixels. keep these or not?

    have elevation model... need to apply to output mask (obv no kelp pixels anywhere but where sea level is 0)


---------------------------------------------------------------------
