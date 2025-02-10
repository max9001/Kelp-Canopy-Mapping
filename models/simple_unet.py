# import torch
# import torch.nn as nn
# import pytorch_lightning as pl
# import tifffile as tiff
# import torch.optim as optim
# from torchmetrics import Accuracy, JaccardIndex
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# import sys
# from pyprojroot import here
# from pathlib import Path
# import os
# import numpy as np  # Import numpy

# dir_root = here()
# sys.path.append(str(dir_root))

# from utils.get_data import prepare_filenames  # Assuming this function is correct

# # --- Improved U-Net Model ---
# class UNet(pl.LightningModule):
#     def __init__(self, in_channels=7, out_channels=1, features=[64, 128, 256, 512]):
#         super(UNet, self).__init__()
#         self.ups = nn.ModuleList()
#         self.downs = nn.ModuleList()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Down part of UNet
#         for feature in features:
#             self.downs.append(DoubleConv(in_channels, feature))
#             in_channels = feature

#         # Up part of UNet
#         for feature in reversed(features):
#             self.ups.append(
#                 nn.ConvTranspose2d(
#                     feature * 2, feature, kernel_size=2, stride=2,
#                 )
#             )
#             self.ups.append(DoubleConv(feature * 2, feature))

#         self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
#         self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

#         self.loss_fn = nn.BCEWithLogitsLoss()  # More numerically stable than BCELoss with Sigmoid
#         self.accuracy = Accuracy(task="binary")
#         self.iou = JaccardIndex(task="binary") # Use JaccordIndex for IoU
        
#         self.output_dir = Path().resolve().parent / "output" / "predictions"
#         os.makedirs(self.output_dir, exist_ok=True)

#     def forward(self, x):
#         skip_connections = []

#         for down in self.downs:
#             x = down(x)
#             skip_connections.append(x)
#             x = self.pool(x)

#         x = self.bottleneck(x)
#         skip_connections = skip_connections[::-1]

#         for idx in range(0, len(self.ups), 2):
#             x = self.ups[idx](x)
#             skip_connection = skip_connections[idx // 2]

#             if x.shape != skip_connection.shape:
#                 x = transforms.functional.resize(x, size=skip_connection.shape[2:])

#             concat_skip = torch.cat((skip_connection, x), dim=1)
#             x = self.ups[idx + 1](concat_skip)

#         return self.final_conv(x)

#     def training_step(self, batch, batch_idx):
#         images, labels, _ = batch
#         preds = self(images)
#         loss = self.loss_fn(preds, labels)  # Use BCEWithLogitsLoss directly
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         images, labels, _ = batch
#         preds = self(images)
#         loss = self.loss_fn(preds, labels)
#         preds_binary = (torch.sigmoid(preds) > 0.5).float() # Apply sigmoid here for metrics
#         self.accuracy(preds_binary, labels)  # Update the metric
#         self.iou(preds_binary, labels)

#         self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('val_accuracy', self.accuracy, on_step=False, on_epoch=True)  # Log the metric
#         self.log('val_iou', self.iou, on_step=False, on_epoch=True)
#         return loss

#     def test_step(self, batch, batch_idx):
#         images, labels, filenames = batch
#         preds = self(images)
#         loss = self.loss_fn(preds, labels)

#         # Convert predictions to binary (threshold = 0.5) AFTER sigmoid
#         preds_binary = (torch.sigmoid(preds) > 0.5).float()

#         acc = self.accuracy(preds_binary, labels.int())
#         iou = self.iou(preds_binary, labels.int())

#         preds_np = preds_binary.cpu().numpy()
#         for i in range(preds_np.shape[0]):
#             file_name = filenames[i]
#             save_path = os.path.join(self.output_dir, file_name)
#             tiff.imwrite(save_path, (preds_np[i, 0] * 255).astype(np.uint8)) #use numpy uint8

#         self.log('test_loss', loss, on_step=False, on_epoch=True)
#         self.log('test_accuracy', acc, on_step=False, on_epoch=True)
#         self.log('test_iou', iou, on_step=False, on_epoch=True)
#         return {'test_loss': loss, 'test_accuracy': acc, 'test_iou': iou}

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-4)
#         return optimizer

# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),  # Kernel=3, stride=1, padding=1
#             nn.BatchNorm2d(out_channels),  # Batch normalization
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.conv(x)


# class KelpDataset(Dataset):
#     def __init__(self, image_filenames, mask_filenames, transform=None):
#         self.image_filenames = image_filenames
#         self.mask_filenames = mask_filenames
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_filenames)

#     def __getitem__(self, idx):
#         image_path = self.image_filenames[idx] #use paths.
#         mask_path = self.mask_filenames[idx]
        
#         image = tiff.imread(image_path).astype(np.float32) #use numpy floats.
#         mask = tiff.imread(mask_path).astype(np.float32)

#         # --- CRITICAL: Normalize the image data ---
#         for i in range(image.shape[0]):  # Iterate over channels
#             if i != 5: #do not normalize cloud layer
#                 channel = image[i, :, :]
#                 min_val = channel.min()
#                 max_val = channel.max()
#                 if max_val > min_val:  # Prevent division by zero
#                     image[i, :, :] = (channel - min_val) / (max_val - min_val)
#         # --- End of normalization ---

#         #ensure masks are only 0 or 1.
#         mask = np.where(mask > 0, 1.0, 0.0).astype(np.float32)


#         if self.transform:
#             # Apply the same transform to both image and mask using a dictionary
#             transformed = self.transform(image=image, mask=mask)
#             image = transformed["image"]
#             mask = transformed["mask"].unsqueeze(0) # Add channel dimension to mask


#         mask_filename = os.path.basename(mask_path)  # Corrected to use mask_path
#         return image, mask, mask_filename #make sure to unsqueeze the mask

    
#     def load_image(self, filename): #this function is no longer needed, kept for consistency
#       # Load the image using tiff.imread
#         img = tiff.imread(filename)
#         return img
    
#     def resize(self, image): #no longer needed
#       # Resize to the desired size (320x320)
#         return Image.fromarray(image).resize((320, 320))



# def main():
#     # Prepare dataset splits
#     filenames = prepare_filenames()
#     train_data, train_masks, val_data, val_masks, test_data, test_masks = filenames

#     # --- Use albumentations for transformations ---
#     import albumentations as A
#     from albumentations.pytorch import ToTensorV2

#     transform = A.Compose([
#         A.Resize(height=320, width=320), #use albumentations to resize.
#         # A.Normalize(mean=[0] * 7, std=[1] * 7),  # We are normalizing per-channel in dataset.
#         ToTensorV2(), #this automatically makes it between 0-1.
#     ])

#     # Create DataLoader
#     train_dataset = KelpDataset(train_data, train_masks, transform)
#     val_dataset = KelpDataset(val_data, val_masks, transform)
#     test_dataset = KelpDataset(test_data, test_masks, transform)

#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, persistent_workers=True)
#     val_loader = DataLoader(val_dataset, batch_size=4, num_workers=4, persistent_workers=True)
#     test_loader = DataLoader(test_dataset, batch_size=4, num_workers=4, persistent_workers=True)

#     # Initialize the UNet model
#     model = UNet()

#     # Initialize PyTorch Lightning Trainer
#     trainer = pl.Trainer(
#         accelerator="gpu" if torch.cuda.is_available() else "cpu",
#         devices=1,
#         max_epochs=10
#         # limit_train_batches=100, #for debugging.
#         # limit_val_batches=10,
#     )

#     # Train
#     trainer.fit(model, train_loader, val_loader)

#     # Test
#     trainer.test(model, test_loader)


# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
import pytorch_lightning as pl
import tifffile as tiff
import torch.optim as optim
from torchmetrics import Accuracy, JaccardIndex
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import sys
from pyprojroot import here
from pathlib import Path
import os
import numpy as np

dir_root = here()
sys.path.append(str(dir_root))

from utils.get_data import prepare_filenames  # Assuming this function is correct


class UNet(pl.LightningModule):
    def __init__(self, in_channels=7, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.loss_fn = nn.BCEWithLogitsLoss()  # More numerically stable
        self.accuracy = Accuracy(task="binary")
        self.iou = JaccardIndex(task="binary")
        
        self.output_dir = Path().resolve().parent / "output" / "predictions"
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_hyperparameters()  # Save hyperparameters for reproducibility


    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = transforms.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

    def training_step(self, batch, batch_idx):
        images, labels, _ = batch
        preds = self(images)
        loss = self.loss_fn(preds, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.shape[0]) #add batch size.
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, _ = batch
        preds = self(images)
        loss = self.loss_fn(preds, labels)
        preds_binary = (torch.sigmoid(preds) > 0.5).float()
        self.accuracy(preds_binary, labels)
        self.iou(preds_binary, labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=images.shape[0]) #add batch size.
        self.log('val_accuracy', self.accuracy, on_step=False, on_epoch=True, batch_size=images.shape[0])
        self.log('val_iou', self.iou, on_step=False, on_epoch=True, batch_size=images.shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        images, labels, filenames = batch
        preds = self(images)
        loss = self.loss_fn(preds, labels)

        preds_binary = (torch.sigmoid(preds) > 0.5).float()

        acc = self.accuracy(preds_binary, labels.int())
        iou = self.iou(preds_binary, labels.int())

        preds_np = preds_binary.cpu().numpy()
        for i in range(preds_np.shape[0]):
            file_name = filenames[i]
            save_path = os.path.join(self.output_dir, file_name)
            tiff.imwrite(save_path, (preds_np[i, 0] * 255).astype(np.uint8))

        self.log('test_loss', loss, on_step=False, on_epoch=True, batch_size=images.shape[0]) #add batch size.
        self.log('test_accuracy', acc, on_step=False, on_epoch=True, batch_size=images.shape[0])
        self.log('test_iou', iou, on_step=False, on_epoch=True, batch_size=images.shape[0])
        return {'test_loss': loss, 'test_accuracy': acc, 'test_iou': iou}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class KelpDataset(Dataset):
    def __init__(self, image_filenames, mask_filenames, transform=None):
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = self.image_filenames[idx]
        mask_path = self.mask_filenames[idx]

        image = tiff.imread(image_path).astype(np.float32)
        mask = tiff.imread(mask_path).astype(np.float32)

        for i in range(image.shape[0]):
            if i != 5:
                channel = image[i, :, :]
                min_val = channel.min()
                max_val = channel.max()
                if max_val > min_val:
                    image[i, :, :] = (channel - min_val) / (max_val - min_val)

        mask = np.where(mask > 0, 1.0, 0.0).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].unsqueeze(0)


        mask_filename = os.path.basename(mask_path)
        return image, mask, mask_filename


def main():
    filenames = prepare_filenames()
    train_data, train_masks, val_data, val_masks, test_data, test_masks = filenames

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(height=320, width=320),
        ToTensorV2(),
    ])

    train_dataset = KelpDataset(train_data, train_masks, transform)
    val_dataset = KelpDataset(val_data, val_masks, transform)
    test_dataset = KelpDataset(test_data, test_masks, transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=4, persistent_workers=True)

    model = UNet()

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=10
    )


    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()