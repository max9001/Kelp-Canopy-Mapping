# https://github.com/milesial/Pytorch-UNet/tree/master

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
import torch.nn.functional as F  # Import F

dir_root = here()
sys.path.append(str(dir_root))
from utils.get_data import prepare_filenames

# --- U-Net Parts (Integrated) ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# --- End U-Net Parts ---


class UNet(pl.LightningModule):
    def __init__(self, n_channels=7, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task="binary")
        self.iou = JaccardIndex(task="binary")

        # Ensure output directory exists
        self.output_dir = Path().resolve().parent / "output" / "predictions"
        os.makedirs(self.output_dir, exist_ok=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        labels = labels.float()
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        labels = labels.float()
        loss = self.loss_fn(logits, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        labels = labels.float()
        loss = self.loss_fn(logits, labels)

        # Apply sigmoid and threshold
        preds = torch.sigmoid(logits)
        preds_binary = (preds > 0.5).float()

        acc = self.accuracy(preds_binary, labels.int())
        iou = self.iou(preds_binary, labels.int())

        # Save predictions
        preds_np = preds_binary.cpu().numpy()
        for i in range(preds_np.shape[0]):
            filename = os.path.join(self.output_dir, f"mask_{batch_idx}_{i}.tif")
            tiff.imwrite(filename, (preds_np[i, 0] * 255).astype('uint8'))

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_accuracy', acc, on_step=False, on_epoch=True)
        self.log('test_iou', iou, on_step=False, on_epoch=True)
        return {'test_loss': loss, 'test_accuracy': acc, 'test_iou': iou}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class KelpDataset(Dataset):
    def __init__(self, image_filenames, mask_filenames, transform=None):
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = self.load_image(self.image_filenames[idx])
        mask = self.load_image(self.mask_filenames[idx])
        image = image.astype('float32')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        mask = mask.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask

    def load_image(self, filename):
        img = tiff.imread(filename)
        return img


def main():
    # Prepare dataset splits
    filenames = prepare_filenames()
    train_data, train_masks, val_data, val_masks, test_data, test_masks = filenames

    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create Datasets
    train_dataset = KelpDataset(train_data, train_masks, transform)
    val_dataset = KelpDataset(val_data, val_masks, transform)
    test_dataset = KelpDataset(test_data, test_masks, transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=4)

    # Initialize the UNet model
    model = UNet(n_channels=7, n_classes=1, bilinear=False)

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=1,
        # limit_train_batches=100,  # Uncomment for debugging
        # limit_val_batches=10,    # Uncomment for debugging
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Test
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()