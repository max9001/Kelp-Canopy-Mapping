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
import torch.nn.functional as F  # Import functional for Dice Loss
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping  # Import callbacks
from lightning.pytorch.tuner import Tuner  # Import Tuner directly


dir_root = here()
sys.path.append(str(dir_root))

from utils.get_data import prepare_filenames

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs are logits, targets are labels (0 or 1)
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to get probabilities
        inputs = inputs.view(-1)  # Flatten
        targets = targets.view(-1)  # Flatten

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice

class UNet(pl.LightningModule):
    def __init__(self, lr=1e-4):  # Add learning rate as a hyperparameter
        super(UNet, self).__init__()
        self.save_hyperparameters() #save it
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

        self.loss_fn = DiceLoss()
        self.accuracy = Accuracy(task="binary")
        self.iou = JaccardIndex(task="binary")
        self.output_dir = Path().resolve().parent / "output" / "predictions"
        os.makedirs(self.output_dir, exist_ok=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.loss_fn(preds, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.loss_fn(preds, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.loss_fn(preds, labels)
        preds_binary = (torch.sigmoid(preds) > 0.5).float()
        acc = self.accuracy(preds_binary, labels.int())
        iou = self.iou(preds_binary, labels.int())
        preds_np = preds_binary.cpu().numpy()
        for i in range(preds_np.shape[0]):
            filename = os.path.join(self.output_dir, f"mask_{batch_idx}_{i}.tif")
            tiff.imwrite(filename, (preds_np[i, 0] * 255).astype("uint8"))
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)
        self.log('test_iou', iou)
        return {'test_loss': loss, 'test_accuracy': acc, 'test_iou': iou}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass


class KelpDataset(Dataset):
    def __init__(self, image_filenames, mask_filenames, transform=None):
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):

        # assert os.path.exists(self.image_filenames[idx]), f"File not found: {self.image_filenames[idx]}"

        image = self.load_image(self.image_filenames[idx])
        mask = self.load_image(self.mask_filenames[idx])

        # Convert image to float32
        image = image.astype('float32')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Ensure mask is binary and float32
        mask = mask.float()

        return image, mask

    def load_image(self, filename):
        img = tiff.imread(filename)
        return img


def main():
    # Prepare dataset splits
    filenames = prepare_filenames(sys.argv[1])
    train_data, train_masks, val_data, val_masks, test_data, test_masks = filenames

    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Create DataLoader
    train_dataset = KelpDataset(train_data, train_masks, transform)
    val_dataset = KelpDataset(val_data, val_masks, transform)
    test_dataset = KelpDataset(test_data, test_masks, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=4, pin_memory = True, persistent_workers=True)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',   # Quantity to monitor (must match the key logged in validation_step)
        patience=10,          # Number of epochs with no improvement after which training will be stopped. Adjust as needed.
        verbose=True,         # Print info when stopping
        mode='min'            # Stop when the quantity monitored has stopped decreasing (correct for loss)
    )

    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=4, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=4, num_workers=4)

    # Initialize the UNet model
    model = UNet(lr = 5e-5)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=200, 
        # callbacks=[early_stop_callback],
        log_every_n_steps=10,  # Log more frequently
    )

    # # --- Learning Rate Finder (Correct Usage) ---
    # #0.005754399373371567
    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(
    #     model,
    #     train_dataloaders=train_loader,
    #     val_dataloaders=val_loader,
    #     min_lr=1e-8,  # Optional: Set min/max LR
    #     max_lr=1.0   # Optional
    # )

    # # Get the suggested learning rate
    # suggested_lr = lr_finder.suggestion()
    # print(f"Suggested learning rate: {suggested_lr}")

    # # Update the model's learning rate
    # model.hparams.lr = suggested_lr
    # model.configure_optimizers() #re-init optimizer


    # Train and test the model
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()