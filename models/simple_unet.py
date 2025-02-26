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

dir_root = here()
sys.path.append(str(dir_root))

from utils.get_data import prepare_filenames

# Define a simple U-Net model for semantic segmentation
class UNet(pl.LightningModule):
    def __init__(self):
        super(UNet, self).__init__()
        # A simple U-Net structure with 2 convolutional blocks
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=3, padding=1),  # 7 input channels (RGB+NIR)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # Output channel is 1 (binary)
            nn.Sigmoid()  # Sigmoid activation for binary segmentation
        )
        self.loss_fn = nn.BCELoss()  # Binary Cross Entropy Loss
        self.accuracy = Accuracy(task="binary")  # Binary accuracy
        self.iou = JaccardIndex(task="binary")  # IoU for binary segmentation

        
        # Ensure output directory exists
        self.output_dir = Path().resolve().parent / "output" / "predictions" 
        os.makedirs(self.output_dir, exist_ok=True)


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.loss_fn(preds, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.loss_fn(preds, labels)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.loss_fn(preds, labels)

        # Convert predictions to binary (threshold = 0.5)
        preds_binary = (preds > 0.5).float()

        acc = self.accuracy(preds_binary, labels.int())  # Binary accuracy
        iou = self.iou(preds_binary, labels.int())  # Compute IoU

        # Convert predictions to NumPy
        preds_np = preds_binary.cpu().numpy()  # Move to CPU, convert to NumPy
        labels_np = labels.cpu().numpy()

        # Save each mask separately
        for i in range(preds_np.shape[0]):  # Batch dimension
            filename = os.path.join(self.output_dir, f"mask_{batch_idx}_{i}.tif")
            tiff.imwrite(filename, (preds_np[i, 0] * 255).astype("uint8"))  # Scale to 0-255 for saving

        # Logging
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)
        self.log('test_iou', iou)

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

        # Convert image to float32 to ensure proper normalization
        image = image.astype('float32')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Ensure that the mask is binary and of type float32
        mask = mask.float()

        return image, mask

    def load_image(self, filename):
        # Load the image using tiff.imread
        img = tiff.imread(filename)
        return img

    def resize(self, image):
        # Resize to the desired size (320x320)
        return Image.fromarray(image).resize((320, 320))

def main():

    # print("enter a name for this run: ")
    # run_name = input()

    # Prepare dataset splits
    filenames = prepare_filenames()
    train_data, train_masks, val_data, val_masks, test_data, test_masks = filenames

    # Create transform to normalize and resize images
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Create DataLoader for training, validation, and testing
    train_dataset = KelpDataset(train_data, train_masks, transform)
    val_dataset = KelpDataset(val_data, val_masks, transform)
    test_dataset = KelpDataset(test_data, test_masks, transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=4)

    # Initialize the UNet model
    model = UNet()

    # Initialize PyTorch Lightning Trainer
    # trainer = pl.Trainer(max_epochs=10)  # Set gpus to 1 if you have GPU, else remove it
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=10
        # limit_train_batches=100,
        # limit_val_batches=10,
    )


    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
