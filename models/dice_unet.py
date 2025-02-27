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

class CombinedLoss(nn.Module): # combining BCE and dice
    def __init__(self, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):
        return self.alpha * self.bce_loss(inputs, targets) + self.beta * self.dice_loss(inputs, targets)



class UNet(pl.LightningModule):
    def __init__(self, use_dice_loss=True, loss_weights=None): #added param for choice of loss function
        super(UNet, self).__init__()
        # A simple U-Net structure with 2 convolutional blocks
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=3, padding=1),  # 7 input channels
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # Output channel is 1 (binary)
        )
        # Choose loss function
        self.use_dice_loss = use_dice_loss
        if self.use_dice_loss:
            self.loss_fn = DiceLoss()
        else:
            if loss_weights is not None:
                 self.loss_fn = nn.BCELoss(weight=torch.tensor(loss_weights, dtype=torch.float32))
            else:
                self.loss_fn = nn.BCELoss() #default BCE
        # self.loss_fn = CombinedLoss() #uncomment to use Combined loss

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
        # print(f"Max label value: {torch.max(labels)}, Min label value: {torch.min(labels)}")
        if not self.use_dice_loss:  # If not using Dice, apply sigmoid
            preds = torch.sigmoid(preds)

        loss = self.loss_fn(preds, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        if not self.use_dice_loss:  # If not using Dice, apply sigmoid
           preds = torch.sigmoid(preds)
        loss = self.loss_fn(preds, labels)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        if not self.use_dice_loss:
            preds = torch.sigmoid(preds) # Sigmoid only if not dice
        loss = self.loss_fn(preds, labels)

        # Convert predictions to binary (threshold = 0.5)
        preds_binary = (preds > 0.5).float()

        acc = self.accuracy(preds_binary, labels.int())  # Binary accuracy
        iou = self.iou(preds_binary, labels.int())  # Compute IoU

        # Convert predictions and labels to NumPy
        preds_np = preds_binary.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Save each mask separately
        for i in range(preds_np.shape[0]):  # Batch dimension
            filename = os.path.join(self.output_dir, f"mask_{batch_idx}_{i}.tif")
            tiff.imwrite(filename, (preds_np[i, 0] * 255).astype("uint8"))  # Scale to 0-255

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
    filenames = prepare_filenames()
    train_data, train_masks, val_data, val_masks, test_data, test_masks = filenames

    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Create DataLoader
    train_dataset = KelpDataset(train_data, train_masks, transform)
    val_dataset = KelpDataset(val_data, val_masks, transform)
    test_dataset = KelpDataset(test_data, test_masks, transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=4)

    # --- Calculate class weights (for weighted BCE) ---
    #This is helpful for highly imbalanced datasets.
    # num_pixels = 0
    # num_kelp = 0
    # for _, mask_file in zip(train_data, train_masks):
    #     mask = tiff.imread(mask_file)
    #     num_pixels += mask.size
    #     num_kelp += np.sum(mask == 1)  # Assuming kelp is 1, no-kelp is 0
    # num_no_kelp = num_pixels - num_kelp
    # class_weights = [num_kelp / num_pixels, num_no_kelp / num_pixels] # normalize, swap to have no-kelp first
    # class_weights = [class_weights[1],class_weights[0]] #order for pytorch.
    # print(f"Class Weights for BCE: {class_weights}")

    # --- Choose Loss Function ---
    use_dice = True  # Set to True to use Dice Loss, False for BCELoss (or weighted BCELoss)
    loss_weights = class_weights if not use_dice else None # pass weights, only if using BCE.

    # Initialize the UNet model
    model = UNet(use_dice_loss=use_dice, loss_weights=loss_weights)


    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=5,
        limit_train_batches=20,  # For faster debugging, limit batches
        limit_val_batches=10,
        limit_test_batches=5, 
    )

    # Train and test the model
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()