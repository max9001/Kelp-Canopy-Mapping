import numpy as np
import tifffile as tiff
from pathlib import Path
import os
from tqdm import tqdm
import sys
from pyprojroot import here # Optional, if you use it for root finding
import torch # Import torch
from torchmetrics import JaccardIndex # Import the metric

# Remove the old NumPy-based calculate_iou function
# def calculate_iou(prediction_mask, ground_truth_mask):
#     ...

def main():
    """
    Calculates and prints the average IoU score using torchmetrics.JaccardIndex.
    """
    try:
        # --- Define Directories ---
        # root_dir = here()
        root_dir = Path().resolve().parent

        prediction_dir = root_dir / "output" / "200_resnet_test" # Or your specific output folder
        ground_truth_dir = root_dir / "data" / "balanced_tiled_40_60" / "train_kelp" # Example: change if needed

        if not prediction_dir.exists():
            print(f"Error: Prediction directory not found: {prediction_dir}")
            return
        if not ground_truth_dir.exists():
            print(f"Error: Ground truth directory not found: {ground_truth_dir}")
            return

        # --- Find Prediction Files ---
        prediction_files = list(prediction_dir.glob("prediction_*.tif"))

        if not prediction_files:
            print(f"Error: No prediction files found in {prediction_dir} matching 'prediction_*.tif'")
            return

        # --- Initialize TorchMetrics IoU ---
        # You can optionally move this to GPU if available: .to(device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        iou_metric = JaccardIndex(task="binary").to(device)

        print(f"Found {len(prediction_files)} prediction files. Calculating IoU using torchmetrics...")

        # --- Iterate and Update Metric ---
        for pred_path in tqdm(prediction_files, desc="Calculating IoU"):
            try:
                base_filename = pred_path.stem.replace("prediction_", "")
                gt_filename = f"{base_filename}_kelp.tif"
                gt_path = ground_truth_dir / gt_filename

                if not gt_path.exists():
                    print(f"Warning: Ground truth file not found for prediction {pred_path.name}. Skipping.")
                    continue

                # Load masks as NumPy arrays
                pred_mask_np = tiff.imread(pred_path)
                gt_mask_np = tiff.imread(gt_path)

                # Ensure masks are binary (0 or 1)
                if pred_mask_np.max() > 1:
                    pred_mask_np = (pred_mask_np > 127).astype(np.uint8)
                else:
                    pred_mask_np = pred_mask_np.astype(np.uint8)
                gt_mask_np = gt_mask_np.astype(np.uint8)

                # Check shapes
                if pred_mask_np.shape != gt_mask_np.shape:
                    print(f"Warning: Shape mismatch for {pred_path.name} ({pred_mask_np.shape}) and {gt_path.name} ({gt_mask_np.shape}). Skipping.")
                    continue

                # --- Convert NumPy arrays to PyTorch Tensors ---
                # Add batch and channel dimensions (N, C, H, W) -> (1, 1, H, W)
                pred_tensor = torch.from_numpy(pred_mask_np).unsqueeze(0).unsqueeze(0).to(device)
                gt_tensor = torch.from_numpy(gt_mask_np).unsqueeze(0).unsqueeze(0).int().to(device) # Target needs to be int/long

                # --- Update the metric state ---
                iou_metric.update(pred_tensor, gt_tensor)

            except Exception as e:
                print(f"Error processing file pair for {pred_path.name}: {e}")

        # --- Compute Final Average IoU ---
        try:
            # Compute the metric over all accumulated batches/samples
            average_iou = iou_metric.compute()
            print(f"\nAverage IoU Score (torchmetrics): {average_iou.item():.8f}") # Use .item() to get Python float
        except Exception as e:
             print(f"Error computing final IoU: {e}")


    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()