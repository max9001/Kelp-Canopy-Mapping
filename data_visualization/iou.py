import numpy as np
import tifffile as tiff
from pathlib import Path
import os
from tqdm import tqdm
import sys
from pyprojroot import here # Optional, if you use it for root finding

def calculate_iou(prediction_mask, ground_truth_mask):
    """
    Calculates the Intersection over Union (IoU) score for binary masks.

    Args:
        prediction_mask (np.ndarray): The predicted binary mask (0 or 1).
        ground_truth_mask (np.ndarray): The ground truth binary mask (0 or 1).

    Returns:
        float: The IoU score.
    """
    # Ensure masks are boolean for logical operations
    pred_bool = prediction_mask.astype(bool)
    gt_bool = ground_truth_mask.astype(bool)

    intersection = np.sum(pred_bool & gt_bool)
    union = np.sum(pred_bool | gt_bool)

    if union == 0:
        # If both masks are empty, IoU is 1 (perfect agreement)
        # If only prediction is empty but GT isn't (or vice versa), union > 0, handled below
        return 1.0 if intersection == 0 else 0.0
    else:
        iou = intersection / union
        return iou

def main():
    """
    Calculates and prints the average IoU score between predicted and ground truth masks.
    """
    try:
        # --- Define Directories ---
        # Use pyprojroot if you have it set up, otherwise define paths directly
        # root_dir = here()
        root_dir = Path().resolve().parent # Assumes script is in a subdirectory of the project root

        # Directory containing the predicted masks (output from your model)
        # Make sure the filenames match the pattern used in your test_step
        prediction_dir = root_dir / "output" / "predictions_resnet_dynamic" # Or your specific output folder

        # Directory containing the original ground truth masks
        # This should correspond to the test set masks used during evaluation
        # Adjust the path based on which dataset you tested on (e.g., original, tiled, balanced)
        # ground_truth_base_dir = root_dir / "data" / "balanced_tiled_40_60" # Example: change if needed
        ground_truth_dir = root_dir / "data" / "train_kelp" 
        if not prediction_dir.exists():
            print(f"Error: Prediction directory not found: {prediction_dir}")
            return
        if not ground_truth_dir.exists():
            print(f"Error: Ground truth directory not found: {ground_truth_dir}")
            return

        # --- Find Prediction Files ---
        # Adjust the glob pattern if your prediction filenames are different
        prediction_files = list(prediction_dir.glob("prediction_*.tif"))

        if not prediction_files:
            print(f"Error: No prediction files found in {prediction_dir} matching 'prediction_*.tif'")
            return

        iou_scores = []
        print(f"Found {len(prediction_files)} prediction files. Calculating IoU...")

        # --- Iterate and Calculate IoU ---
        for pred_path in tqdm(prediction_files, desc="Calculating IoU"):
            try:
                # Extract original base filename from prediction filename
                # Assumes format "prediction_ORIGINALFILENAME.tif"
                base_filename = pred_path.stem.replace("prediction_", "")
                gt_filename = f"{base_filename}_kelp.tif"
                gt_path = ground_truth_dir / gt_filename

                if not gt_path.exists():
                    print(f"Warning: Ground truth file not found for prediction {pred_path.name}. Skipping.")
                    continue

                # Load masks
                pred_mask = tiff.imread(pred_path)
                gt_mask = tiff.imread(gt_path)

                # Ensure masks are binary (0 or 1) - predictions might be 0/255 if saved that way
                if pred_mask.max() > 1:
                    pred_mask = (pred_mask > 127).astype(np.uint8) # Threshold if needed
                else:
                    pred_mask = pred_mask.astype(np.uint8)

                gt_mask = gt_mask.astype(np.uint8) # Ensure GT is also uint8

                # Check if shapes match (optional but good practice)
                if pred_mask.shape != gt_mask.shape:
                    print(f"Warning: Shape mismatch for {pred_path.name} ({pred_mask.shape}) and {gt_path.name} ({gt_mask.shape}). Skipping.")
                    continue

                # Calculate IoU
                iou = calculate_iou(pred_mask, gt_mask)
                iou_scores.append(iou)

            except Exception as e:
                print(f"Error processing file pair for {pred_path.name}: {e}")

        # --- Calculate and Print Average IoU ---
        if not iou_scores:
            print("Error: No IoU scores were calculated. Check file paths and formats.")
        else:
            average_iou = np.mean(iou_scores)
            print(f"\nAverage IoU Score: {average_iou:.8f}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()