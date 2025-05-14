# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# from pathlib import Path
# import os
# from tqdm import tqdm
# import heapq  # Import for efficient top-k selection
# import sys
# import random

# def calculate_kelp_pixel_counts(directory, option):
#     """
#     Calculates kelp pixel counts and returns a list of (count, filename) tuples.
#     """
#     kelp_counts = []
#     filenames = []

#     # --------- option --------
#     if option == "original":
#         filenames = [f for f in directory.iterdir() if f.is_file() and f.name.endswith('_kelp.tif')]

#     if option == "output":
#         filenames = [f for f in directory.iterdir() if f.is_file() and f.name.startswith('prediction_') and f.name.endswith('.tif')] 

#     if option == "tile" or option == "tile_balanced":
#         # filenames = []
#         # count = 0
#         # limit = 100000
#         # for f in directory.iterdir():
#         #     if f.is_file() and f.name.endswith('_kelp.tif'):
#         #         filenames.append(f)
#         #         count += 1
#         #         if count >= limit:
#         #             break
#         for f in tqdm(directory.iterdir(), desc="Scanning directory for kelp files"):
#             # Check if the item is a file and ends with the desired suffix
#             if f.is_file() and f.name.endswith('_kelp.tif'):
#                 filenames.append(f) # Append the Path object to the list
#         filenames = random.sample(filenames, 100000)

#     for filename in tqdm(filenames, desc="Processing Images"):
#         try:
#             image_GT = Image.open(filename)
#             image_GT = np.array(image_GT)
#             kelp_count = np.sum(image_GT == 1)
#             kelp_counts.append((kelp_count, filename.name))
#         except Exception as e:
#             print(f"Error processing {filename}: {e}")

#     return kelp_counts

# def plot_histogram_with_outlier_handling(kelp_counts_with_filenames, bins=50, output_file="example_filenames.txt",
#                                         outlier_threshold_lower=0.01, outlier_threshold_upper=0.99):
#     """
#     Plots a histogram, handles outliers, shows zero counts, and writes example filenames.

#     Args:
#         kelp_counts_with_filenames: List of (count, filename) tuples.
#         bins: Number of bins for the histogram.
#         output_file: Path to the output text file.
#         outlier_threshold_lower: Lower percentile for outlier removal (e.g., 0.01 for 1st percentile).
#         outlier_threshold_upper: Upper percentile for outlier removal (e.g., 0.99 for 99th percentile).
#     """

#     # Separate zero and non-zero counts, keeping filenames
#     zero_counts = [(count, filename) for count, filename in kelp_counts_with_filenames if count == 0]
#     non_zero_counts = [(count, filename) for count, filename in kelp_counts_with_filenames if count > 0]
#     zero_count_num = len(zero_counts)

#     # --- Outlier Handling ---
#     if non_zero_counts:
#         non_zero_values = [count for count, _ in non_zero_counts]
#         lower_bound = np.percentile(non_zero_values, outlier_threshold_lower * 100)
#         upper_bound = np.percentile(non_zero_values, outlier_threshold_upper * 100)

#         # Filter out outliers, keeping filenames
#         filtered_non_zero_counts = [(count, filename) for count, filename in non_zero_counts
#                                      if lower_bound <= count <= upper_bound]
#     else:
#         filtered_non_zero_counts = [] #if there are not any nonzero, create empty list.
#         lower_bound = 0 #placeholder
#         upper_bound = 0 #placeholder

#     # --- Histogram Plotting ---
#     plt.figure(figsize=(12, 6))
#     plt.bar(0, zero_count_num, color='red', label='Zero Kelp Pixels', width=50)  # Adjust width as needed


#     if filtered_non_zero_counts:
#           filtered_non_zero_values = [count for count, _ in filtered_non_zero_counts]
#           counts, edges, _ = plt.hist(filtered_non_zero_values, bins=bins, color='skyblue',
#                                     edgecolor='black', label='Non-Zero Kelp Pixels',
#                                     range=(1, max(filtered_non_zero_values)))
#           plt.xlim(-200, max(filtered_non_zero_values) + 200)

#     else: # keep behavior for only 0s
#         counts, edges = np.histogram([], bins = bins, range = (1,1))
#         plt.xlim(-200,1000)


#     plt.title('Distribution of Kelp Pixel Counts (with Outlier Handling and Explicit Zero Count)', fontsize=16)
#     plt.xlabel('Number of Kelp Pixels', fontsize=14)
#     plt.ylabel('Number of Images', fontsize=14)
#     plt.grid(axis='y', alpha=0.75)
#     plt.legend()
#     plt.show()

#     # --- Write Example Filenames ---
#     with open(output_file, "w") as f:
#         f.write("Example Filenames for Each Bin (Outliers Removed):\n\n")
#         f.write(f"Outlier Thresholds: Lower={lower_bound:.2f}, Upper={upper_bound:.2f}\n\n")

#         # Zero bin
#         if zero_counts:
#             example_zero_filename = zero_counts[0][1]
#             f.write(f"Bin [0]: {example_zero_filename}\n")

#         # Non-zero bins
#         for i in range(len(counts)):
#             bin_lower = edges[i]
#             bin_upper = edges[i + 1]
#             examples_in_bin = [filename for count, filename in filtered_non_zero_counts
#                                if bin_lower <= count < bin_upper] # Use filtered data

#             if examples_in_bin:
#                 example_filename = examples_in_bin[0]
#                 f.write(f"Bin [{int(bin_lower)}-{int(bin_upper)}): {example_filename}\n")
#             else:
#                 f.write(f"Bin [{int(bin_lower)}-{int(bin_upper)}): No example found\n")
#     # --- End File Write ---

# def get_top_k_counts(kelp_counts_with_filenames, k=1000):
#     """
#     Efficiently gets the top k (count, filename) pairs using a heap.

#     Args:
#         kelp_counts_with_filenames: List of (count, filename) tuples.
#         k: The number of top elements to retrieve.

#     Returns:
#         dict: A dictionary where keys are filenames and values are pixel counts,
#               containing the top k entries.  Returns an empty dictionary if
#               k is greater than the number of available items, or if the input list
#               is empty.
#     """
#     if not kelp_counts_with_filenames or k <= 0:  # Handle empty input or invalid k
#         return {}
#     if k > len(kelp_counts_with_filenames):
#         k = len(kelp_counts_with_filenames) #take all the counts

#     # Use heapq.nlargest to get the top k elements efficiently.
#     # We directly use the (count, filename) tuples; heapq will use the count for comparison.
#     top_k_list = heapq.nlargest(k, kelp_counts_with_filenames, key=lambda x: x[0])

#     # Convert the list of (count, filename) tuples to a dictionary {filename: count}
#     top_k_dict = {filename: count for count, filename in top_k_list}
#     return top_k_dict






# #---------------------------------------------------------------------------------

# def main():
#     option = sys.argv[1]
    
#     if option == "original":
#         directory = Path().resolve().parent / "data" / "train_kelp"

#     if option == "output":
#         directory = Path().resolve().parent / "output" / "predictions_resnet_test"
        
#     if option == "tile":
#         directory = Path().resolve().parent / "data" / "tiled_kelp"

#     if option == "tile_balanced":
#         directory = Path().resolve().parent / "data" / "balanced_tiled_40_60" / "train_kelp"

    

 
#     if not directory.exists():
#         raise FileNotFoundError(f"The directory {directory} does not exist.")

#     kelp_counts_with_filenames = calculate_kelp_pixel_counts(directory, option)

    
#     # --- Calculate and print statistics (no changes here) ---
#     kelp_counts = [count for count, _ in kelp_counts_with_filenames] # Extract just the counts
#     print(f"Total number of images processed: {len(kelp_counts)}")
#     print(f"Number of images with zero kelp pixels: {kelp_counts.count(0)}")
#     print(f"Maximum kelp pixel count: {max(kelp_counts)}")
#     print(f"Minimum kelp pixel count: {min(kelp_counts)}")
#     print(f"Average kelp pixel count: {np.mean(kelp_counts):.2f}")
#     print(f"Median kelp pixel count: {np.median(kelp_counts):.2f}")
#     # --- End of statistics ---

#     print("\n--- Top 5 Files with Most Kelp Pixels ---")
#     # Sort the list by count (descending) and take the top 5
#     top_5 = sorted(kelp_counts_with_filenames, key=lambda item: item[0], reverse=True)[:5]

#     for i, (count, filename) in enumerate(top_5):
#         print(f"{i+1}. {filename} (Count: {count})")
#     # --- End Top 5 ---


#     # Adjust outlier thresholds as needed
#     plot_histogram_with_outlier_handling(kelp_counts_with_filenames, bins=20,
#                                          outlier_threshold_lower=0.01, outlier_threshold_upper=0.99)


# if __name__ == "__main__":
#     main()


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import heapq  # For efficient top-k selection
import random # For selecting example filenames

# --- Configuration Constants ---
ROOT_DIR = Path().resolve().parent  # Defines the project root directory
BASE_DATA_DIR = ROOT_DIR / "data" / "original"

# Histogram and Plotting Parameters
HISTOGRAM_BINS = 20  # Default number of bins for the histogram's non-zero part
OUTLIER_LOWER_PERCENTILE = 0  # Lower percentile (0.0-1.0) for outlier removal
OUTLIER_UPPER_PERCENTILE = 0  # Upper percentile (0.0-1.0) for outlier removal
# Name of the output file listing example filenames for each histogram bin
EXAMPLE_FILENAMES_OUTPUT_PATH = "original_kelp_histogram_examples.txt"

# Number of top images (by kelp count) to list in the console output
NUM_TOP_KELP_IMAGES_TO_DISPLAY = 5
# --- End Configuration Constants ---


def calculate_original_kelp_pixel_counts(directory: Path):
    """
    Scans a directory for original kelp mask files (ending with "_kelp.tif"),
    calculates the number of kelp pixels (pixels with value 1) in each, and
    returns a list of (count, filename) tuples.

    Args:
        directory (Path): The directory to scan for original kelp mask files.

    Returns:
        list: A list of (kelp_pixel_count, filename_str) tuples.
              `kelp_pixel_count` is the integer count of pixels with value 1.
              `filename_str` is the string name of the processed file.
              Returns an empty list if no suitable files are found or errors occur.
    """
    kelp_counts = []
    filenames_to_process = [
        f for f in directory.iterdir()
        if f.is_file() and f.name.endswith("_kelp.tif")
    ]

    if not filenames_to_process:
        print(f"No files ending with '{"_kelp.tif"}' found in directory: {directory}")
        return []

    print(f"Processing {len(filenames_to_process)} original kelp mask files...")
    for file_path in tqdm(filenames_to_process, desc="Processing Original Kelp Images"):
        try:
            image_GT = Image.open(file_path)
            image_GT_array = np.array(image_GT)
            kelp_count = np.sum(image_GT_array == 1)  # Assuming kelp pixels are value 1
            kelp_counts.append((kelp_count, file_path.name))
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    return kelp_counts

def plot_histogram_with_outlier_handling(
    kelp_counts_with_filenames: list,
    bins: int = 50,
    output_file: Path = Path("example_filenames.txt"), # Default, overridden by global
    outlier_threshold_lower: float = 0.01,
    outlier_threshold_upper: float = 0.99
):
    """
    Generates and displays a histogram of kelp pixel counts, with special handling
    for zero counts and outliers in non-zero counts. Writes example filenames for
    each bin to a text file.

    Args:
        kelp_counts_with_filenames (list): A list of (count, filename_str) tuples.
        bins (int): Number of bins for the non-zero part of the histogram.
        output_file (Path): Path to save the text file with example filenames per bin.
        outlier_threshold_lower (float): Lower percentile (0.0 to 1.0) for identifying
                                         outliers in non-zero counts.
        outlier_threshold_upper (float): Upper percentile (0.0 to 1.0) for identifying
                                         outliers.
    """
    if not kelp_counts_with_filenames:
        print("No data to plot for histogram.")
        return

    zero_counts_info = [(count, filename) for count, filename in kelp_counts_with_filenames if count == 0]
    non_zero_counts_info = [(count, filename) for count, filename in kelp_counts_with_filenames if count > 0]
    num_zero_count_images = len(zero_counts_info)

    filtered_non_zero_counts_info = non_zero_counts_info
    lower_bound, upper_bound = None, None

    if non_zero_counts_info:
        non_zero_values = [count for count, _ in non_zero_counts_info]
        lower_bound = np.percentile(non_zero_values, outlier_threshold_lower * 100)
        upper_bound = np.percentile(non_zero_values, outlier_threshold_upper * 100)

        if lower_bound >= upper_bound and len(set(non_zero_values)) > 1:
             print(f"Warning: Outlier bounds are problematic (lower={lower_bound}, upper={upper_bound}). Using all non-zero data for histogram.")
        else:
            filtered_non_zero_counts_info = [
                (count, filename) for count, filename in non_zero_counts_info
                if lower_bound <= count <= upper_bound
            ]
            num_outliers = len(non_zero_counts_info) - len(filtered_non_zero_counts_info)
            if num_outliers > 0:
                print(f"Identified and excluded {num_outliers} non-zero outliers for histogram (bounds: {lower_bound:.2f}-{upper_bound:.2f}).")

    plt.figure(figsize=(12, 7))
    zero_bar_width = 50
    if filtered_non_zero_counts_info:
        max_val = max(count for count, _ in filtered_non_zero_counts_info) if filtered_non_zero_counts_info else 1
        zero_bar_width = max(1, (max_val / bins) * 0.8)

    plt.bar(0, num_zero_count_images, color='red', label=f'Zero Kelp Pixels ({num_zero_count_images} images)', width=zero_bar_width)

    hist_counts, hist_edges = [], []

    if filtered_non_zero_counts_info:
        filtered_values = [count for count, _ in filtered_non_zero_counts_info]
        # Ensure range starts at 1 for non-zero counts, or a sensible default if no data
        hist_range_min = 1
        hist_range_max = max(filtered_values) if filtered_values else hist_range_min

        hist_counts, hist_edges, _ = plt.hist(
            filtered_values, bins=bins, color='skyblue', edgecolor='black',
            label=f'Non-Zero Kelp Pixels ({len(filtered_values)} images, outliers removed)',
            range=(hist_range_min, hist_range_max)
        )
        plot_upper_xlim = hist_range_max + zero_bar_width
        plt.xlim(-zero_bar_width * 1.5, plot_upper_xlim)
    else:
        plt.xlim(-zero_bar_width * 1.5, 1000)

    plt.title('Distribution of Kelp Pixel Counts (Original Masks)', fontsize=16)
    plt.xlabel('Number of Kelp Pixels per Image', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    plt.tight_layout()
    plt.show()

    with open(output_file, "w") as f:
        f.write("Example Filenames for Each Bin (Non-Zero Counts are Outlier-Filtered for Histogram Bins):\n")
        if lower_bound is not None and upper_bound is not None:
             f.write(f"Non-Zero Outlier Filter Percentiles: Lower={outlier_threshold_lower*100:.1f}th ({lower_bound:.2f} pixels), Upper={outlier_threshold_upper*100:.1f}th ({upper_bound:.2f} pixels)\n\n")
        else:
            f.write("No outlier filtering applied to non-zero counts or no non-zero counts present.\n\n")

        if zero_counts_info:
            example_zero_filename = random.choice(zero_counts_info)[1]
            f.write(f"Bin [0 Pixels] (Total: {num_zero_count_images} images): Example: {example_zero_filename}\n")
        else:
            f.write("Bin [0 Pixels]: No images with zero kelp pixels.\n")

        if filtered_non_zero_counts_info and len(hist_counts) > 0:
            f.write("\nNon-Zero Bins (from histogram):\n")
            for i in range(len(hist_counts)):
                bin_lower_edge = hist_edges[i]
                bin_upper_edge = hist_edges[i+1]
                examples_in_bin = [
                    filename for count, filename in filtered_non_zero_counts_info
                    if bin_lower_edge <= count < bin_upper_edge
                ]
                if i == len(hist_counts) - 1: # Include exact upper edge for the last bin
                     examples_in_bin.extend([
                        filename for count, filename in filtered_non_zero_counts_info
                        if count == bin_upper_edge
                    ])
                     examples_in_bin = list(set(examples_in_bin)) # Remove duplicates

                if examples_in_bin:
                    example_filename = random.choice(examples_in_bin)
                    f.write(f"Bin [{int(bin_lower_edge)}-{int(bin_upper_edge)}): {int(hist_counts[i])} images. Example: {example_filename}\n")
                else:
                    f.write(f"Bin [{int(bin_lower_edge)}-{int(bin_upper_edge)}): {int(hist_counts[i])} images. No example found in this range.\n")
        elif non_zero_counts_info:
             f.write("\nAll non-zero counts were filtered as outliers or data is sparse.\n")
        else:
            f.write("\nNo non-zero kelp pixel counts found.\n")
    print(f"Example filenames per bin written to: {output_file}")


def get_top_k_counts(kelp_counts_with_filenames: list, k: int = 10):
    """
    Efficiently retrieves the top k (count, filename) pairs,
    sorted by count in descending order.

    Args:
        kelp_counts_with_filenames (list): List of (count, filename_str) tuples.
        k (int): The number of top elements to retrieve.

    Returns:
        list: A list of the top k (count, filename_str) tuples, sorted by count
              in descending order. Returns a shorter list if fewer than k items
              are available, or an empty list if input is empty or k is non-positive.
    """
    if not kelp_counts_with_filenames or k <= 0:
        return []
    return heapq.nlargest(k, kelp_counts_with_filenames, key=lambda x: x[0])


#---------------------------------------------------------------------------------

def main():
    """
    Main function to perform kelp pixel count analysis for original kelp mask files.
    It calculates kelp pixel counts from the configured subdirectory within BASE_DATA_DIR,
    prints summary statistics, lists the top N images with the most kelp,
    and generates a histogram plot of the counts.
    """
    # Directory for original kelp masks is constructed from global constants
    original_kelp_dir = BASE_DATA_DIR / "train_kelp"

    print(f"--- Kelp Pixel Analysis for Original Masks ---")
    print(f"Target Directory: {original_kelp_dir}")

    if not original_kelp_dir.exists():
        print(f"Error: The directory {original_kelp_dir} does not exist. Please check configuration and paths.")
        return

    kelp_counts_with_filenames = calculate_original_kelp_pixel_counts(original_kelp_dir)

    if not kelp_counts_with_filenames:
        print("No kelp count data was generated. Exiting.")
        return

    # --- Calculate and print statistics ---
    kelp_counts_values = [count for count, _ in kelp_counts_with_filenames]
    print(f"\n--- Statistics for {len(kelp_counts_values)} Original Kelp Images ---")
    if kelp_counts_values:
        print(f"Number of images with zero kelp pixels: {kelp_counts_values.count(0)}")
        print(f"Maximum kelp pixel count: {max(kelp_counts_values)}")
        print(f"Minimum kelp pixel count: {min(kelp_counts_values)}")
        print(f"Average kelp pixel count: {np.mean(kelp_counts_values):.2f}")
        print(f"Median kelp pixel count: {np.median(kelp_counts_values):.2f}")
    else:
        print("No counts available to calculate statistics.")
    # --- End of statistics ---

    print(f"\n--- Top {NUM_TOP_KELP_IMAGES_TO_DISPLAY} Original Files with Most Kelp Pixels ---")
    top_n_files = get_top_k_counts(kelp_counts_with_filenames, k=NUM_TOP_KELP_IMAGES_TO_DISPLAY)

    if top_n_files:
        for i, (count, filename) in enumerate(top_n_files):
            print(f"{i+1}. {filename} (Count: {count})")
    else:
        print("No files to display in top N list.")
    # --- End Top N ---

    # Plot histogram using global configuration for defaults
    plot_histogram_with_outlier_handling(
        kelp_counts_with_filenames,
        bins=HISTOGRAM_BINS,
        output_file=Path(EXAMPLE_FILENAMES_OUTPUT_PATH),
        outlier_threshold_lower=OUTLIER_LOWER_PERCENTILE,
        outlier_threshold_upper=OUTLIER_UPPER_PERCENTILE
    )


if __name__ == "__main__":
    main()