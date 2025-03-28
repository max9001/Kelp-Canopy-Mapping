import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm
import heapq  # Import for efficient top-k selection

def calculate_kelp_pixel_counts(directory):
    """
    Calculates kelp pixel counts and returns a list of (count, filename) tuples.
    """
    kelp_counts = []
    

    filenames = []
    filenames = [f for f in directory.iterdir() if f.is_file() and f.name.startswith('mask_') and f.name.endswith('.tif')] 
    # filenames = [f for f in directory.iterdir() if f.is_file() and f.name.endswith('_kelp.tif')]
    for f in tqdm(directory.iterdir(), desc="Scanning directory for kelp files"):
        # Check if the item is a file and ends with the desired suffix
        if f.is_file() and f.name.endswith('_kelp.tif'):
            filenames.append(f) # Append the Path object to the list

    for filename in tqdm(filenames, desc="Processing Images"):
        try:
            image_GT = Image.open(filename)
            image_GT = np.array(image_GT)
            kelp_count = np.sum(image_GT == 1)
            kelp_counts.append((kelp_count, filename.name))
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return kelp_counts

def plot_histogram_with_outlier_handling(kelp_counts_with_filenames, bins=50, output_file="example_filenames.txt",
                                        outlier_threshold_lower=0.01, outlier_threshold_upper=0.99):
    """
    Plots a histogram, handles outliers, shows zero counts, and writes example filenames.

    Args:
        kelp_counts_with_filenames: List of (count, filename) tuples.
        bins: Number of bins for the histogram.
        output_file: Path to the output text file.
        outlier_threshold_lower: Lower percentile for outlier removal (e.g., 0.01 for 1st percentile).
        outlier_threshold_upper: Upper percentile for outlier removal (e.g., 0.99 for 99th percentile).
    """

    # Separate zero and non-zero counts, keeping filenames
    zero_counts = [(count, filename) for count, filename in kelp_counts_with_filenames if count == 0]
    non_zero_counts = [(count, filename) for count, filename in kelp_counts_with_filenames if count > 0]
    zero_count_num = len(zero_counts)

    # --- Outlier Handling ---
    if non_zero_counts:
        non_zero_values = [count for count, _ in non_zero_counts]
        lower_bound = np.percentile(non_zero_values, outlier_threshold_lower * 100)
        upper_bound = np.percentile(non_zero_values, outlier_threshold_upper * 100)

        # Filter out outliers, keeping filenames
        filtered_non_zero_counts = [(count, filename) for count, filename in non_zero_counts
                                     if lower_bound <= count <= upper_bound]
    else:
        filtered_non_zero_counts = [] #if there are not any nonzero, create empty list.
        lower_bound = 0 #placeholder
        upper_bound = 0 #placeholder

    # --- Histogram Plotting ---
    plt.figure(figsize=(12, 6))
    plt.bar(0, zero_count_num, color='red', label='Zero Kelp Pixels', width=50)  # Adjust width as needed


    if filtered_non_zero_counts:
          filtered_non_zero_values = [count for count, _ in filtered_non_zero_counts]
          counts, edges, _ = plt.hist(filtered_non_zero_values, bins=bins, color='skyblue',
                                    edgecolor='black', label='Non-Zero Kelp Pixels',
                                    range=(1, max(filtered_non_zero_values)))
          plt.xlim(-200, max(filtered_non_zero_values) + 200)

    else: # keep behavior for only 0s
        counts, edges = np.histogram([], bins = bins, range = (1,1))
        plt.xlim(-200,1000)


    plt.title('Distribution of Kelp Pixel Counts (with Outlier Handling and Explicit Zero Count)', fontsize=16)
    plt.xlabel('Number of Kelp Pixels', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    plt.show()

    # --- Write Example Filenames ---
    with open(output_file, "w") as f:
        f.write("Example Filenames for Each Bin (Outliers Removed):\n\n")
        f.write(f"Outlier Thresholds: Lower={lower_bound:.2f}, Upper={upper_bound:.2f}\n\n")

        # Zero bin
        if zero_counts:
            example_zero_filename = zero_counts[0][1]
            f.write(f"Bin [0]: {example_zero_filename}\n")

        # Non-zero bins
        for i in range(len(counts)):
            bin_lower = edges[i]
            bin_upper = edges[i + 1]
            examples_in_bin = [filename for count, filename in filtered_non_zero_counts
                               if bin_lower <= count < bin_upper] # Use filtered data

            if examples_in_bin:
                example_filename = examples_in_bin[0]
                f.write(f"Bin [{int(bin_lower)}-{int(bin_upper)}): {example_filename}\n")
            else:
                f.write(f"Bin [{int(bin_lower)}-{int(bin_upper)}): No example found\n")
    # --- End File Write ---

def get_top_k_counts(kelp_counts_with_filenames, k=1000):
    """
    Efficiently gets the top k (count, filename) pairs using a heap.

    Args:
        kelp_counts_with_filenames: List of (count, filename) tuples.
        k: The number of top elements to retrieve.

    Returns:
        dict: A dictionary where keys are filenames and values are pixel counts,
              containing the top k entries.  Returns an empty dictionary if
              k is greater than the number of available items, or if the input list
              is empty.
    """
    if not kelp_counts_with_filenames or k <= 0:  # Handle empty input or invalid k
        return {}
    if k > len(kelp_counts_with_filenames):
        k = len(kelp_counts_with_filenames) #take all the counts

    # Use heapq.nlargest to get the top k elements efficiently.
    # We directly use the (count, filename) tuples; heapq will use the count for comparison.
    top_k_list = heapq.nlargest(k, kelp_counts_with_filenames, key=lambda x: x[0])

    # Convert the list of (count, filename) tuples to a dictionary {filename: count}
    top_k_dict = {filename: count for count, filename in top_k_list}
    return top_k_dict

def main(option):
    # directory = Path().resolve().parent / "data" / "train_kelp"
    directory = Path().resolve().parent / "output" / "predictions"
    print("started")
    # directory = Path().resolve().parent / "data" / "tiled_kelp"
    if not directory.exists():
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    kelp_counts_with_filenames = calculate_kelp_pixel_counts(directory)
    # top_1000_counts = get_top_k_counts(kelp_counts_with_filenames, k=1000)

    # print(f"Total number of images processed: {len(list(top_1000_counts.values()))}")
    # print(f"Number of images with zero kelp pixels: {list(top_1000_counts.values()).count(0)}")
    # print(f"Maximum kelp pixel count: {max(list(top_1000_counts.values()))}")
    # print(f"Minimum kelp pixel count: {min(list(top_1000_counts.values()))}")
    # print(f"Average kelp pixel count: {np.mean(list(top_1000_counts.values())):.2f}")
    # print(f"Median kelp pixel count: {np.median(list(top_1000_counts.values())):.2f}")
    # exit()
    # Adjust outlier thresholds as needed
    # plot_histogram_with_outlier_handling(kelp_counts_with_filenames, bins=20,
    #                                      outlier_threshold_lower=0.01, outlier_threshold_upper=0.99)

    # --- Calculate and print statistics (no changes here) ---
    kelp_counts = [count for count, _ in kelp_counts_with_filenames] # Extract just the counts
    print(f"Total number of images processed: {len(kelp_counts)}")
    print(f"Number of images with zero kelp pixels: {kelp_counts.count(0)}")
    print(f"Maximum kelp pixel count: {max(kelp_counts)}")
    print(f"Minimum kelp pixel count: {min(kelp_counts)}")
    print(f"Average kelp pixel count: {np.mean(kelp_counts):.2f}")
    print(f"Median kelp pixel count: {np.median(kelp_counts):.2f}")
    # --- End of statistics ---



if __name__ == "__main__":
    main()