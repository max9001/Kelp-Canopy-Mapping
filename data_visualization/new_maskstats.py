import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm
import heapq
from itertools import islice # Import islice to limit the iterator

def calculate_kelp_pixel_counts(directory, limit=100000): # Add limit parameter
    """
    Calculates kelp pixel counts for a limited number of files,
    returning a list of (count, filename) tuples.
    """
    kelp_counts = []

    print(f"Scanning directory for the first {limit} kelp files...")
    # Use glob to create an iterator for files matching the tiled pattern
    file_iterator = directory.glob('*_kelp.tif')

    # Use islice to get only the first 'limit' items from the iterator
    # This avoids loading all filenames into memory if there are millions
    limited_filenames_iterator = islice(file_iterator, limit)

    # Convert the limited iterator to a list to get the total count for tqdm
    # For 100k filenames, this memory usage should be acceptable.
    filenames_to_process = list(limited_filenames_iterator)
    print(f"Found {len(filenames_to_process)} files to process (up to limit of {limit}).")


    if not filenames_to_process:
        print("No matching kelp files found in the specified directory.")
        return kelp_counts # Return empty list

    # Process the limited list of filenames
    for filename_path in tqdm(filenames_to_process, desc="Processing Images"):
        try:
            # filename_path is already a Path object from glob
            image_GT = Image.open(filename_path)
            image_GT = np.array(image_GT)
            kelp_count = np.sum(image_GT == 1)
            kelp_counts.append((kelp_count, filename_path.name)) # Store only the name
        except Exception as e:
            print(f"Error processing {filename_path.name}: {e}") # Log error with filename

    return kelp_counts

# --- plot_histogram_with_outlier_handling remains the same ---
# It operates on the list returned by calculate_kelp_pixel_counts
# and doesn't need to know about the file naming convention itself.
def plot_histogram_with_outlier_handling(kelp_counts_with_filenames, bins=50, output_file="example_filenames.txt",
                                        outlier_threshold_lower=0.01, outlier_threshold_upper=0.99):
    """
    Plots a histogram, handles outliers, shows zero counts, and writes example filenames.
    (No changes needed in this function's logic)
    """
    # Separate zero and non-zero counts, keeping filenames
    zero_counts = [(count, filename) for count, filename in kelp_counts_with_filenames if count == 0]
    non_zero_counts = [(count, filename) for count, filename in kelp_counts_with_filenames if count > 0]
    zero_count_num = len(zero_counts)

    # --- Outlier Handling ---
    if non_zero_counts:
        non_zero_values = [count for count, _ in non_zero_counts]
        # Handle edge case where all non-zero values might be the same
        if len(set(non_zero_values)) > 1:
             lower_bound = np.percentile(non_zero_values, outlier_threshold_lower * 100)
             upper_bound = np.percentile(non_zero_values, outlier_threshold_upper * 100)
        else: # If all values are the same, don't filter
             lower_bound = min(non_zero_values)
             upper_bound = max(non_zero_values)


        # Filter out outliers, keeping filenames
        filtered_non_zero_counts = [(count, filename) for count, filename in non_zero_counts
                                     if lower_bound <= count <= upper_bound]
    else:
        filtered_non_zero_counts = [] #if there are not any nonzero, create empty list.
        lower_bound = 0 #placeholder
        upper_bound = 0 #placeholder

    # --- Histogram Plotting ---
    plt.figure(figsize=(12, 6))
    # Use a smaller width if there are many bins or counts are low
    bar_width = 0.8 if not filtered_non_zero_counts else max(1, int(max(c for c,_ in filtered_non_zero_counts)/bins * 0.5)) if filtered_non_zero_counts else 1
    plt.bar(0, zero_count_num, color='red', label='Zero Kelp Pixels', width=bar_width)


    if filtered_non_zero_counts:
          filtered_non_zero_values = [count for count, _ in filtered_non_zero_counts]
          # Ensure range starts at 1 if there are non-zero values
          hist_range = (1, max(filtered_non_zero_values)) if filtered_non_zero_values else (1, 1)
          counts, edges, _ = plt.hist(filtered_non_zero_values, bins=bins, color='skyblue',
                                    edgecolor='black', label='Non-Zero Kelp Pixels',
                                    range=hist_range)
          # Adjust xlim based on actual data range
          plt.xlim(-bar_width, hist_range[1] + bar_width)


    else: # keep behavior for only 0s
        counts, edges = np.histogram([], bins = bins, range = (1,1))
        plt.xlim(-bar_width, bar_width) # Adjust xlim for only zero bar


    plt.title('Distribution of Kelp Pixel Counts (First 100k Files)', fontsize=16) # Updated title
    plt.xlabel('Number of Kelp Pixels', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    plt.tight_layout() # Add tight layout
    plt.show()

    # --- Write Example Filenames ---
    # (Logic remains the same, operates on the filtered list)
    with open(output_file, "w") as f:
        f.write("Example Filenames for Each Bin (Outliers Removed):\n\n")
        f.write(f"Outlier Thresholds: Lower={lower_bound:.2f}, Upper={upper_bound:.2f}\n\n")

        if zero_counts:
            # Handle case where zero_counts might be empty if limit is small
            example_zero_filename = zero_counts[0][1] if zero_counts else "N/A"
            f.write(f"Bin [0]: {example_zero_filename}\n")
        else:
             f.write(f"Bin [0]: No zero-kelp images found in sample\n")


        # Check if counts were generated (i.e., if there were non-zero values)
        if 'counts' in locals() and len(counts) > 0:
            for i in range(len(counts)):
                bin_lower = edges[i]
                bin_upper = edges[i + 1]
                # Find examples within the current bin using the filtered list
                examples_in_bin = [filename for count, filename in filtered_non_zero_counts
                                if bin_lower <= count < bin_upper]

                if examples_in_bin:
                    example_filename = examples_in_bin[0]
                    f.write(f"Bin [{int(bin_lower)}-{int(bin_upper)}): {example_filename}\n")
                else:
                    # It's possible a bin is empty after outlier removal
                    f.write(f"Bin [{int(bin_lower)}-{int(bin_upper)}): No example found (or removed as outlier)\n")
        else:
             f.write("No non-zero bins to report.\n")

# --- get_top_k_counts remains the same ---
# It also operates on the list returned by calculate_kelp_pixel_counts
def get_top_k_counts(kelp_counts_with_filenames, k=1000):
    """
    Efficiently gets the top k (count, filename) pairs using a heap.
    (No changes needed in this function's logic)
    """
    if not kelp_counts_with_filenames or k <= 0:
        return {}
    # Adjust k if it's larger than the number of items we actually processed
    actual_k = min(k, len(kelp_counts_with_filenames))
    if actual_k == 0: return {}


    top_k_list = heapq.nlargest(actual_k, kelp_counts_with_filenames, key=lambda x: x[0])
    top_k_dict = {filename: count for count, filename in top_k_list}
    return top_k_dict

def main():
    print("started")
    # --- IMPORTANT: Set directory to the TILED kelp masks ---
    directory = Path().resolve().parent / "data" / "balanced_tiled_40_60" / "train_kelp"
    if not directory.exists():
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    # calculate_kelp_pixel_counts now handles the limit internally
    kelp_counts_with_filenames = calculate_kelp_pixel_counts(directory, limit=100000)

    # Check if any data was returned before proceeding
    if not kelp_counts_with_filenames:
        print("No kelp files processed. Cannot generate plot or statistics.")
        return

    # --- Plotting ---
    # You might want fewer bins if analyzing only 100k files
    plot_histogram_with_outlier_handling(kelp_counts_with_filenames, bins=20,
                                         outlier_threshold_lower=0.01, outlier_threshold_upper=0.99)

    # --- Statistics ---
    kelp_counts = [count for count, _ in kelp_counts_with_filenames]
    print(f"--- Statistics based on {len(kelp_counts)} processed files ---")
    print(f"Number of images with zero kelp pixels: {kelp_counts.count(0)}")
    # Handle case where there might be only zero-count images
    if any(c > 0 for c in kelp_counts):
        print(f"Maximum kelp pixel count: {max(kelp_counts)}")
        print(f"Minimum kelp pixel count (among non-zero): {min(c for c in kelp_counts if c > 0)}")
    else:
        print("Maximum kelp pixel count: 0")
        print("Minimum kelp pixel count (among non-zero): N/A")
    print(f"Average kelp pixel count: {np.mean(kelp_counts):.2f}")
    print(f"Median kelp pixel count: {np.median(kelp_counts):.2f}")
    # --- End of statistics ---

if __name__ == "__main__":
    main()