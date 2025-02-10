# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# from pathlib import Path
# import os
# from tqdm import tqdm  # For progress bar


# def calculate_kelp_pixel_counts(directory):
#     """
#     Calculates the number of kelp pixels (value 1) in each ground truth image.

#     Args:
#         directory (Path): The path to the directory containing the ground truth images.

#     Returns:
#         list: A list of kelp pixel counts for each image.
#     """
#     kelp_counts = []
#     # Correctly filtering _kelp.tif files and constructing full paths
#     filenames = [f for f in directory.iterdir() if f.is_file() and f.name.endswith('_kelp.tif')]
    
#     for filename in tqdm(filenames, desc="Processing Images"):  # Added progress bar
#         try:
#             image_GT = Image.open(filename)
#             image_GT = np.array(image_GT)
#             kelp_count = np.sum(image_GT == 1)  # Count pixels where value is 1
#             kelp_counts.append(kelp_count)
#         except Exception as e:
#             print(f"Error processing {filename}: {e}") #print any errors.
            
#     return kelp_counts


# def plot_histogram(kelp_counts, bins=50):
#     """
#     Plots a histogram of the kelp pixel counts.

#     Args:
#         kelp_counts (list): A list of kelp pixel counts.
#         bins (int): The number of bins to use in the histogram.
#     """
#     plt.figure(figsize=(10, 6))
#     plt.hist(kelp_counts, bins=bins, color='skyblue', edgecolor='black')
#     plt.title('Distribution of Kelp Pixel Counts in Ground Truth Images', fontsize=16)
#     plt.xlabel('Number of Kelp Pixels', fontsize=14)
#     plt.ylabel('Number of Images', fontsize=14)
#     plt.grid(axis='y', alpha=0.75)
#     plt.show()
    
    
# def plot_histogram_with_zeros(kelp_counts, bins=50):
#     """
#     Plots a histogram, explicitly showing the zero counts.  Includes a separate
#     bar for zero counts and adjusts the x-axis limits.

#     Args:
#         kelp_counts (list): List of kelp pixel counts.
#         bins (int): Number of bins for the non-zero counts.
#     """
#     plt.figure(figsize=(12, 6))

#     # Separate zero and non-zero counts
#     zero_counts = kelp_counts.count(0)
#     non_zero_counts = [count for count in kelp_counts if count > 0]

#     # Plot zero counts as a separate bar
#     plt.bar(0, zero_counts, color='red', label='Zero Kelp Pixels', width=50)  # Adjust width as needed

#     # Plot non-zero counts
#     if non_zero_counts:  # Check if non_zero_counts is not empty
#         plt.hist(non_zero_counts, bins=bins, color='skyblue', edgecolor='black', label='Non-Zero Kelp Pixels', range=(1, max(kelp_counts)))
#         plt.xlim(-200, max(kelp_counts) + 100)  # Adjust x-axis limits (dynamic based off max value)
#     else: # in the condition that there are not any nonzero counts, only show the 0 count.
#         plt.xlim(-200, 1000)
        

#     plt.title('Distribution of Kelp Pixel Counts (with Explicit Zero Count)', fontsize=16)
#     plt.xlabel('Number of Kelp Pixels', fontsize=14)
#     plt.ylabel('Number of Images', fontsize=14)
#     plt.grid(axis='y', alpha=0.75)
#     plt.legend()
#     plt.show()


# def main():
#     """
#     Main function to execute the analysis.
#     """
#     # Get the root directory of the project (adjust if needed)
#     directory = Path().resolve().parent / "data" / "train_kelp" #changed the root directory
    
#     #Ensure files are present.
#     if not directory.exists():
#         raise FileNotFoundError(f"The directory {directory} does not exist. Please check path.")
    
#     kelp_counts = calculate_kelp_pixel_counts(directory)
    
#     # Plot the histogram (choose either regular or with explicit zero counts)
#     # plot_histogram(kelp_counts, bins=20)
#     plot_histogram_with_zeros(kelp_counts, bins=50)

#     print(f"Total number of images processed: {len(kelp_counts)}")
#     print(f"Number of images with zero kelp pixels: {kelp_counts.count(0)}")
#     print(f"Maximum kelp pixel count: {max(kelp_counts)}")
#     print(f"Minimum kelp pixel count: {min(kelp_counts)}")
#     print(f"Average kelp pixel count: {np.mean(kelp_counts):.2f}")
#     print(f"Median kelp pixel count: {np.median(kelp_counts):.2f}")



# if __name__ == "__main__":
#     main()

# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# from pathlib import Path
# import os
# from tqdm import tqdm  # For progress bar

# def calculate_kelp_pixel_counts(directory):
#     """
#     Calculates the number of kelp pixels (value 1) in each ground truth image and stores the filenames.

#     Args:
#         directory (Path): The path to the directory containing the ground truth images.

#     Returns:
#         list: A list of tuples, where each tuple contains (kelp_count, filename).
#     """
#     kelp_counts = []
#     filenames = [f for f in directory.iterdir() if f.is_file() and f.name.endswith('_kelp.tif')]

#     for filename in tqdm(filenames, desc="Processing Images"):
#         try:
#             image_GT = Image.open(filename)
#             image_GT = np.array(image_GT)
#             kelp_count = np.sum(image_GT == 1)
#             kelp_counts.append((kelp_count, filename.name))  # Store count and filename
#         except Exception as e:
#             print(f"Error processing {filename}: {e}")

#     return kelp_counts


# def plot_histogram_with_zeros_and_write_examples(kelp_counts_with_filenames, bins=50, output_file="example_filenames.txt"):
#     """
#     Plots a histogram, explicitly showing zero counts, and writes an example filename to a file for each bin.

#     Args:
#         kelp_counts_with_filenames (list): List of (kelp_count, filename) tuples.
#         bins (int): Number of bins for the non-zero counts.
#         output_file (str): Path to the output text file.
#     """
#     plt.figure(figsize=(12, 6))

#     # Separate zero and non-zero counts, keeping filenames
#     zero_counts = [(count, filename) for count, filename in kelp_counts_with_filenames if count == 0]
#     non_zero_counts = [(count, filename) for count, filename in kelp_counts_with_filenames if count > 0]
#     zero_count_num = len(zero_counts)


#     # --- Histogram plotting (same as before, but using counts from the separated lists) ---
#     plt.bar(0, zero_count_num, color='red', label='Zero Kelp Pixels', width=50)

#     if non_zero_counts:
#         non_zero_values = [count for count, _ in non_zero_counts]
#         counts, edges, _ = plt.hist(non_zero_values, bins=bins, color='skyblue', edgecolor='black', label='Non-Zero Kelp Pixels', range=(1, max(non_zero_values)))
#         plt.xlim(-200, max(non_zero_values) + 100)
#     else:
#         counts, edges = np.histogram([], bins=bins, range=(1,1)) # bins will not matter for empty array
#         plt.xlim(-200, 1000)


#     plt.title('Distribution of Kelp Pixel Counts (with Explicit Zero Count)', fontsize=16)
#     plt.xlabel('Number of Kelp Pixels', fontsize=14)
#     plt.ylabel('Number of Images', fontsize=14)
#     plt.grid(axis='y', alpha=0.75)
#     plt.legend()
#     plt.show()
#     # --- End of histogram plotting ---



#     # --- Write example filenames to file ---
#     with open(output_file, "w") as f:
#         f.write("Example Filenames for Each Bin:\n\n")

#         # Handle zero bin separately
#         if zero_counts:
#             example_zero_filename = zero_counts[0][1]  # Get the first filename with zero count
#             f.write(f"Bin [0]: {example_zero_filename}\n")

#         # Iterate through the non-zero bins
#         for i in range(len(counts)):
#             bin_lower = edges[i]
#             bin_upper = edges[i + 1]

#             # Find filenames within the current bin
#             examples_in_bin = [filename for count, filename in non_zero_counts if bin_lower <= count < bin_upper]

#             if examples_in_bin:
#                 example_filename = examples_in_bin[0]  # Take the first example
#                 f.write(f"Bin [{int(bin_lower)}-{int(bin_upper)}): {example_filename}\n")
#             else:
#                 f.write(f"Bin [{int(bin_lower)}-{int(bin_upper)}): No example found\n")
#     # --- End of file writing ---


# def main():
#     directory = Path().resolve().parent / "data" / "train_kelp"
#     if not directory.exists():
#         raise FileNotFoundError(f"The directory {directory} does not exist.")

#     kelp_counts_with_filenames = calculate_kelp_pixel_counts(directory)  # Get (count, filename) tuples

#     plot_histogram_with_zeros_and_write_examples(kelp_counts_with_filenames, bins=50)

#     # --- Calculate and print statistics (no changes here) ---
#     kelp_counts = [count for count, _ in kelp_counts_with_filenames] # Extract just the counts
#     print(f"Total number of images processed: {len(kelp_counts)}")
#     print(f"Number of images with zero kelp pixels: {kelp_counts.count(0)}")
#     print(f"Maximum kelp pixel count: {max(kelp_counts)}")
#     print(f"Minimum kelp pixel count: {min(kelp_counts)}")
#     print(f"Average kelp pixel count: {np.mean(kelp_counts):.2f}")
#     print(f"Median kelp pixel count: {np.median(kelp_counts):.2f}")
#     # --- End of statistics ---


# if __name__ == "__main__":
#     main()


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm

def calculate_kelp_pixel_counts(directory):
    """
    Calculates kelp pixel counts and returns a list of (count, filename) tuples.
    """
    kelp_counts = []
    filenames = [f for f in directory.iterdir() if f.is_file() and f.name.endswith('_kelp.tif')]

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



def main():
    directory = Path().resolve().parent / "data" / "train_kelp"
    if not directory.exists():
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    kelp_counts_with_filenames = calculate_kelp_pixel_counts(directory)

    # Adjust outlier thresholds as needed
    plot_histogram_with_outlier_handling(kelp_counts_with_filenames, bins=20,
                                         outlier_threshold_lower=0.01, outlier_threshold_upper=0.99)

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