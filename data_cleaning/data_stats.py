import tifffile as tiff
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
import warnings

# --- Configuration ---
# Assume the script is run from the project root or adjust as needed
try:
    # If running as a script, use command line arg for specific file check
    # For full analysis, this part can be ignored or adapted
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Usage: python analyze_data.py [optional_base_filename]")
        print("If filename is provided, only that file is analyzed.")
        print("Otherwise, the full dataset analysis is performed.")
        sys.exit(0)
    SPECIFIC_FILENAME_TO_CHECK = sys.argv[1] if len(sys.argv) > 1 else None
except Exception:
    SPECIFIC_FILENAME_TO_CHECK = None # Default if running interactively

# Define base directory (adjust if your script isn't in the project root)
# Using Path().resolve() assumes the script is run from the project root.
# If not, you might need: base_dir = Path(__file__).resolve().parent.parent # Example
base_dir = Path().resolve()
if not (base_dir / "data").exists():
     # A common structure might be src/scripts, data/
     base_dir = base_dir.parent
     if not (base_dir / "data").exists():
         raise FileNotFoundError("Could not automatically find the 'data' directory relative to the script.")


data_dir = base_dir / "data" / "original"
sat_dir = data_dir / "train_satellite"
kelp_dir = data_dir / "train_kelp"

# Check if directories exist
if not sat_dir.is_dir():
    raise FileNotFoundError(f"Satellite image directory not found: {sat_dir}")
if not kelp_dir.is_dir():
    raise FileNotFoundError(f"Kelp mask directory not found: {kelp_dir}")

# Band names for easier reference
BAND_NAMES = [
    "0_SWIR",
    "1_NIR",
    "2_Red",
    "3_Green",
    "4_Blue",
    "5_CloudMask",
    "6_DEM"
]

# Outlier detection thresholds (can be adjusted)
Z_SCORE_THRESHOLD = 3.0 # Standard deviations from the mean
IQR_MULTIPLIER = 1.5    # For IQR method: Q1 - mult*IQR, Q3 + mult*IQR

# --- Helper Functions ---

def calculate_band_stats(image_path: Path):
    """
    Loads a satellite image and calculates min, max, mean, std dev per band.
    Also checks basic properties like shape and dtype.
    """
    stats = {'filename': image_path.stem.replace('_satellite', '')}
    valid = True
    error_msg = None

    try:
        img = tiff.imread(image_path)

        # Basic checks
        if img.ndim != 3 or img.shape[2] != 7:
            stats['error'] = f"Incorrect shape: {img.shape}"
            valid = False
        # Add dtype check if needed, e.g. expecting uint16?
        # if img.dtype != np.uint16:
        #     stats['error'] = f"Unexpected dtype: {img.dtype}"
        #     valid = False

        if valid:
            for i, band_name in enumerate(BAND_NAMES):
                band_data = img[:, :, i].astype(np.float32) # Cast for calculations

                # Handle constant bands (std=0) to avoid division by zero later if needed
                band_min = np.min(band_data)
                band_max = np.max(band_data)
                band_mean = np.mean(band_data)
                band_std = np.std(band_data)

                stats[f'{band_name}_min'] = band_min
                stats[f'{band_name}_max'] = band_max
                stats[f'{band_name}_mean'] = band_mean
                stats[f'{band_name}_std'] = band_std

                # Specific checks
                if band_name == "5_CloudMask":
                     # Check if values are strictly 0 or 1
                     if not np.all(np.isin(band_data, [0, 1])):
                          unique_vals = np.unique(band_data)
                          stats[f'{band_name}_invalid_values'] = True
                          stats[f'{band_name}_unique_vals'] = unique_vals
                     else:
                          stats[f'{band_name}_invalid_values'] = False
                     # Percentage of clouds
                     stats[f'{band_name}_cloud_pct'] = np.mean(band_data) * 100

                if band_name == "6_DEM":
                     # Check for potentially unrealistic DEM values (e.g., negative elevation over ocean)
                     # This threshold is arbitrary, adjust based on expected range
                     if band_min < -100: # Allow for some negative values near sea level
                          stats[f'{band_name}_suspect_min'] = True
                     else:
                          stats[f'{band_name}_suspect_min'] = False

    except FileNotFoundError:
        stats['error'] = "File not found"
        valid = False
    except Exception as e:
        stats['error'] = f"Error reading/processing file: {e}"
        valid = False

    stats['is_valid'] = valid
    return stats

def find_outliers_iqr(series, multiplier=1.5):
    """Find outliers using the Interquartile Range method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return series[(series < lower_bound) | (series > upper_bound)]

def find_outliers_zscore(series, threshold=3.0):
    """Find outliers using the Z-score method."""
    mean = series.mean()
    std = series.std()
    if std == 0: # Avoid division by zero for constant series
        return pd.Series(dtype=series.dtype)
    z_scores = (series - mean) / std
    return series[np.abs(z_scores) > threshold]

# --- Main Analysis ---

print("Starting Data Analysis...")

# Get list of satellite images
sat_files = list(sat_dir.glob("*_satellite.tif"))
if not sat_files:
    print(f"Error: No satellite TIFF files found in {sat_dir}")
    sys.exit(1)

print(f"Found {len(sat_files)} satellite images.")

# If a specific file is requested, just analyze that one
if SPECIFIC_FILENAME_TO_CHECK:
    specific_file_path = sat_dir / f"{SPECIFIC_FILENAME_TO_CHECK}_satellite.tif"
    if specific_file_path.exists():
        print(f"\n--- Analyzing specific file: {SPECIFIC_FILENAME_TO_CHECK} ---")
        stats = calculate_band_stats(specific_file_path)
        # Pretty print the stats for the single file
        for key, value in stats.items():
             if isinstance(value, float):
                 print(f"  {key}: {value:.4f}")
             else:
                 print(f"  {key}: {value}")
        sys.exit(0) # Exit after analyzing the single file
    else:
        print(f"Error: Specific file not found: {specific_file_path}")
        sys.exit(1)


# --- Full Dataset Analysis ---
all_stats = []
print("Calculating statistics for each image...")
# Use tqdm for progress bar (works well in notebooks/consoles)
for f in tqdm(sat_files, desc="Processing Images"):
    # Optional: Check if corresponding kelp file exists
    kelp_file = kelp_dir / f.name.replace("_satellite.tif", "_kelp.tif")
    if not kelp_file.exists():
        warnings.warn(f"Missing corresponding kelp file for {f.name}")
        # Decide whether to skip or proceed if GT is missing
        # continue # uncomment to skip if GT is missing

    stats = calculate_band_stats(f)
    all_stats.append(stats)

# Convert statistics to a Pandas DataFrame
stats_df = pd.DataFrame(all_stats)
stats_df.set_index('filename', inplace=True)

print(f"\nProcessed {len(stats_df)} images.")

# --- Identify Problematic Files ---

problematic_files = {} # Dict to store filename -> reason

# 1. Check for loading errors or basic validation failures
error_files = stats_df[~stats_df['is_valid']]
if not error_files.empty:
    print(f"\nFound {len(error_files)} files with loading/validation errors:")
    for filename, row in error_files.iterrows():
        reason = f"Error: {row.get('error', 'Unknown validation error')}"
        print(f"  - {filename}: {reason}")
        problematic_files[filename] = problematic_files.get(filename, []) + [reason]

# Filter out invalid files for statistical analysis
valid_stats_df = stats_df[stats_df['is_valid']].copy()
# Drop boolean/error columns before calculating numerical stats summaries
cols_to_drop = ['is_valid', 'error'] + [col for col in valid_stats_df if 'invalid_values' in col or 'suspect_min' in col or 'unique_vals' in col]
valid_stats_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

if valid_stats_df.empty:
    print("\nNo valid files remaining for statistical outlier analysis.")
    sys.exit(0)

print(f"\n--- Analyzing Statistics for {len(valid_stats_df)} Valid Images ---")

# Display overall statistics summary
print("\nOverall Descriptive Statistics (Valid Images):")
# Select only numerical columns for describe()
numerical_cols = valid_stats_df.select_dtypes(include=np.number).columns
print(valid_stats_df[numerical_cols].describe().to_string())


# 2. Specific Band Checks (Cloud Mask values, DEM min)
cloud_mask_invalid = stats_df[stats_df['5_CloudMask_invalid_values'] == True]
if not cloud_mask_invalid.empty:
    print(f"\nFound {len(cloud_mask_invalid)} files with invalid Cloud Mask values (not 0 or 1):")
    for filename, row in cloud_mask_invalid.iterrows():
        reason = f"CloudMask has values other than 0/1: {row['5_CloudMask_unique_vals']}"
        print(f"  - {filename}: {row['5_CloudMask_unique_vals']}")
        problematic_files[filename] = problematic_files.get(filename, []) + [reason]

dem_suspect = stats_df[stats_df['6_DEM_suspect_min'] == True]
if not dem_suspect.empty:
    print(f"\nFound {len(dem_suspect)} files with suspect DEM minimum values (< -100):")
    for filename, row in dem_suspect.iterrows():
        reason = f"DEM min is suspiciously low: {row['6_DEM_min']:.2f}"
        print(f"  - {filename}: {row['6_DEM_min']:.2f}")
        problematic_files[filename] = problematic_files.get(filename, []) + [reason]


# # 3. Statistical Outlier Detection (Min, Max, Std Dev using IQR)
print(f"\n--- Detecting Statistical Outliers (using IQR Multiplier: {IQR_MULTIPLIER}) ---")
for band_name in BAND_NAMES:
    print(f"\n  Outliers for Band: {band_name}")
    for stat in ['min', 'max', 'std']:
        col_name = f'{band_name}_{stat}'
        if col_name in valid_stats_df.columns:
            outliers = find_outliers_iqr(valid_stats_df[col_name], multiplier=IQR_MULTIPLIER)
            if not outliers.empty:
                print(f"    - {stat.capitalize()} Outliers ({len(outliers)}):")
                for filename, value in outliers.items():
                    reason = f"{band_name} {stat} ({value:.4f}) is an outlier (IQR method)"
                    print(f"      - {filename}: {value:.4f}")
                    problematic_files[filename] = problematic_files.get(filename, []) + [reason]
            # else:
            #     print(f"    - No significant outliers found for {stat}.")


# --- Optional: Outlier Detection using Z-score ---
# print(f"\n--- Detecting Statistical Outliers (using Z-score Threshold: {Z_SCORE_THRESHOLD}) ---")
# for band_name in BAND_NAMES:
#     print(f"\n  Outliers for Band: {band_name}")
#     for stat in ['min', 'max', 'std', 'mean']:
#         col_name = f'{band_name}_{stat}'
#         if col_name in valid_stats_df.columns:
#             outliers = find_outliers_zscore(valid_stats_df[col_name], threshold=Z_SCORE_THRESHOLD)
#             if not outliers.empty:
#                 print(f"    - {stat.capitalize()} Outliers ({len(outliers)}):")
#                 for filename, value in outliers.items():
#                     reason = f"{band_name} {stat} ({value:.4f}) is an outlier (Z-score > {Z_SCORE_THRESHOLD})"
#                     # print(f"      - {filename}: {value:.4f}") # Uncomment if you want details here too
#                     problematic_files[filename] = problematic_files.get(filename, []) + [reason]


# --- Generate Visualizations (Optional but Recommended) ---
# Create plots to visually inspect distributions
plot_dir = base_dir / "analysis_plots"
plot_dir.mkdir(exist_ok=True)
print(f"\nGenerating distribution plots in: {plot_dir}")

for band_name in BAND_NAMES:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Distribution of Stats for Band: {band_name}', fontsize=16)

    for i, stat in enumerate(['min', 'max', 'std']):
        col_name = f'{band_name}_{stat}'
        if col_name in valid_stats_df.columns:
            sns.histplot(valid_stats_df[col_name], kde=True, ax=axes[i])
            axes[i].set_title(f'{stat.capitalize()} Distribution')
            axes[i].set_xlabel(stat.capitalize())
            axes[i].set_ylabel('Frequency')
        else:
             axes[i].set_title(f'{stat.capitalize()} (Not Available)')
             axes[i].axis('off') # Turn off axis if column doesn't exist

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plot_filename = plot_dir / f"distribution_{band_name}.png"
    plt.savefig(plot_filename)
    plt.close(fig) # Close the figure to free memory

# # Plot cloud percentage distribution
# cloud_pct_col = '5_CloudMask_cloud_pct'
# if cloud_pct_col in valid_stats_df.columns:
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.histplot(valid_stats_df[cloud_pct_col], kde=False, ax=ax)
#     ax.set_title('Distribution of Cloud Percentage (Band 5)')
#     ax.set_xlabel('Cloud Percentage')
#     ax.set_ylabel('Frequency')
#     plt.tight_layout()
#     plt.savefig(plot_dir / "distribution_cloud_percentage.png")
#     plt.close(fig)


# --- Summary of Problematic Files ---
print("\n--- Summary of Potentially Problematic Files ---")
if not problematic_files:
    print("No potentially problematic files identified based on the criteria.")
else:
    print(f"Found {len(problematic_files)} potentially problematic files:")
    # Sort by filename for consistent output
    sorted_filenames = sorted(problematic_files.keys())
    for filename in sorted_filenames:
        reasons = "; ".join(problematic_files[filename])
        print(f"  - {filename}: {reasons}")

    # Save the list to a file
    output_file = base_dir / "problematic_files.txt"
    print(f"\nSaving list of problematic filenames to: {output_file}")
    with open(output_file, 'w') as f:
         f.write("Filename\tReason(s)\n")
         for filename in sorted_filenames:
            reasons = "; ".join(problematic_files[filename])
            f.write(f"{filename}\t{reasons}\n")


print("\nAnalysis complete.")