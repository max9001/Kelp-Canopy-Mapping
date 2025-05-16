import tifffile as tiff
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings # Kept for warnings.warn

'''
Prints a summary of overall descriptive statistics for the valid images (mean, std, min, max, quartiles for each band's stats).

Generates and saves histogram plots for the distribution of min, max, and standard deviation for each band. This helps visually assess the data distribution and identify skewness or potential outliers.

plots the distribution of cloud percentages.

Prints a final summary of all identified problematic files and their issues.
Saves this list of problematic files and their reasons to a text file (problematic_files.txt).
'''

DATA_FOLDER_NAME = "original"      # "original", "cleaned"
PLOT_OUTPUT_SUBDIR_NAME = "analysis_plots" # Subdirectory within ROOT_DIR for saving plots


ROOT_DIR = Path().resolve()
if not (ROOT_DIR / "data").exists():
    ROOT_DIR = ROOT_DIR.parent
    if not (ROOT_DIR / "data").exists():
        ROOT_DIR = ROOT_DIR.parent
        if not (ROOT_DIR / "data").exists():
            raise FileNotFoundError(
                "Could not automatically find the 'data' directory. "
                "Please ensure your 'data' directory is in the project root "
                "or adjust ROOT_DIR manually."
            )
print(f"Using project root directory: {ROOT_DIR}")



# --- Derived Paths ---
SATELLITE_SUBDIR_NAME = "train_satellite"
KELP_SUBDIR_NAME = "train_kelp"
BASE_DATA_DIR = ROOT_DIR / "data" / DATA_FOLDER_NAME
SAT_DIR = BASE_DATA_DIR / SATELLITE_SUBDIR_NAME
KELP_DIR = BASE_DATA_DIR / KELP_SUBDIR_NAME
PLOT_DIR = ROOT_DIR / PLOT_OUTPUT_SUBDIR_NAME # Plots saved to root/analysis_plots

# --- Analysis Parameters ---
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
IQR_MULTIPLIER = 1.5    # For IQR method: Q1 - mult*IQR, Q3 + mult*IQR
# Z_SCORE_THRESHOLD = 3.0 # Standard deviations from the mean (Z-score method currently commented out)

# --- End Global Configuration Constants ---


# --- Helper Functions ---

def calculate_band_stats(image_path: Path):
    """
    Loads a satellite image and calculates min, max, mean, std dev per band.
    Also checks basic properties like shape and dtype.
    """
    stats = {'filename': image_path.stem.replace('_satellite', '')}
    valid = True
    # error_msg = None # Not used

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

                band_min = np.min(band_data)
                band_max = np.max(band_data)
                band_mean = np.mean(band_data)
                band_std = np.std(band_data)

                stats[f'{band_name}_min'] = band_min
                stats[f'{band_name}_max'] = band_max
                stats[f'{band_name}_mean'] = band_mean
                stats[f'{band_name}_std'] = band_std

                if band_name == "5_CloudMask":
                     if not np.all(np.isin(band_data, [0, 1])):
                          unique_vals = np.unique(band_data)
                          stats[f'{band_name}_invalid_values'] = True
                          stats[f'{band_name}_unique_vals'] = unique_vals
                     else:
                          stats[f'{band_name}_invalid_values'] = False
                     stats[f'{band_name}_cloud_pct'] = np.mean(band_data) * 100

                if band_name == "6_DEM":
                     if band_min < -100:
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
    if iqr == 0: # Handle cases where IQR is zero (e.g., constant series)
        return pd.Series(dtype=series.dtype)
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return series[(series < lower_bound) | (series > upper_bound)]

# def find_outliers_zscore(series, threshold=3.0): # Z-score currently unused
#     """Find outliers using the Z-score method."""
#     mean = series.mean()
#     std = series.std()
#     if std == 0:
#         return pd.Series(dtype=series.dtype)
#     z_scores = (series - mean) / std
#     return series[np.abs(z_scores) > threshold]

# --- Main Analysis ---

def main():
    print("Starting Full Dataset Analysis...")

    # Check if directories exist using global paths
    if not SAT_DIR.is_dir():
        raise FileNotFoundError(f"Satellite image directory not found: {SAT_DIR}")
    if not KELP_DIR.is_dir():
        raise FileNotFoundError(f"Kelp mask directory not found: {KELP_DIR}")

    # Get list of satellite images
    sat_files = list(SAT_DIR.glob("*_satellite.tif"))
    if not sat_files:
        print(f"Error: No satellite TIFF files found in {SAT_DIR}")
        return # Changed from sys.exit(1) to allow use in other contexts

    print(f"Found {len(sat_files)} satellite images in {SAT_DIR}.")

    # --- Full Dataset Analysis ---
    all_stats = []
    print("Calculating statistics for each image...")
    for f_path in tqdm(sat_files, desc="Processing Images"):
        kelp_file_path = KELP_DIR / f_path.name.replace("_satellite.tif", "_kelp.tif")
        if not kelp_file_path.exists():
            warnings.warn(f"Missing corresponding kelp file for {f_path.name}")
            # Decide whether to skip or proceed if GT is missing
            # continue # uncomment to skip if GT is missing

        stats = calculate_band_stats(f_path)
        all_stats.append(stats)

    stats_df = pd.DataFrame(all_stats)
    if stats_df.empty:
        print("No statistics were generated. Exiting.")
        return
    stats_df.set_index('filename', inplace=True)

    print(f"\nProcessed {len(stats_df)} images.")

    # --- Identify Problematic Files ---
    problematic_files = {}

    error_files = stats_df[~stats_df['is_valid']]
    if not error_files.empty:
        print(f"\nFound {len(error_files)} files with loading/validation errors:")
        for filename, row in error_files.iterrows():
            reason = f"Error: {row.get('error', 'Unknown validation error')}"
            print(f"  - {filename}: {reason}")
            problematic_files[filename] = problematic_files.get(filename, []) + [reason]

    valid_stats_df = stats_df[stats_df['is_valid']].copy()
    cols_to_drop = ['is_valid', 'error'] + [col for col in valid_stats_df if 'invalid_values' in col or 'suspect_min' in col or 'unique_vals' in col]
    valid_stats_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    if valid_stats_df.empty:
        print("\nNo valid files remaining for statistical outlier analysis.")
        return # Changed from sys.exit(0)

    print(f"\n--- Analyzing Statistics for {len(valid_stats_df)} Valid Images ---")
    print("\nOverall Descriptive Statistics (Valid Images):")
    numerical_cols = valid_stats_df.select_dtypes(include=np.number).columns
    print(valid_stats_df[numerical_cols].describe().to_string())

    cloud_mask_invalid = stats_df[stats_df.get('5_CloudMask_invalid_values', pd.Series(False, index=stats_df.index)) == True] # Handle if column missing
    if not cloud_mask_invalid.empty:
        print(f"\nFound {len(cloud_mask_invalid)} files with invalid Cloud Mask values (not 0 or 1):")
        for filename, row in cloud_mask_invalid.iterrows():
            reason = f"CloudMask has values other than 0/1: {row.get('5_CloudMask_unique_vals', 'N/A')}"
            print(f"  - {filename}: {row.get('5_CloudMask_unique_vals', 'N/A')}")
            problematic_files[filename] = problematic_files.get(filename, []) + [reason]

    dem_suspect_min_col = '6_DEM_suspect_min'
    dem_suspect = stats_df[stats_df.get(dem_suspect_min_col, pd.Series(False, index=stats_df.index)) == True]
    if not dem_suspect.empty:
        print(f"\nFound {len(dem_suspect)} files with suspect DEM minimum values (< -100):")
        for filename, row in dem_suspect.iterrows():
            dem_min_val = row.get('6_DEM_min', np.nan)
            reason = f"DEM min is suspiciously low: {dem_min_val:.2f}"
            print(f"  - {filename}: {dem_min_val:.2f}")
            problematic_files[filename] = problematic_files.get(filename, []) + [reason]

    print(f"\n--- Detecting Statistical Outliers (using IQR Multiplier: {IQR_MULTIPLIER}) ---")
    for band_name in BAND_NAMES:
        print(f"\n  Outliers for Band: {band_name}")
        for stat_type in ['min', 'max', 'std']:
            col_name = f'{band_name}_{stat_type}'
            if col_name in valid_stats_df.columns:
                outliers = find_outliers_iqr(valid_stats_df[col_name], multiplier=IQR_MULTIPLIER)
                if not outliers.empty:
                    print(f"    - {stat_type.capitalize()} Outliers ({len(outliers)}):")
                    for filename, value in outliers.items():
                        reason = f"{band_name} {stat_type} ({value:.4f}) is an outlier (IQR method)"
                        print(f"      - {filename}: {value:.4f}")
                        problematic_files[filename] = problematic_files.get(filename, []) + [reason]

    # --- Generate Visualizations ---
    PLOT_DIR.mkdir(parents=True, exist_ok=True) # Ensure plot directory exists
    print(f"\nGenerating distribution plots in: {PLOT_DIR}")

    for band_name in BAND_NAMES:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Distribution of Stats for Band: {band_name}', fontsize=16)
        for i, stat_type in enumerate(['min', 'max', 'std']):
            col_name = f'{band_name}_{stat_type}'
            if col_name in valid_stats_df.columns:
                sns.histplot(valid_stats_df[col_name], kde=True, ax=axes[i])
                axes[i].set_title(f'{stat_type.capitalize()} Distribution')
                axes[i].set_xlabel(stat_type.capitalize())
                axes[i].set_ylabel('Frequency')
            else:
                 axes[i].set_title(f'{stat_type.capitalize()} (Not Available)')
                 axes[i].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = PLOT_DIR / f"distribution_{band_name}.png"
        plt.savefig(plot_filename)
        plt.close(fig)

    # Plot cloud percentage distribution (always on)
    cloud_pct_col = '5_CloudMask_cloud_pct'
    if cloud_pct_col in valid_stats_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(valid_stats_df[cloud_pct_col], kde=False, ax=ax, bins=20) # Added bins for clarity
        ax.set_title('Distribution of Cloud Percentage (Band 5)')
        ax.set_xlabel('Cloud Percentage (%)')
        ax.set_ylabel('Number of Images')
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "distribution_cloud_percentage.png")
        plt.close(fig)
        print(f"Saved cloud percentage distribution plot to: {PLOT_DIR / 'distribution_cloud_percentage.png'}")
    else:
        print(f"Column '{cloud_pct_col}' not found in valid_stats_df, skipping cloud percentage plot.")


    # --- Summary of Problematic Files ---
    print("\n--- Summary of Potentially Problematic Files ---")
    if not problematic_files:
        print("No potentially problematic files identified based on the criteria.")
    else:
        print(f"Found {len(problematic_files)} potentially problematic files:")
        sorted_filenames = sorted(problematic_files.keys())
        for filename in sorted_filenames:
            reasons = "; ".join(problematic_files[filename])
            print(f"  - {filename}: {reasons}")

        output_file_path = ROOT_DIR / "problematic_files.txt" # Save to project root
        print(f"\nSaving list of problematic filenames to: {output_file_path}")
        with open(output_file_path, 'w') as f:
             f.write("Filename\tReason(s)\n")
             for filename in sorted_filenames:
                reasons = "; ".join(problematic_files[filename])
                f.write(f"{filename}\t{reasons}\n")

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()