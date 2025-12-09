# -*- coding: utf-8 -*-
"""
#!/usr/bin/env python3

Process UV-Vis spectra files (CSV or two-column DAT format) in a given folder.

#   Author:         Runpeng Miao
#   Contact:        Runpeng.miao@phd.unipd.it
#   Version:        1.0
#   Date:           September 2025

Workflow:
1. Automatic file detection:
   - The program scans the folder and determines the file type.
   - If CSV files are found, all files in the folder are assumed to be CSV (exported from UV-Vis spectrometers, e.g., JASCO).
   - If DAT files are found, all files in the folder are assumed to be DAT (two-column text: wavelength and absorbance).
   - The folder must contain either only CSV files or only DAT files. Mixed formats are not supported.

2. Data extraction:
   - For CSV files:
     - Metadata and headers are skipped.
     - Spectral data starts after a line containing "XYDATA".
     - Each data row: wavelength, absorbance
       - Columns can be separated by commas (`,`), spaces (` `), or tabs (`\t`)
       - Leading or trailing spaces/tabs are ignored
       - Only the first two numeric columns are used; extra columns are ignored
   - For DAT files:
     - Plain text with two columns per row: wavelength and absorbance
     - Columns separated by spaces or tabs
     - Leading or trailing spaces/tabs are ignored
     - No header is required

3. Data processing:
   - Interpolate each spectrum to a common wavelength range of 300–900 nm with 1 nm interval.
   - Normalize each spectrum by the maximum absorbance in the 400–600 nm range.
   - Detect and warn if the surface plasmon resonance (SPR) peak wavelength is greater than 580 nm.

4. Output:
   - Saves a single CSV file with the following format:
     File Name | 300 | 301 | 302 | ... | 900
   - Each row corresponds to one spectrum with normalized absorbance values.
   - The output CSV is saved in the same folder level as the input spectra folder.
"""

import subprocess
import sys

# Optionally install missing packages using subprocess
required_packages = ["numpy", "scipy"]
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import os
import csv
import numpy as np
from scipy.interpolate import interp1d

# -------------------------
# 0. Specify input folder
# -------------------------
# Enter the spectra folder path
input_folder = input("Enter the full path of the spectra folder: ").strip()

# Check if folder exists
if not os.path.isdir(input_folder):
    raise ValueError(f"The folder '{input_folder}' does not exist. Please check the path.")

print(f"Spectra folder set to: {input_folder}")

# -------------------------
# 1. Data extraction functions
# -------------------------
def extract_spectra_from_csv(csv_file_path):
    """
    Extract wavelength and absorbance from JASCO CSV (comma-separated).
    The function still works if lines have extra spaces or tab characters at the end.

    Only supports the following delimiters in the data lines:
        - Comma ','  (default JASCO CSV)
        - Space ' ' or Tab '\\t'  (line.split() handles these)

    Returns:
        wavelengths: np.array of wavelengths
        absorbances: np.array of absorbance values
    """
    wavelengths, absorbances = [], []

    with open(csv_file_path, 'r', encoding='gbk') as infile:
        reader = csv.reader(infile)
        xydata_found = False

        for idx, row in enumerate(reader):
            if not row:
                continue

            if not xydata_found and any('XYDATA' in cell.upper() for cell in row):
                xydata_found = True
                print(f"'XYDATA' found at line {idx+1}")
                continue

            if xydata_found:
                try:
                    wavelength = float(row[0])
                    absorbance = float(row[1])
                    wavelengths.append(wavelength)
                    absorbances.append(absorbance)
                except (ValueError, IndexError):
                    continue

    if len(wavelengths) == 0:
        raise ValueError(f"No numeric data found in {csv_file_path} after 'XYDATA'")

    return np.array(wavelengths), np.array(absorbances)

def extract_spectra_from_dat(dat_file_path):
    """Extract wavelength and absorbance from a two-column DAT file."""
    with open(dat_file_path, 'r', encoding='gbk') as infile:
        lines = [line.strip() for line in infile if line.strip()]

    wavelengths, absorbances = [], []
    for line in lines:
        elems = line.split()
        if len(elems) < 2:
            continue
        wavelengths.append(float(elems[0]))
        absorbances.append(float(elems[1]))

    return np.array(wavelengths), np.array(absorbances)

# -------------------------
# 2. Batch processing function
# -------------------------
def batch_process_spectra(input_folder, file_type='csv'):
    """
    Process all spectra files of the given type in input_folder:
    - file_type: 'csv' or 'dat'
    - Interpolate to 300–900 nm (1 nm step)
    - Normalize using max value in 400–600 nm range
    - Warn if SPR peak > 580 nm
    - Save results to 'NP_spectra_interpolated_normalized.csv'
      at the same level as input_folder
    """
    if file_type == 'csv':
        files_list = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]
        extract_func = extract_spectra_from_csv
    elif file_type == 'dat':
        files_list = [f for f in os.listdir(input_folder) if f.lower().endswith('.dat')]
        extract_func = extract_spectra_from_dat
    else:
        raise ValueError("file_type must be 'csv' or 'dat'")

    print(f"Found {len(files_list)} {file_type} files in {input_folder}: {files_list}")

    interp_wavelengths = np.arange(300, 901, 1)
    parent_folder = os.path.dirname(os.path.abspath(input_folder))
    output_csv_path = os.path.join(parent_folder, 'NP_spectra_interpolated_normalized.csv')
    out_of_range_files = []

    with open(output_csv_path, 'w', encoding='utf-8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ['File Name'] + list(interp_wavelengths)
        csv_writer.writerow(header)

        for file_name in files_list:
            file_path = os.path.join(input_folder, file_name)
            print(f"\nProcessing {file_type} file: {file_name}")

            try:
                wavelengths, absorbances = extract_func(file_path)
            except ValueError as e:
                print(e)
                continue

            if len(wavelengths) < 2:
                print(f"Not enough data in {file_name}, skipping...")
                continue

            f_interp = interp1d(
                wavelengths,
                absorbances,
                kind='linear',
                bounds_error=False,
                fill_value="extrapolate"
            )
            interp_abs = f_interp(interp_wavelengths)

            mask = (interp_wavelengths >= 400) & (interp_wavelengths <= 600)
            max_val = np.max(interp_abs[mask])
            if max_val != 0:
                interp_abs /= max_val

            spr_peak_wl = interp_wavelengths[np.argmax(interp_abs)]
            if spr_peak_wl > 580:
                out_of_range_files.append((file_name, spr_peak_wl))

            row = [file_name] + list(interp_abs)
            csv_writer.writerow(row)

    if out_of_range_files:
        print("\nWARNING: Some NP samples have SPR peak wavelength > 580 nm:")
        for fname, peak in out_of_range_files:
            print(f"File: {fname}, SPR peak: {peak:.1f} nm")
        print(f"Total out-of-range samples: {len(out_of_range_files)}")
    else:
        print("\nAll NP samples are within SPR peak range <= 580 nm.")

    print(f"\nProcessing complete. Saved to {output_csv_path}")

# -------------------------
# 3. Main
# -------------------------
if __name__ == "__main__":
    current_files = os.listdir(input_folder)

    if any(f.lower().endswith('.csv') for f in current_files):
        print("Detected CSV spectra files. Running CSV processing...")
        batch_process_spectra(input_folder, file_type='csv')
    elif any(f.lower().endswith('.dat') for f in current_files):
        print("Detected DAT spectra files. Running DAT processing...")
        batch_process_spectra(input_folder, file_type='dat')
    else:
        print("No CSV or DAT spectra files found in the specified folder.")