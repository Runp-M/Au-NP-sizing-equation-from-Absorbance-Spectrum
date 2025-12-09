# -*- coding: utf-8 -*-
"""

#!/usr/bin/env python3

This script performs the following:

1. Loads a CSV file containing absorbance spectra of Au nanoparticles.
2. Finds the SPR peak (maximum absorbance) in the spectrum.
3. If the SPR peak wavelength is between 545–580 nm, it estimates the NP size
   using linear interpolation of the Mie prediction table (SPR λ vs. mean NP size).

#   Author:         Runpeng Miao
#   Contact:        Runpeng.miao@phd.unipd.it
#   Version:        1.0
#   Date:           September 2025

Input CSV format:
- Column 1: wavelength (nm)
- Column 2: absorbance value

Notes:
- The CSV can contain more columns, but only the first two are used for SPR detection.
- Wavelengths must be numeric and sorted ascendingly for accurate peak detection.
- Output: estimated NP size (nm) at the SPR peak within the specified range.
"""

import subprocess
import sys

# Function to install a package if not already installed
def install_if_missing(package_name):
    try:
        __import__(package_name)
    except ImportError:
        print(f"{package_name} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Check and install required packages
packages = ['pandas', 'numpy', 'scipy']
for pkg in packages:
    install_if_missing(pkg)

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# -------------------------------
# Mie Prediction Table
# -------------------------------
data = {
    "SRP peak wavelength": [
        545, 545.5, 546, 546.5, 547, 547.5, 548, 548.5, 549, 549.5, 550, 550.5, 551, 551.5, 552,
        552.5, 553, 553.5, 554, 554.5, 555, 555.5, 556, 556.5, 557, 557.5, 558, 558.5, 559, 559.5,
        560, 560.5, 561, 561.5, 562, 562.5, 563, 563.5, 564, 564.5, 565, 565.5, 566, 566.5, 567,
        567.5, 568, 568.5, 569, 569.5, 570, 570.5, 571, 571.5, 572, 572.5, 573, 573.5, 574, 574.5,
        575, 575.5, 576, 576.5, 577, 577.5, 578, 578.5, 579, 579.5, 580
    ],
    "Au NP mean size": [
        82, 82.6, 83.2, 83.8, 84.4, 85, 85.6, 86.2, 86.8, 87.4, 88, 88.6, 89.2, 89.8, 90.4, 91, 91.6,
        92.2, 92.8, 93.4, 94, 94.6, 95.2, 95.8, 96.4, 97, 97.6, 98.2, 98.8, 99.4, 100, 100.42857,
        100.85714, 101.28571, 101.71429, 102.14286, 102.57143, 103, 103.42857, 103.85714, 104.28571,
        104.71429, 105.14286, 105.57143, 106, 106.42857, 106.85714, 107.28571, 107.71429, 108.14286,
        108.57143, 109, 109.42857, 109.85714, 110.28571, 110.71429, 111.14286, 111.57143, 112,
        112.42857, 112.85714, 113.28571, 113.71429, 114.14286, 114.57143, 115, 115.42857, 115.85714,
        116.28571, 116.71429, 117.14286
    ]
}

df_table = pd.DataFrame(data)
interp_func = interp1d(df_table["wavelength"], df_table["size"], kind="linear")

# -------------------------------
# Example: absorbance spectra CSV
# Columns: wavelength, absorbance
# -------------------------------
spectra_file = "./example_spectrum.csv"
df_spectra = pd.read_csv(spectra_file)

# Find SPR peak (wavelength of maximum absorbance)
spr_peak = df_spectra.loc[df_spectra['absorbance'].idxmax(), 'wavelength']
print(f"SPR peak wavelength = {spr_peak} nm")

# If SPR peak is in 545–580 nm, also estimate NP size using Mie table
if 545 <= spr_peak <= 580:
    estimated_size = interp_func(spr_peak)
    print(f"Estimated Au NP size from Mie table: {estimated_size:.2f} nm")
else:
    print("SPR peak is outside 545–580 nm; cannot estimate size from table.")