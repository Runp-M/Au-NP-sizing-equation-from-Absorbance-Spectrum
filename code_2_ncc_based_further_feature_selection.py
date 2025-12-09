# -*- coding: utf-8 -*-
"""

#!/usr/bin/env python3

#   Author:         Runpeng Miao
#   Contact:        Runpeng.miao@phd.unipd.it
#   Version:        1.0
#   Date:           September 2025

This script performs the following:
1. Loads a CSV file containing absorbance spectra data (top 10 wavelengths) and NP sizes.
2. Calculates the nonlinear correlation coefficient (NLC) between each wavelength and the NP size.
3. Fits an exponential model for each wavelength vs size.
4. Visualizes scatter plots with the fitted exponential curve.
5. Outputs the NLC values for all wavelengths.

CSV input format (example):
-------------------------------------------------------------
390     420     376     361     Size
0.5527  0.5173  0.5709  0.5846  35
0.5812  0.5459  0.6023  0.6198  23
...
-------------------------------------------------------------
- Columns 1-10: absorbance values at top 10 wavelengths (user-defined)
- Last column: measured NP size (Size)
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
packages = ['pandas', 'numpy', 'matplotlib', 'nlcor', 'scipy']
for pkg in packages:
    install_if_missing(pkg)

# -------------------------------
# User Inputs
# -------------------------------
INPUT_FILE = "./Top 10 wavelengths of Absorbance spectra.csv"  # CSV file path

# -------------------------------
# Import Required Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nlcor import nlcor
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

# -------------------------------
# Load Data
# -------------------------------
data = pd.read_csv(INPUT_FILE)

# Extract top 10 wavelengths and NP size
top10_wavelengths = data.columns[:-1]  # assuming last column is 'Size'
Y = data['Size'].values
X = data[top10_wavelengths]

print("Top 10 wavelengths in the dataset:", list(top10_wavelengths))

# -------------------------------
# Define Exponential Fitting Function
# -------------------------------
def exponential_func(x, a, b, c):
    """Exponential function: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c

# -------------------------------
# Nonlinear Correlation and Exponential Fit
# -------------------------------
nlcor_results = []

for col in X.columns:
    # Compute nonlinear correlation coefficient
    result = nlcor(X[col], Y)
    nlcor_value = result['cor_estimate']
    nlcor_results.append(nlcor_value)
    print(f"Nonlinear correlation between {col} and Size:", nlcor_value)

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X[col], Y, color='skyblue', label=f'NLC: {nlcor_value:.3f}')

    # Exponential fitting
    x_fit = np.linspace(X[col].min(), X[col].max(), 100)
    try:
        popt, _ = curve_fit(exponential_func, X[col], Y, maxfev=10000)
        y_fit = exponential_func(x_fit, *popt)
        plt.plot(x_fit, y_fit, 'r-', label=f'Exponential Fit: y={popt[0]:.3f}*exp({popt[1]:.3f}*x)+{popt[2]:.3f}')
    except RuntimeError:
        plt.plot([], [], 'r-', label='Exponential Fit failed')

    plt.xlabel(col, fontsize=12)
    plt.ylabel('Size (nm)', fontsize=12)
    plt.title(f'Exponential Fit and NLC for {col}', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# -------------------------------
# Summary Bar Plot of Nonlinear Correlation Coefficients
# -------------------------------
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(top10_wavelengths)+1), nlcor_results, color='skyblue')
plt.xticks(range(1, len(top10_wavelengths)+1), top10_wavelengths)
plt.xlabel('Wavelengths', fontsize=12)
plt.ylabel('Nonlinear Correlation Coefficient', fontsize=12)
plt.title('Nonlinear Correlation Coefficient between Top 10 Wavelengths and NP Size', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# -------------------------------
# Notes
# -------------------------------
# - User can use the NLC results to select the most correlated wavelengths.
# - Selected wavelengths can be input to the GA-based polynomial/exponential optimization script.
# - Ensure the CSV file has the last column as 'Size' and the first 10 columns as absorbance values.