# -*- coding: utf-8 -*-
"""

#!/usr/bin/env python3

Au Nanoparticle Size Prediction from Absorbance Spectra.

#   Author:         Runpeng Miao
#   Contact:        Runpeng.miao@phd.unipd.it
#   Version:        1.0
#   Date:           September 2025

This script uses an optimized polynomial or exponential model (obtained from
GA optimization on selected wavelengths) to predict Au NP sizes from a new
absorbance spectra dataset.
-------------------------------
Expected CSV Format:
-------------------------------
The input CSV must include:
- Columns corresponding to the wavelengths used in the optimized model
  (e.g., '390', '420', '376', '361', etc.)
- A 'Size' column (actual NP size, for reference/comparison)
- Each row corresponds to one sample

Example:
390     420     376     361     Size
0.5527  0.5173  0.5709  0.5846  35
0.5812  0.5459  0.6023  0.6198  23
...
-------------------------------
Workflow:
-------------------------------
1. Load the optimized model parameters (JSON file containing selected wavelengths,
   model form 'poly' or 'exp', and optimized GA parameters).
2. Read the new spectra CSV file.
3. Extract values corresponding to selected wavelengths.
4. For each sample:
    - Predict NP size using the optimized model
    - Check if predicted size < 5 nm; if so, raise a warning
5. Save results to a new CSV with original wavelengths and predicted size.
6. Print any warnings for samples outside accuracy limits.
"""

import subprocess
import sys

# Optionally install missing packages using subprocess
required_packages = [
    "pandas",
    "numpy",
    "scikit-learn",
]

for package in required_packages:
    try:
        __import__(package)  # Try to import
    except ImportError:
        print(f"{package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import pandas as pd
import numpy as np
import json
import os

# -------------------------------
# User Inputs
# -------------------------------
INPUT_FILE = "./new_Au_NP_spectra.csv"   # CSV file containing selected wavelengths + Size
OPT_MODEL_FILE = "./optimized_model.json"  # JSON file from code 4:GA-based optimization of polynomial or exponential models for GA+ES or NCC-selected wavelengths
OUTPUT_FILE = "./predicted_sizes.csv"      # Output CSV with predicted sizes

# -------------------------------
# Load optimized model parameters
# -------------------------------
with open(OPT_MODEL_FILE, "r") as f:
    opt_model = json.load(f)

# Extract selected wavelengths, model form, and optimized parameters
selected_wavelengths = opt_model["selected_wavelengths"]  # List of wavelength strings
MODEL_FORM = opt_model["model_form"]                       # 'poly' or 'exp'
optimized_params = np.array(opt_model["optimized_params"])
NUM_PREDICTORS = len(selected_wavelengths)

# -------------------------------
# Load new spectra dataset
# -------------------------------
data = pd.read_csv(INPUT_FILE)

# Check that all selected wavelengths are present in the CSV
missing_wavs = [w for w in selected_wavelengths if w not in data.columns]
if missing_wavs:
    raise ValueError(f"Missing wavelengths in CSV: {missing_wavs}")

# Extract wavelength values (X) and true NP sizes (Y)
X = data[selected_wavelengths].values
true_sizes = data["Size"].values

# -------------------------------
# Define prediction function
# -------------------------------
def predict_np_size(sample, model_form, params):
    """
    Predict Au NP size for a single sample using the optimized model.

    Args:
        sample: 1D array of wavelength absorbances
        model_form: 'poly' or 'exp'
        params: list or array of optimized parameters (2 per wavelength)

    Returns:
        Predicted NP size (float)
    """
    y_pred = 0
    for i in range(NUM_PREDICTORS):
        a = params[2*i]
        b = params[2*i+1]
        if model_form == "poly":
            y_pred += a * sample[i]**b
        else:
            y_pred += a * np.exp(-sample[i]/b)
    return y_pred

# -------------------------------
# Predict NP sizes and check < 5 nm
# -------------------------------
predicted_sizes = []
warnings = []

for i, sample in enumerate(X):
    size_pred = predict_np_size(sample, MODEL_FORM, optimized_params)
    predicted_sizes.append(size_pred)
    if size_pred < 5:
        warnings.append(f"Warning: Predicted size for sample {i} < 5 nm, outside accuracy limits")

# -------------------------------
# Save results to CSV
# -------------------------------
results_df = data[selected_wavelengths].copy()
results_df["Predicted_Size"] = predicted_sizes

results_df.to_csv(OUTPUT_FILE, index=False)
print(f"Predictions saved to {OUTPUT_FILE}")

# Print warnings if any
for w in warnings:
    print(w)