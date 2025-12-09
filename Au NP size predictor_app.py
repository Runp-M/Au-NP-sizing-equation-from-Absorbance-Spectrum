# -*- coding: utf-8 -*-
# =============================================================================
#   Au NP Size Predictor
# =============================================================================
#
#   Author:         Runpeng Miao
#   Contact:        Runpeng.miao@phd.unipd.it
#   Version:        1.0
#   Date:           September 2025
#
#   Description:
#   A graphical user interface (GUI) application that predicts the size
#   of Gold Nanoparticles (Au NPs) from their UV-Vis spectra.
#
#   This tool performs the following key functions:
#   - Reads and processes spectra files (.dat or .csv) from a user-selected folder.
#   - Interpolates each spectrum from 300-900 nm with a 1 nm step.
#   - Normalizes the spectrum based on the maximum absorbance in the 400-600 nm range.
#   - Automatically filters out samples with a Surface Plasmon Resonance (SPR)
#     peak position greater than 580 nm or with non-positive absorbance values at the SPR peak.
#   - Allows the user to select from several pre-defined analytical models to
#     predict the nanoparticle size.
#   - Saves the prediction results to a CSV file.
#
#   This script is designed to be packaged into a standalone executable.
#   
# =============================================================================

import os
import csv
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import queue

# ==================================================================================
# 1. BACKEND CALCULATION LOGIC
# ==================================================================================

# --- Model Definitions with Descriptions ---
# This dictionary holds all prediction models, their parameters, and a brief
# description of their pros and cons based on performance metrics.
ALL_MODELS = {
    "1": {
        "name": "NCC-poly", "model_form": "poly", "selected_wavelengths": ["420", "490", "468", "437"], "constant": 0.0, 
        "optimized_params": [0.392188, -6.0912, 0.084638, -8.2034, 2.154087, -0.891375, 1.822986, -0.171050],
        "description": "MAPE: 10.4%, R²: 0.9627, RMSE: 4.259, MAE: 2.71"
    },
    "2": {
        "name": "ES-poly", "model_form": "poly", "selected_wavelengths": ["390", "420", "468", "371"], "constant": 0.0, 
        "optimized_params": [1.035045, -5.84369, 107.049, 57.3078, -653.915, 22.7075, 18.4436, 7.13884],
        "description": "MAPE: 10.8%, R²: 0.9682, RMSE: 3.930, MAE: 2.72"
    },
    "3": {
        "name": "WLS-poly (single variable, 371nm)", "model_form": "wls_poly", "selected_wavelengths": ["371"], "constant": -5347.67, 
        "optimized_params": [54255.93, -216585.90, 455295.86, -552304.36, 389679.22, -148762.38, 23774.56],
        "description": "MAPE: 11.0%, R²: 0.9672, RMSE: 3.990, MAE: 2.66"
    },
    "4": {
        "name": "ES-exp", "model_form": "exp", "selected_wavelengths": ["390", "376", "344", "490"], "constant": -62.1801, 
        "optimized_params": [36.0063, 81574.96, 3486.93, 0.117347, 29.8649, 61230.41, 963.483, 0.117868],
        "description": "MAPE: 12.2%, R²: 0.9653, RMSE: 4.104, MAE: 2.92"
    },
    "5": {
        "name": "NCC-exp", "model_form": "exp", "selected_wavelengths": ["420", "490", "468", "437"], "constant": -83.5352, 
        "optimized_params": [57.8276, 65857.99, 975.172, 0.161061, 12351.33, 0.074391, 27.6582, 40084.88],
        "description": "MAPE: 12.6%, R²: 0.9393, RMSE: 5.431, MAE: 3.50"
    },
    "6": {
        "name": "WLS-exp (single variable, 490nm)", "model_form": "exp", "selected_wavelengths": ["490"], "constant": 2.77, 
        "optimized_params": [2739.26, 0.1368],
        "description": "MAPE: 14.3%, R²: 0.9410, RMSE: 5.360, MAE: 3.66"
    },
    "7": {
        "name": "Mie Theory (SPR Peak, 545-580 nm only)", "model_form": "mie_theory", "selected_wavelengths": [],
        "description": "A theoretical model based on SPR peak position, not absorbance values.\nApplicable only to a narrow size/SPR range."
    }
}

# --- Mie Theory ---
MIE_LOOKUP_TABLE = [(545, 70.0), (550, 75.0), (555, 80.0), (560, 85.0), (565, 90.0), (570, 95.0), (575, 98.0), (580, 100.0)]
mie_wl, mie_size = zip(*MIE_LOOKUP_TABLE)
mie_interp_func = interp1d(mie_wl, mie_size, kind='linear', bounds_error=True)

def mie_table_estimate(spr_peak_wl):
    """Estimates particle size from SPR peak wavelength using the Mie lookup table."""
    try:
        return float(mie_interp_func(spr_peak_wl))
    except ValueError:
        return np.nan

# --- Backend Core Function ---
def run_analysis_backend(folder_path, model_key, log_queue):
    """
    This is the core calculation engine of the application. It runs in a separate
    thread to prevent the GUI from freezing during computation. It communicates
    with the GUI via a thread-safe queue.
    """
    try:
        log_queue.put("Starting preprocessing and filtering of files...")
        model = ALL_MODELS[model_key]
        
        processed_df, analysis_data = batch_process_spectra_backend(folder_path, log_queue)
        if processed_df is None or processed_df.empty:
            log_queue.put("Error: No valid samples available after preprocessing.")
            return

        log_queue.put(f"Model selected: {model['name']}")
        
        predicted_sizes, size_warnings = [], []
        file_names = processed_df["File_Name"].values

        if model['model_form'] == 'mie_theory':
            log_queue.put("This model is only applicable for samples with an SPR peak between 545-580 nm. Filtering...")
            applicable_files, inapplicable_files_info = [], []
            for file_name in file_names:
                spr_wl = analysis_data[file_name]['spr_peak_wl']
                if 545 <= spr_wl <= 580:
                    applicable_files.append(file_name)
                else:
                    inapplicable_files_info.append(f"File: {file_name}, SPR Peak: {spr_wl:.1f} nm")
            
            if inapplicable_files_info:
                log_queue.put("\n--- Model Applicability Warning ---")
                log_queue.put("The following files cannot be processed with the Mie Theory model:")
                for info in inapplicable_files_info:
                    log_queue.put(f"  - {info}")
            if not applicable_files:
                log_queue.put("\nError: No samples fall within the 545-580 nm SPR range for Mie Theory calculation.")
                return
            
            log_queue.put(f"\nFound {len(applicable_files)} applicable samples. Starting calculation...")
            file_names = applicable_files
            for file_name in file_names:
                size_pred = mie_table_estimate(analysis_data[file_name]['spr_peak_wl'])
                predicted_sizes.append(size_pred)
                if size_pred < 5:
                    size_warnings.append(f"Warning (d < 5 nm): Predicted size for {file_name} is {size_pred:.2f} nm.")
        else:
            log_queue.put("Predicting sizes for valid samples...")
            selected_wavelengths = model["selected_wavelengths"]
            if not all(w in processed_df.columns for w in selected_wavelengths):
                log_queue.put(f"Error: Data is missing required wavelengths: {[w for w in selected_wavelengths if w not in processed_df.columns]}")
                return
            
            input_df = processed_df[selected_wavelengths]
            for i, file_name in enumerate(file_names):
                size_pred = predict_size_backend(model, input_df.iloc[i].values)
                predicted_sizes.append(size_pred)
                if size_pred < 5:
                    size_warnings.append(f"Warning (d < 5 nm): Predicted size for {file_name} is {size_pred:.2f} nm, which is outside the model's accuracy limits.")

        log_queue.put("Prediction complete.")

        results_df = pd.DataFrame({"File_Name": file_names, f"Predicted_Size_nm ({model['name']})": predicted_sizes})
        if model['model_form'] != 'mie_theory':
            mie_predictions = []
            for fname in file_names:
                spr_wl = analysis_data[fname]['spr_peak_wl']
                if 545 <= spr_wl <= 580:
                    mie_predictions.append(f"{mie_table_estimate(spr_wl):.2f} (SPR={spr_wl:.1f}nm)")
                else:
                    mie_predictions.append("N/A")
            results_df["Mie_Estimate_nm (if applicable)"] = mie_predictions

        folder_name = os.path.basename(folder_path)
        sanitized_model_name = model['name']
        for char in " ()-,:/":
            sanitized_model_name = sanitized_model_name.replace(char, '_')
        sanitized_model_name = sanitized_model_name.replace('__', '_')
        output_filename = f"Results_{folder_name}_{sanitized_model_name}.csv"
        output_file_path = os.path.join(os.path.dirname(folder_path), output_filename)
        results_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        log_queue.put(f"\nResults successfully saved to: {output_file_path}")

        if size_warnings:
            log_queue.put("\n--- Prediction Size Warnings ---")
            for w in sorted(size_warnings):
                log_queue.put(w)
        else:
            log_queue.put("\nAll predicted sizes are within the model's effective range (>= 5 nm).")
            
    except Exception as e:
        log_queue.put(f"\nAn unexpected error occurred: {e}")
    finally:
        log_queue.put("\n======= Analysis Finished =======")

def extract_spectra_from_csv(csv_file_path):
    """Extracts wavelength and absorbance data from a JASCO-style .csv file."""
    wavelengths, absorbances = [], []
    try:
        with open(csv_file_path, 'r', encoding='utf-8', errors='ignore') as infile: content = infile.readlines()
    except Exception:
        with open(csv_file_path, 'r', encoding='gbk', errors='ignore') as infile: content = infile.readlines()
    
    xydata_found = False
    for line in content:
        if not line.strip(): continue
        if 'XYDATA' in line.upper(): xydata_found = True; continue
        if xydata_found:
            try:
                parts = line.strip().replace(',', ' ').split()
                if len(parts) >= 2:
                    wavelengths.append(float(parts[0])); absorbances.append(float(parts[1]))
            except (ValueError, IndexError): continue
    if not wavelengths: raise ValueError(f"No valid data found after 'XYDATA' in {os.path.basename(csv_file_path)}")
    return np.array(wavelengths), np.array(absorbances)

def extract_spectra_from_dat(dat_file_path):
    """Extracts wavelength and absorbance data from a two-column .dat file."""
    try:
        data = np.loadtxt(dat_file_path, encoding='utf-8')
    except Exception:
        data = np.loadtxt(dat_file_path, encoding='gbk')
    if data.ndim != 2 or data.shape[1] < 2: raise ValueError(f"Incorrect format in {os.path.basename(dat_file_path)}")
    return data[:, 0], data[:, 1]

def batch_process_spectra_backend(input_folder, log_queue):
    """
    Processes all spectra files in a folder: finds files, interpolates,
    normalizes, and filters out invalid samples.
    """
    files_list_csv = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]
    files_list_dat = [f for f in os.listdir(input_folder) if f.lower().endswith('.dat')]
    
    if files_list_csv:
        files_list, file_type, extract_func = files_list_csv, 'csv', extract_spectra_from_csv
    elif files_list_dat:
        files_list, file_type, extract_func = files_list_dat, 'dat', extract_spectra_from_dat
    else:
        log_queue.put("Error: No .csv or .dat files found in the specified folder.")
        return None, {}
        
    log_queue.put(f"Found {len(files_list)} '{file_type}' files.")
    interp_wavelengths_str = np.arange(300, 901, 1).astype(str)
    interp_wavelengths_float = interp_wavelengths_str.astype(float)
    output_data, analysis_data, excluded_files_info = [], {}, []
    normalization_exclusions = []
    
    for file_name in files_list:
        file_path = os.path.join(input_folder, file_name)
        try:
            wavelengths, absorbances = extract_func(file_path)
            if len(wavelengths) < 2:
                log_queue.put(f"Warning: Insufficient data in {file_name}, skipping.")
                continue
            
            f_interp = interp1d(wavelengths, absorbances, kind='linear', bounds_error=False, fill_value=0)
            interp_abs = f_interp(interp_wavelengths_float)
            
            mask_norm = (interp_wavelengths_float >= 400) & (interp_wavelengths_float <= 600)
            max_val = np.max(interp_abs[mask_norm])
            
            if max_val <= 0:
                normalization_exclusions.append(f"File: {file_name}, Max Absorbance in 400-600nm: {max_val:.4f}")
                continue
            
            normalized_abs = interp_abs / max_val
            spr_peak_idx = np.argmax(normalized_abs)
            spr_peak_wl = interp_wavelengths_float[spr_peak_idx]
            
            if spr_peak_wl > 580:
                excluded_files_info.append(f"File: {file_name}, SPR Peak: {spr_peak_wl:.1f} nm")
                continue
            
            analysis_data[file_name] = {'spr_peak_wl': spr_peak_wl}
            output_data.append([file_name] + list(normalized_abs))
        except Exception as e:
            log_queue.put(f"Error processing {file_name}: {e}")
            continue
            
    if excluded_files_info:
        log_queue.put("\n--- File Exclusion Summary (SPR > 580 nm) ---")
        for info in excluded_files_info:
            log_queue.put(f"  - {info} (Excluded)")

    if normalization_exclusions:
        log_queue.put("\n--- File Exclusion Summary (Invalid Absorbance) ---")
        log_queue.put("The following files were excluded due to non-positive max absorbance in the 400-600nm range:")
        for info in normalization_exclusions:
            log_queue.put(f"  - {info} (Excluded)")
            
    if not output_data:
        return None, {}
        
    header = ['File_Name'] + list(interp_wavelengths_str)
    processed_df = pd.DataFrame(output_data, columns=header)
    log_queue.put(f"Preprocessing complete. {len(output_data)} valid files will be used for prediction.")
    return processed_df, analysis_data

def predict_size_backend(model, input_values):
    """A generic prediction function that handles various model forms."""
    model_form = model["model_form"]
    params = model["optimized_params"]
    constant = model.get("constant", 0.0)
    
    if model_form == "wls_poly":
        x = input_values[0]
        return constant + sum(params[i] * (x ** (i + 1)) for i in range(len(params)))
        
    y_pred = 0
    for i in range(len(input_values)):
        a, b, x = params[2 * i], params[2 * i + 1], input_values[i]
        if model_form == "poly":
            y_pred += a * (x ** b)
        elif model_form == "exp":
            y_pred += a * np.exp(-x / b)
        else:
            raise ValueError(f"Unknown model form: {model_form}")
    return y_pred + constant

# ==================================================================================
# 2. GRAPHICAL USER INTERFACE (GUI)
# ==================================================================================

class Application(tk.Frame):
    """Defines the main application window and all its widgets."""
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Gold Nanoparticle (Au NP) Size Predictor")
        self.master.geometry("750x600")
        self.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.folder_path = tk.StringVar()
        self.model_key = tk.StringVar()
        self.log_queue = queue.Queue()

        self.create_widgets()
        self.process_log_queue()

    def create_widgets(self):
        """Create and arrange all the widgets in the main window."""
        
        frame1 = ttk.LabelFrame(self, text="1. Select Data Folder")
        frame1.pack(fill="x", padx=5, pady=5)
        
        folder_label = ttk.Label(frame1, textvariable=self.folder_path, foreground="blue")
        folder_label.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.folder_path.set("No folder selected...")
        
        browse_button = ttk.Button(frame1, text="Browse...", command=self.browse_folder)
        browse_button.pack(side="right", padx=5, pady=5)

        frame2 = ttk.LabelFrame(self, text="2. Select Analysis Model")
        frame2.pack(fill="x", padx=5, pady=5)
        
        model_names = [f"{k}: {v['name']}" for k, v in ALL_MODELS.items()]
        self.model_combo = ttk.Combobox(frame2, values=model_names, state="readonly")
        self.model_combo.pack(fill="x", padx=5, pady=5)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_select)

        self.model_desc_label = ttk.Label(frame2, text="Select a model to see its description.", wraplength=700, justify="left", foreground="gray")
        self.model_desc_label.pack(fill="x", padx=5, pady=(0, 5))
        
        frame3 = ttk.LabelFrame(self, text="3. Run Analysis")
        frame3.pack(fill="x", padx=5, pady=5)
        
        self.run_button = ttk.Button(frame3, text="Start Prediction", command=self.start_analysis_thread)
        self.run_button.pack(pady=5)
        self.run_button.config(state="disabled")

        frame4 = ttk.LabelFrame(self, text="Information and Results Log")
        frame4.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(frame4, wrap=tk.WORD, width=80, height=20)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_text.config(state="disabled")

    def browse_folder(self):
        """Opens a dialog to select a directory and updates the GUI state."""
        path = filedialog.askdirectory(title="Select the folder containing spectra files")
        if path:
            self.folder_path.set(path)
            self.update_run_button_state()
            self.log_message(f"Selected folder: {path}")

    def on_model_select(self, event=None):
        """Handles the event when a user selects a model from the dropdown."""
        selection = self.model_combo.get()
        key = selection.split(':')[0]
        self.model_key.set(key)
        self.model_desc_label.config(text=ALL_MODELS[key]['description'])
        self.update_run_button_state()

    def update_run_button_state(self):
        """Enables the 'Start' button only when both a folder and a model are selected."""
        if self.folder_path.get() != "No folder selected..." and self.model_key.get():
            self.run_button.config(state="normal")
        else:
            self.run_button.config(state="disabled")

    def log_message(self, message):
        """Safely adds a message to the scrolled text log widget."""
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def start_analysis_thread(self):
        """Starts the backend analysis in a new thread to prevent the GUI from freezing."""
        self.run_button.config(state="disabled")
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")
        
        self.log_message("======= Starting Analysis =======")

        analysis_thread = threading.Thread(
            target=run_analysis_backend,
            args=(self.folder_path.get(), self.model_key.get(), self.log_queue),
            daemon=True
        )
        analysis_thread.start()

    def process_log_queue(self):
        """
        Periodically checks the message queue from the backend thread and updates
        the log display in the GUI. This method is crucial for thread-safe UI updates.
        """
        try:
            message = self.log_queue.get_nowait()
            self.log_message(message)
            if "Analysis Finished" in message:
                self.run_button.config(state="normal")
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.process_log_queue)

# ==================================================================================
# 3. APPLICATION LAUNCHER
# ==================================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()