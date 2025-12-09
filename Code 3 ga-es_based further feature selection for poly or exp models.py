# -*- coding: utf-8 -*-
"""

#!/usr/bin/env python3

General feature selection + GA parameter optimization for polynomial or exponential models.

#   Author:         Runpeng Miao
#   Contact:        Runpeng.miao@phd.unipd.it
#   Version:        1.0
#   Date:           September 2025

CSV Input Format:
- Each row is one sample.
- The first N columns are predictor variables (e.g., absorbance at different wavelengths).
- The last column is the target variable Y (e.g., particle size).
- Column headers are required. Wavelengths can be used as column names for clarity.

Workflow Overview:
1. Load CSV data and select the first 10 predictor columns (wavelength absorbances).
2. Choose the number of predictors to select (NUM_PREDICTORS, 2-9) and the model form ('poly' or 'exp').
3. Loop over all combinations of predictors of the specified size.
4. For each combination:
    a. Set up GA (NSGA-II) for optimizing model parameters.
    b. Evaluate fitness (minimize MAE, maximize R²) for polynomial or exponential models.
    c. Save the best individual for this combination.
5. Print progress: current wavelength combination and its best MAE and R².
6. After all combinations are evaluated, report the overall best combination, parameters, and scores.
7. Save all results to a CSV file.

Requirements:
- Python packages: pandas, numpy, deap, sklearn
- DEAP halloffame is used to track the best solutions during GA evolution.
"""

import subprocess
import sys

# Optionally install missing packages using subprocess
required_packages = [
    "pandas",
    "numpy",
    "scikit-learn",
    "deap"
]

for package in required_packages:
    try:
        __import__(package)  # Try to import
    except ImportError:
        print(f"{package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


import pandas as pd
import numpy as np
import itertools
import random
from deap import base, creator, tools, algorithms
from sklearn.metrics import mean_absolute_error

# -------------------------------
# User inputs
# -------------------------------
INPUT_FILE = "./Top 10 wavelengths of Abs spectra.csv"  # Replace with your CSV path
NUM_PREDICTORS = 2            # Choose number of predictors to select (2-9)
MODEL_FORM = 'poly'           # 'poly' or 'exp'
POPSIZE = 500                 # GA initial population
NGEN = 25                     # Number of generation in GA

# -------------------------------
# Load data (only first 10 wavelength columns)
# -------------------------------
data = pd.read_csv(INPUT_FILE)
wavelengths = data.columns[1:11]  # Top 10 wavelength columns'
X_all = data[wavelengths].values.astype(float)
Y = data['Size'].values.astype(float)

n_features = X_all.shape[1]
if NUM_PREDICTORS < 2 or NUM_PREDICTORS > n_features:
    raise ValueError(f"NUM_PREDICTORS must be between 2 and {n_features}")


# -------------------------------
# Model definitions
# -------------------------------
def polynomial_model(X, *params):
    """Multi-feature polynomial model: a*x**b per predictor"""
    n_features = X.shape[1]
    y_pred = np.zeros(X.shape[0])
    for i in range(n_features):
        a = params[2*i]
        b = params[2*i+1]
        y_pred += a * X[:, i]**b
    return y_pred

def exponential_model(X, *params):
    """Multi-feature exponential decay model: a*exp(-x/b) per predictor"""
    n_features = X.shape[1]
    y_pred = np.zeros(X.shape[0])
    def safe_exp(z):
        z = np.clip(z, -700, 700)
        return np.exp(z)
    for i in range(n_features):
        a = params[2*i]
        b = params[2*i+1]
        if a < 0 or b <= 1e-6:
            return np.ones(X.shape[0])*1e10  # Penalize invalid params
        y_pred += a * safe_exp(-X[:, i]/b)
    return y_pred

# -------------------------------
# Fitness function
# -------------------------------
def fitness_multi_objective(individual, X, Y, model_type='poly'):
    try:
        if model_type == 'poly':
            y_pred = polynomial_model(X, *individual)
        elif model_type == 'exp':
            y_pred = exponential_model(X, *individual)
        else:
            raise ValueError("model_type must be 'poly' or 'exp'")
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)) or np.any(y_pred <= 0):
            return 1e6, -1e6
        residuals = Y - y_pred
        mae = np.mean(np.abs(residuals))
        r2 = 1 - (np.sum(residuals**2) / np.sum((Y - np.mean(Y))**2))
        return mae, r2
    except Exception:
        return 1e6, -1e6

# -------------------------------
# DEAP setup
# -------------------------------
# Delete previous classes if exist
if "FitnessMulti" in creator.__dict__:
    del creator.FitnessMulti
if "Individual" in creator.__dict__:
    del creator.Individual

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", lambda: random.uniform(-100, 100))  # GA parameter range
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2*NUM_PREDICTORS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# -------------------------------
# Loop over predictor combinations
# -------------------------------
best_results = []
total_combinations = len(list(itertools.combinations(range(n_features), NUM_PREDICTORS)))
combo_counter = 0

for cols in itertools.combinations(range(n_features), NUM_PREDICTORS):
    combo_counter += 1
    X_sub = X_all[:, cols]
    selected_wavelengths = wavelengths[list(cols)]

    def eval_individual(individual):
        return fitness_multi_objective(individual, X_sub, Y, MODEL_FORM)

    toolbox.register("evaluate", eval_individual)

    pop = toolbox.population(n=POPSIZE)
    hof = tools.ParetoFront()
    algorithms.eaMuPlusLambda(pop, toolbox, mu=500, lambda_=1000,
                              cxpb=0.7, mutpb=0.3, ngen=NGEN, stats=None,
                              halloffame=hof, verbose=False)

    if len(hof) == 0:
        continue
    best_ind = sorted(hof, key=lambda ind: (ind.fitness.values[0], -ind.fitness.values[1]))[0]
    mae, r2 = best_ind.fitness.values
    best_results.append({
        "wavelengths": list(selected_wavelengths),
        "params": list(best_ind),
        "MAE": mae,
        "R2": r2
    })

    # --- Print progress ---
    print(f"[{combo_counter}/{total_combinations}] Current combination: {list(selected_wavelengths)}, "
          f"Best MAE: {mae:.4f}, Best R2: {r2:.4f}")

# -------------------------------
# Report final best
# -------------------------------
if len(best_results) == 0:
    print("No valid GA results found. Check data or parameter ranges.")
else:
    best_overall = sorted(best_results, key=lambda r: (r["MAE"], -r["R2"]))[0]
    print("\nBest wavelength combination found:")
    print("Wavelengths:", best_overall["wavelengths"])
    print("MAE:", best_overall["MAE"])
    print("R2:", best_overall["R2"])
    print("Parameters:", best_overall["params"])

    pd.DataFrame(best_results).to_csv("best_results.csv", index=False)
    print("\nAll results saved to best_results.csv")
