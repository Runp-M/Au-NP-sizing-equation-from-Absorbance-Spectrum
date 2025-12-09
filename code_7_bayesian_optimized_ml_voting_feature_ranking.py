# -*- coding: utf-8 -*-
"""

#!/usr/bin/env python3

This script performs wavelength selection for Au nanoparticle (Au NP) size prediction
from UV-Vis absorbance spectra using Bayesian Optimization and ensemble learning.

#   Author:         Runpeng Miao
#   Contact:        Runpeng.miao@phd.unipd.it
#   Version:        1.0
#   Date:           September 2025

---------------------------------------------------------------------------------
Input file format requirements:
    - CSV file containing three main types of columns (from code 1 - preprocessed dataset with corresponding Au NP mean size):
        1. Normalized wvelength absorbance features (e.g., 300 nm – 900 nm, one column per wavelength)
        2. Absorbance values corresponding to each wavelength
        3. Target column: 'Size' (nanoparticle size in nm, located in the last column)
    - Example column structure:
        Wavelength_300, Wavelength_301, ..., Wavelength_900, Size

---------------------------------------------------------------------------------
Code functionality:
    1. Load and preprocess spectral dataset.
    2. Use Bayesian Optimization to tune hyperparameters for four ML models:
        - XGBoost
        - Gradient Boosting Regressor
        - CatBoost Regressor
        - AdaBoost (with XGBoost as base estimator)
    3. Train each model with optimized hyperparameters.
    4. Ensemble the four optimized models using a Voting Regressor.
    5. Compute and average feature importance across the models.
    6. Output the Top-10 influential wavelengths for Au NP size prediction.
---------------------------------------------------------------------------------
"""

import subprocess
import sys

# --- List of Required Packages ---
# Format: (name_for_import, name_for_pip_install)
packages_to_install = [
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'seaborn'),
    ('plotly', 'plotly'),
    ('sklearn', 'scikit-learn'),  # Special case where names differ
    ('xgboost', 'xgboost'),
    ('scipy', 'scipy'),
    ('bayesian-optimization', 'bayesian-optimization'),
    ('catboost', 'catboost'),
]

print("Checking for required packages...")

# Loop through the list and install if missing
for import_name, install_name in packages_to_install:
    try:
        # Try to import the package
        __import__(import_name)
    except ImportError:
        # If import fails, install the package
        print(f"-> Installing {install_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
        except Exception as e:
            print(f"   [ERROR] Could not install {install_name}. Please install it manually.")
            print(f"   Command: pip install {install_name}")

print("\nPackage check complete. All required libraries should now be installed.")

# =============================
# Import required libraries
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from bayes_opt import BayesianOptimization

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

# =============================
# Load dataset
# =============================
df = pd.read_csv("NP_spectra_normalized_interpolated.csv") # The file path of normalized spectra dataset with NP mean size

# Define target variable (Au NP size)
y = df['Size'] if 'Size' in df.columns else df.iloc[:, -1]
y = y.reset_index(drop=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# =============================
# Define Bayesian Optimization functions
# =============================
def optimize_xgb(learning_rate, max_depth, n_estimators, gamma, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda, seed):
    val = cross_val_score(
        XGBRegressor(
            learning_rate=min(learning_rate, 0.99), max_depth=int(max_depth),
            n_estimators=int(n_estimators), gamma=min(gamma, 0.99),
            min_child_weight=int(min_child_weight), subsample=min(subsample, 1),
            colsample_bytree=min(colsample_bytree, 1),
            reg_alpha=int(reg_alpha), reg_lambda=int(reg_lambda), seed=int(seed)
        ), X_train, y_train, scoring='r2', cv=5, n_jobs=-1
    ).mean()
    return val

def optimize_gbr(n_estimators, learning_rate, max_depth, subsample, random_state):
    val = cross_val_score(
        GradientBoostingRegressor(
            n_estimators=int(n_estimators), learning_rate=learning_rate,
            max_depth=int(max_depth), subsample=subsample,
            random_state=int(random_state)
        ), X_train, y_train, scoring='r2', cv=5, n_jobs=-1
    ).mean()
    return val

def optimize_cat(iterations, learning_rate, depth, l2_leaf_reg, random_strength):
    val = cross_val_score(
        CatBoostRegressor(
            iterations=int(iterations), learning_rate=min(learning_rate, 0.99),
            depth=int(depth), l2_leaf_reg=int(l2_leaf_reg),
            random_strength=min(random_strength, 10),
            loss_function='RMSE', verbose=0
        ), X_train, y_train, scoring='r2', cv=5, n_jobs=-1
    ).mean()
    return val

# AdaBoost with XGBoost as base
def optimize_ada(n_estimators, learning_rate, random_state):
    base_est = XGBRegressor()
    val = cross_val_score(
        AdaBoostRegressor(
            base_estimator=base_est, n_estimators=int(n_estimators),
            learning_rate=min(learning_rate, 2), random_state=int(random_state)
        ), X_train, y_train, scoring='r2', cv=5, n_jobs=-1
    ).mean()
    return val

# =============================
# Define parameter ranges
# =============================
pbounds = {
    'xgb': {
        'learning_rate': (0.001, 0.999), 'max_depth': (3, 30),
        'n_estimators': (200, 1500), 'gamma': (0, 1),
        'min_child_weight': (0, 10), 'subsample': (0.1, 1),
        'colsample_bytree': (0.1, 1), 'reg_alpha': (0, 200),
        'reg_lambda': (0, 200), 'seed': (0, 2023)
    },
    'gbr': {
        'n_estimators': (200, 900), 'learning_rate': (0.01, 0.5),
        'max_depth': (3, 15), 'subsample': (0.3, 1),
        'random_state': (0, 100)
    },
    'cat': {
        'iterations': (200, 900), 'learning_rate': (0.01, 0.5),
        'depth': (3, 15), 'l2_leaf_reg': (2, 10),
        'random_strength': (0, 10)
    },
    'ada': {
        'n_estimators': (200, 1000), 'learning_rate': (0.01, 1.2),
        'random_state': (0, 200)
    }
}

models = {
    'xgb': optimize_xgb,
    'gbr': optimize_gbr,
    'cat': optimize_cat,
    'ada': optimize_ada
}

# =============================
# Bayesian Optimization wrapper
# =============================
def optimize_model(model_name, init_points=10, n_iter=50):
    optimizer = BayesianOptimization(
        f=models[model_name], pbounds=pbounds[model_name], random_state=42
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    return optimizer.max

# =============================
# Run optimization
# =============================
xgb_opt = optimize_model('xgb')
gbr_opt = optimize_model('gbr')
cat_opt = optimize_model('cat')
ada_opt = optimize_model('ada')

print("\nOptimized Results:")
print("XGBoost:", xgb_opt)
print("GradientBoost:", gbr_opt)
print("CatBoost:", cat_opt)
print("AdaBoost:", ada_opt)

# =============================
# Build models with best params
# =============================
best_xgb = XGBRegressor(**xgb_opt['params'])
best_gbr = GradientBoostingRegressor(**gbr_opt['params'])
best_cat = CatBoostRegressor(**cat_opt['params'], verbose=0)
best_ada = AdaBoostRegressor(base_estimator=XGBRegressor(), **ada_opt['params'])

# Fit models
for model in [best_xgb, best_gbr, best_cat, best_ada]:
    model.fit(X_train, y_train)

# =============================
# Voting Regressor (Ensemble)
# =============================
vtres = VotingRegressor(
    estimators=[
        ('XGBoost', best_xgb),
        ('GradientBoost', best_gbr),
        ('CatBoost', best_cat),
        ('AdaBoost', best_ada)
    ],
    n_jobs=-1, weights=(1, 0.95, 1, 0.95), verbose=1
)

vtres.fit(X_train, y_train)

# Evaluate performance
print(f"Voting Regressor Train R²: {vtres.score(X_train, y_train):.3f}")
print(f"Voting Regressor Test R²: {vtres.score(X_test, y_test):.3f}")

# =============================
# Feature Importance Analysis
# =============================
all_importances = []
for estimator in vtres.estimators_:
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        all_importances.append(importances)

all_importances = np.vstack(all_importances)
mean_importances = np.mean(all_importances, axis=0)

# Sort and select Top-10 influential wavelengths
sort_idx = np.argsort(mean_importances)[::-1]
top_features = X.columns[sort_idx][:10]
top_importances = mean_importances[sort_idx][:10]

# =============================
# Visualization
# =============================
plt.figure(figsize=(6, 5))
plt.barh(top_features[::-1], top_importances[::-1], color='red')
plt.xlabel("Mean Feature Importance")
plt.ylabel("Wavelengths (nm)")
plt.title("Top-10 Influential Wavelengths for Au NP Size Prediction")
plt.show()

# Save importance table
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean Importance": mean_importances
}).sort_values("Mean Importance", ascending=False)

print("\nTop-10 influential wavelengths:")
print(importance_df.head(10))
importance_df.to_csv("Top10_Wavelengths.csv", index=False)