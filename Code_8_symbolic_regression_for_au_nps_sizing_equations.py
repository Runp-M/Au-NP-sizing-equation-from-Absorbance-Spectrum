"""
Two-stage pipeline for Au nanoparticle size estimation from UV–Vis spectra.

Stage 1 – Feature (wavelength) subset selection
    • Multi-objective optimization with NSGA-III
    • Objective 1: prediction error (RMSE) using an ensemble regressor
    • Objective 2: (here fixed) number of selected wavelengths

Stage 2 – Symbolic regression
    • Multi-objective symbolic regression with NSGA-III
    • Objectives: RMSE, MAE, and (1 – R²)
    • Produces closed-form expressions linking absorbance at selected wavelengths
      to nanoparticle size.

Input CSV format
----------------
The script expects a CSV file with columns:

    390, 420, 376, 361, 437, 344, 394, 468, 490, 371, Size, Std

i.e.:
    • First 10 columns: absorbance at 10 wavelengths (column names are wavelengths in nm)
    • Column 'Size': target particle size (nm)
    • Column 'Std': optional, not used by the models

Dependencies
------------
numpy, pandas, scipy, sympy, scikit-learn, xgboost, catboost, pymoo, tqdm

Example usage
-------------
python au_size_pipeline.py --data Top_10_wavelengths_35_water_spectra.csv
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import warnings
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import least_squares
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    VotingRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from pymoo.core.problem import Problem
from pymoo.core.termination import Termination
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.core.callback import Callback

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Stage 1 (feature selection) parameters
N_SELECTED_WAVELENGTHS = 5  # number of wavelengths to select
STAGE1_POP_SIZE = 50

# Stage 2 (symbolic regression) parameters
GENOME_LENGTH = 60          # length of integer genome encoding the expression tree
N_CONSTS = 3                # number of optimizable constants in each expression
MAX_TREE_DEPTH = 6
MAX_NODES = 50              # maximum allowed nodes per expression (complexity control)
POPUP_REPLACE_MOD = 1_000_000
EVALUATION_TIMEOUT = 15     # seconds per-individual evaluation (safety timeout)


# The number of input features for Stage 2.
# Here we assume it equals N_SELECTED_WAVELENGTHS; change if needed.
N_INPUT_FEATURES = N_SELECTED_WAVELENGTHS
TERMINALS = [f"A{i+1}" for i in range(N_INPUT_FEATURES)] + ["CONST"]

BINOPS = ["+", "-", "*", "/"]
UNOPS = ["sin", "cos", "exp", "log", "sqrt", "square", "tanh", "asin", "acos"]
EPS = 1e-9


# ---------------------------------------------------------------------------
# Utility: dataset loading
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    X_full : np.ndarray
        (n_samples, n_features) array of absorbance data (first 10 columns).
    y : np.ndarray
        (n_samples,) array of particle sizes (column 'Size').
    wavelengths : np.ndarray
        (n_features,) array of wavelength values (column names of the first 10 columns).
    df : pd.DataFrame
        Original dataframe (for optional further use).
    """
    df = pd.read_csv(csv_path)

    # First 10 columns are absorbance spectra (wavelengths in nm)
    X_full = df.iloc[:, :10].values.astype(float)

    # Target: particle size (nm)
    if "Size" not in df.columns:
        raise ValueError("Input CSV must contain a 'Size' column.")
    y = df["Size"].values.astype(float)

    # Wavelengths in nm taken from the column names of the first 10 columns
    wavelengths = df.columns[:10].astype(int).values

    return X_full, y, wavelengths, df


# ---------------------------------------------------------------------------
# Stage 1: Feature subset (wavelength) selection
# ---------------------------------------------------------------------------

class AdaptiveTermination(Termination):
    """
    Custom termination for Stage 1.

    Stops the optimization if the best objective (RMSE) has not improved
    for a given number of generations (patience).
    """

    def __init__(self, patience: int = 100):
        super().__init__()
        self.patience = patience
        self.best_so_far = np.inf
        self.counter = 0

    def _update(self, algorithm) -> float:
        current_best_rmse = np.min(algorithm.pop.get("F")[:, 0])

        if current_best_rmse < self.best_so_far - 1e-6:
            self.best_so_far = current_best_rmse
            self.counter = 0
        else:
            self.counter += 1

        # Pymoo stops when this "progress" reaches 1.0
        progress = self.counter / self.patience
        return progress


class FeatureSubsetProblem(Problem):
    """
    Multi-objective wavelength subset selection problem.

    Decision variables:
        - Integer indices of wavelengths (length = n_features_to_select)

    Objectives:
        - f1: prediction RMSE on a hold-out set using an ensemble regressor
        - f2: number of selected features (here constant, but kept explicit)
    """

    def __init__(self, X_full: np.ndarray, y: np.ndarray, n_features_to_select: int):
        super().__init__(
            n_var=n_features_to_select,
            n_obj=2,
            xl=0,
            xu=X_full.shape[1] - 1,
            type_var=int,
        )
        self.X_full = X_full
        self.y = y

    def _evaluate(self, X_pop: np.ndarray, out: Dict[str, Any], *args, **kwargs) -> None:
        pop_size = X_pop.shape[0]
        f1 = np.zeros(pop_size)
        f2 = np.zeros(pop_size)

        for i in range(pop_size):
            subset_idx = X_pop[i].astype(int)

            # Enforce uniqueness of selected wavelengths
            if len(np.unique(subset_idx)) < len(subset_idx):
                # Penalize individuals with duplicate indices
                f1[i], f2[i] = 1e6, self.n_var
                continue

            X_sub = self.X_full[:, subset_idx]

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_sub, self.y, test_size=0.25, random_state=42
                )

                # Base regressors
                xgb_main = XGBRegressor(
                    learning_rate=0.8325,
                    max_depth=12,
                    n_estimators=1182,
                    gamma=0,
                    min_child_weight=0,
                    subsample=0.8171,
                    colsample_bytree=0.3297,
                    reg_alpha=10,
                    reg_lambda=47,
                    seed=1174,
                    n_jobs=-1,
                    verbosity=0,
                )

                ada = AdaBoostRegressor(
                    estimator=XGBRegressor(max_depth=5, n_estimators=50, verbosity=0),
                    random_state=15,
                )

                gbr = GradientBoostingRegressor(
                    n_estimators=280,
                    learning_rate=0.2978,
                    max_depth=8,
                    max_features="sqrt",
                    loss="quantile",
                    subsample=0.3282,
                    random_state=15,
                )

                cat = CatBoostRegressor(
                    iterations=345,
                    learning_rate=0.04223,
                    depth=5,
                    l2_leaf_reg=7,
                    random_strength=7.478,
                    verbose=0,
                )

                voter = VotingRegressor(
                    estimators=[
                        ("XgBoost", xgb_main),
                        ("GradientBoost", gbr),
                        ("CatBoost", cat),
                        ("AdaBoost", ada),
                    ],
                    n_jobs=-1,
                    weights=[1, 0.95, 1, 0.95],
                )

                voter.fit(X_train, y_train)
                y_pred = voter.predict(X_test)
                rmse = math.sqrt(mean_squared_error(y_test, y_pred))

            except Exception as e:
                # If anything goes wrong, strongly penalize this individual
                print(f"[Stage 1] Error during model evaluation: {e}")
                rmse = 1e6

            f1[i] = rmse
            f2[i] = len(subset_idx)

        out["F"] = np.column_stack([f1, f2])


def run_stage1_feature_selection(
    X_full: np.ndarray,
    y: np.ndarray,
    pop_size: int = STAGE1_POP_SIZE,
    n_features_to_select: int = N_SELECTED_WAVELENGTHS,
    patience: int = 100,
    random_seed: int = 1,
) -> np.ndarray:
    """
    Run Stage 1: NSGA-III-based wavelength subset selection.

    Returns
    -------
    best_subset : np.ndarray
        Indices (with respect to the original X_full columns) of the
        selected wavelengths.
    """
    print("\n=== Stage 1: Wavelength subset selection (NSGA-III) ===")

    problem = FeatureSubsetProblem(X_full, y, n_features_to_select=n_features_to_select)
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
    )

    termination = AdaptiveTermination(patience=patience)

    res = minimize(problem, algorithm, termination, seed=random_seed, verbose=True)

    # Select individual with the smallest RMSE in the final population
    best_solution_idx = np.argmin(res.F[:, 0])
    best_subset = np.unique(res.X[best_solution_idx]).astype(int)

    print(f"Stage 1: Best wavelength indices selected: {best_subset}")
    return best_subset


# ---------------------------------------------------------------------------
# Stage 2: Symbolic regression (NSGA-III)
# ---------------------------------------------------------------------------

class ProgressBarCallback(Callback):
    """
    Tqdm progress bar callback for pymoo.
    """

    def __init__(self, n_max_gen: int):
        super().__init__()
        self.n_max_gen = n_max_gen
        self.progress_bar: tqdm | None = None

    def notify(self, algorithm):
        if self.progress_bar is None:
            self.progress_bar = tqdm(
                total=self.n_max_gen,
                desc="Stage 2 – evolving equations",
                unit="gen",
            )
        self.progress_bar.update(1)
        best_rmse = np.min(algorithm.pop.get("F")[:, 0])
        self.progress_bar.set_postfix(best_rmse=f"{best_rmse:.4f}")


def expand_term(genome: np.ndarray, ptr: int, node_counter: Dict[str, int]):
    """
    Decode a terminal node from the integer genome.
    """
    node_counter["count"] += 1
    if node_counter["count"] > MAX_NODES:
        raise ValueError("Expression complexity limit reached.")

    safe_ptr = ptr % GENOME_LENGTH
    idx = genome[safe_ptr] % len(TERMINALS)
    tok = TERMINALS[idx]

    if tok == "CONST":
        const_id = (genome[safe_ptr] // len(TERMINALS)) % N_CONSTS
        return sp.Symbol(f"CONST{const_id}")
    else:
        return sp.Symbol(tok)


def expand_expr(genome: np.ndarray, ptr: int, depth: int, node_counter: Dict[str, int]):
    """
    Recursively decode an expression tree from the integer genome.
    """
    node_counter["count"] += 1
    if depth >= MAX_TREE_DEPTH or node_counter["count"] > MAX_NODES:
        return expand_term(genome, ptr, node_counter), ptr + 1

    safe_ptr = ptr % GENOME_LENGTH
    choice = genome[safe_ptr] % 3
    ptr += 1

    if choice == 0:
        # Binary operator
        left, ptr = expand_expr(genome, ptr, depth + 1, node_counter)
        op = BINOPS[genome[ptr % GENOME_LENGTH] % len(BINOPS)]
        ptr += 1
        right, ptr = expand_expr(genome, ptr, depth + 1, node_counter)

        if op == "+":
            return left + right, ptr
        if op == "-":
            return left - right, ptr
        if op == "*":
            return left * right, ptr
        if op == "/":
            return left / (right + sp.Float(EPS)), ptr

    elif choice == 1:
        # Unary operator
        op = UNOPS[genome[ptr % GENOME_LENGTH] % len(UNOPS)]
        ptr += 1
        operand, ptr = expand_expr(genome, ptr, depth + 1, node_counter)

        if op == "sin":
            return sp.sin(operand), ptr
        if op == "cos":
            return sp.cos(operand), ptr
        if op == "exp":
            return sp.exp(operand), ptr
        if op == "log":
            return sp.log(sp.Abs(operand) + sp.Float(EPS)), ptr
        if op == "sqrt":
            return sp.sqrt(sp.Abs(operand)), ptr
        if op == "square":
            return operand ** 2, ptr
        if op == "tanh":
            return sp.tanh(operand), ptr
        if op == "asin":
            safe_operand = sp.Max(-1.0, sp.Min(1.0, operand))
            return sp.asin(safe_operand), ptr
        if op == "acos":
            safe_operand = sp.Max(-1.0, sp.Min(1.0, operand))
            return sp.acos(safe_operand), ptr

    # Terminal fallback
    term = expand_term(genome, ptr, node_counter)
    ptr += 1
    return term, ptr


def decode_genome_to_expr_robust(genome: np.ndarray):
    """
    Decode an integer genome to a SymPy expression.

    Any failure or excessive complexity triggers a fallback to a simple A1 expression.
    """
    genome = np.asarray(genome, dtype=int)
    node_counter = {"count": 0}
    try:
        expr, _ = expand_expr(genome, 0, 0, node_counter)
    except ValueError:
        expr = sp.Symbol("A1")
    return expr, node_counter["count"]


def expr_to_callable(expr: sp.Expr):
    """
    Convert a SymPy expression to a NumPy-callable function.

    Inputs correspond to:
        A1, A2, ..., A_N_INPUT_FEATURES, CONST0, CONST1, ..., CONST(N_CONSTS-1)
    """

    a_syms = sp.symbols(" ".join([f"A{i+1}" for i in range(N_INPUT_FEATURES)]))
    c_syms = sp.symbols(" ".join([f"CONST{i}" for i in range(N_CONSTS)]))
    all_syms = a_syms + c_syms

    f = sp.lambdify(all_syms, expr, "numpy")

    def wrapper(X: np.ndarray, consts: np.ndarray):
        # X has shape (n_samples, N_INPUT_FEATURES)
        args = [X[:, i] for i in range(X.shape[1])] + list(consts)
        return f(*args)

    return wrapper


def fit_constants(expr: sp.Expr, X: np.ndarray, y: np.ndarray):
    """
    Fit the N_CONSTS constants in the symbolic expression using nonlinear least squares.
    """

    if N_CONSTS == 0:
        return [], expr

    func = expr_to_callable(expr)
    const_init = np.ones(N_CONSTS) * 0.1

    def residuals(consts):
        try:
            y_pred = func(X, consts)
            if np.isscalar(y_pred):
                y_pred = np.full_like(y, y_pred)
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return np.full_like(y, 1e6)
            return np.clip(y_pred, -1e6, 1e6) - y
        except Exception:
            return np.full_like(y, 1e6)

    try:
        res = least_squares(residuals, const_init, max_nfev=2000)
        c_opt = res.x
    except Exception:
        c_opt = const_init

    expr_sub = expr
    for i, v in enumerate(c_opt):
        expr_sub = expr_sub.subs({sp.Symbol(f"CONST{i}"): sp.Float(v)})

    return c_opt, expr_sub


def single_individual_evaluation_task(
    genome: np.ndarray, X: np.ndarray, y: np.ndarray, result_queue: mp.Queue
) -> None:
    """
    Evaluation of a single individual in a separate process (for timeout safety).

    Computes RMSE, MAE, and R² for the decoded expression after constant fitting.
    """
    try:
        expr, _ = decode_genome_to_expr_robust(genome)
        c_opt, expr_fitted = fit_constants(expr, X, y)
        func = expr_to_callable(expr_fitted)
        y_pred = func(X, c_opt)

        if np.isscalar(y_pred):
            y_pred = np.full_like(y, y_pred)

        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            raise ValueError("Unstable prediction")

        y_pred = np.maximum(0.0, y_pred)

        rmse = math.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        result_queue.put({"rmse": rmse, "mae": mae, "r2": r2})

    except Exception:
        result_queue.put({"rmse": 1e6, "mae": 1e6, "r2": -1.0})


class SymbolicRegressionProblem(Problem):
    """
    Multi-objective symbolic regression problem.

    Decision variables:
        - Integer genome of length GENOME_LENGTH

    Objectives:
        - f1: RMSE
        - f2: MAE
        - f3: 1 - R²  (so minimizing f3 corresponds to maximizing R²)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__(
            n_var=GENOME_LENGTH,
            n_obj=3,
            xl=0,
            xu=POPUP_REPLACE_MOD - 1,
            type_var=int,
        )
        self.X = X.astype(float)
        self.y = y.astype(float)

    def _evaluate(self, X_pop: np.ndarray, out: Dict[str, Any], *args, **kwargs) -> None:
        f1 = np.zeros(X_pop.shape[0])  # RMSE
        f2 = np.zeros(X_pop.shape[0])  # MAE
        f3 = np.zeros(X_pop.shape[0])  # 1 - R²

        for i, genome in enumerate(X_pop):
            result_queue: mp.Queue = mp.Queue()
            process = mp.Process(
                target=single_individual_evaluation_task,
                args=(genome, self.X, self.y, result_queue),
            )
            process.start()
            process.join(timeout=EVALUATION_TIMEOUT)

            if process.is_alive():
                process.terminate()
                process.join()
                result = {"rmse": 2e6, "mae": 2e6, "r2": -1.0}
            else:
                try:
                    result = result_queue.get_nowait()
                except mp.queues.Empty:
                    result = {"rmse": 1e6, "mae": 1e6, "r2": -1.0}

            f1[i] = result["rmse"]
            f2[i] = result["mae"]
            f3[i] = 1.0 - result["r2"]

        out["F"] = np.column_stack([f1, f2, f3])


def run_stage2_symbolic_regression(
    X: np.ndarray,
    y: np.ndarray,
    ngen: int = 100,
    pop_size: int = 150,
    random_seed: int = 1,
) -> pd.DataFrame:
    """
    Run Stage 2: NSGA-III-based symbolic regression on selected features.

    Returns
    -------
    df_solutions : pd.DataFrame
        DataFrame containing all discovered equations and their performance
        metrics, sorted by RMSE (ascending).
    """
    print("\n=== Stage 2: Symbolic regression (NSGA-III) ===")

    problem = SymbolicRegressionProblem(X, y)
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        callback=ProgressBarCallback(n_max_gen=ngen),
    )

    termination = DefaultMultiObjectiveTermination(n_max_gen=ngen)

    res = minimize(problem, algorithm, termination, seed=random_seed, verbose=False)

    print("\nStage 2: Analyzing final Pareto front...")

    rows: List[Dict[str, Any]] = []
    pred_upper_bound = np.max(y) * 10.0

    for ind in res.pop:
        try:
            expr, _ = decode_genome_to_expr_robust(ind.X)
            c_opt, expr_fitted = fit_constants(expr, X, y)

            try:
                expr_final = sp.simplify(expr_fitted)
            except Exception:
                expr_final = expr_fitted

            func = expr_to_callable(expr_final)
            y_pred = func(X, c_opt)
            if np.isscalar(y_pred):
                y_pred = np.full_like(y, y_pred)

            y_pred = np.maximum(0.0, np.clip(y_pred, 0.0, pred_upper_bound))

            rmse = math.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            mape = mean_absolute_percentage_error(y, y_pred)

            # A simple complexity estimate: count functions and internal nodes
            complexity = sum(
                1
                for node in sp.preorder_traversal(expr_final)
                if node.is_Function
                or (
                    node.is_Symbol
                    and "CONST" not in node.name
                    and "A" not in node.name
                )
            )

            rows.append(
                {
                    "expr": str(expr_final),
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "r2": r2,
                    "complexity": complexity,
                    "y_pred": list(y_pred),
                }
            )
        except Exception:
            continue

    df_solutions = pd.DataFrame(rows).sort_values(by="rmse").reset_index(drop=True)
    df_solutions.to_csv("symbolic_regression_solutions.csv", index=False)
    print("Stage 2: All Pareto solutions saved to 'symbolic_regression_solutions.csv'.")

    if not df_solutions.empty:
        best = df_solutions.iloc[0]
        print("\n=== Best equation found ===")
        print("Equation:", best["expr"])
        print("RMSE    :", best["rmse"])
        print("MAE     :", best["mae"])
        print("MAPE    :", best["mape"])
        print("R²      :", best["r2"])
        print("Complexity:", best["complexity"])

    return df_solutions


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage Au NP size regression from UV–Vis spectra."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV file containing absorbance spectra and 'Size' column.",
    )
    parser.add_argument(
        "--stage2_ngen",
        type=int,
        default=100,
        help="Number of generations for Stage 2 symbolic regression.",
    )
    parser.add_argument(
        "--stage2_pop",
        type=int,
        default=150,
        help="Population size for Stage 2 symbolic regression.",
    )
    args = parser.parse_args()

    # Load dataset
    X_full, y, wavelengths, df = load_dataset(args.data)
    print(f"Loaded dataset from {args.data} with shape {df.shape}")

    # Stage 1: wavelength selection
    best_subset_indices = run_stage1_feature_selection(
        X_full,
        y,
        pop_size=STAGE1_POP_SIZE,
        n_features_to_select=N_SELECTED_WAVELENGTHS,
        patience=10,
        random_seed=1,
    )
    selected_wavelengths_nm = wavelengths[best_subset_indices]
    print("Selected wavelengths (nm):", selected_wavelengths_nm)

    X_selected = X_full[:, best_subset_indices]
    print("Stage 2 input matrix shape:", X_selected.shape)

    # Stage 2: symbolic regression on the selected features
    solutions_df = run_stage2_symbolic_regression(
        X_selected,
        y,
        ngen=args.stage2_ngen,
        pop_size=args.stage2_pop,
        random_seed=1,
    )

    if not solutions_df.empty:
        best = solutions_df.iloc[0]
        print("\n" + "=" * 70)
        print("Final best Au nanoparticle size estimation equation")
        print("=" * 70)
        print(f"Equation:\n  {best['expr']}")
        print("\nPerformance:")
        print(f"  RMSE       : {best['rmse']:.4f}")
        print(f"  MAE        : {best['mae']:.4f}")
        print(f"  R-squared  : {best['r2']:.4f}")
        print(f"  Complexity : {best['complexity']}")
        print(
            "\nVariables A1–A5 correspond to the absorbance at the selected "
            f"wavelengths {selected_wavelengths_nm} nm, respectively."
        )
        print("=" * 70)
    else:
        print("\nSymbolic regression finished, but no valid solutions were found.")


if __name__ == "__main__":
    # For multiprocessing safety on Windows
    mp.freeze_support()
    main()
