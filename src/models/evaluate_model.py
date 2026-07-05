"""Model evaluation pipeline for the retail sales forecasting project.

This module loads the model persisted by ``src.models.train_model``
(``best_model.joblib``), regenerates the same chronological validation
split used at training time, scores the model, and produces diagnostic
plots and reports. It deliberately does NOT retrain anything and does
NOT redefine metric or split logic that already exists in
``train_model.py`` -- those are imported and reused so the evaluation
numbers are guaranteed to be computed the same way training-time
validation numbers were.

Typical usage::

    python -m src.models.evaluate_model

Or programmatically::

    from src.models.evaluate_model import run_evaluation_pipeline
    metrics = run_evaluation_pipeline()

Do NOT modify:
    - src/config.py
    - src/data/load_data.py
    - src/data/merge_data.py
    - src/data/clean_data.py
    - src/features/build_features.py
    - src/models/train_model.py
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib

# Use a non-interactive backend: this module may run in headless
# environments (CI, servers) with no display, and an interactive
# backend would raise or hang when saving figures.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------- #
# Project imports.
#
# Reuse -- rather than reimplement -- the data loading, validation
# splitting, and metric computation already defined in train_model.py.
# This guarantees the validation set and metric formulas used here are
# identical to the ones used during training, and avoids maintaining two
# copies of the same logic.
# --------------------------------------------------------------------------- #
from src import config  # TODO: Match this import path to your package layout if different
from src.models.train_model import (
    MODEL_DIR,
    STORE_COLUMN,
    TARGET_COLUMN,
    DATE_COLUMN,
    SplitData,
    build_split_data,
    compute_regression_metrics,
    configure_logging,
    load_feature_dataset,
    sort_by_store_and_date,
    time_based_train_val_split,
    validate_required_columns,
)

try:
    FIGURES_DIR: Path = config.FIGURES_DIR  # TODO: Match this constant name to config.py
except AttributeError:
    # Fall back to a directory alongside MODEL_DIR so the module stays
    # importable even before config.py defines a dedicated figures path.
    FIGURES_DIR = Path(MODEL_DIR).parent / "outputs" / "figures"  # TODO: Match this constant name to config.py

BEST_MODEL_FILENAME = "best_model.joblib"
EVALUATION_METRICS_FILENAME = "evaluation_metrics.json"
EVALUATION_REPORT_FILENAME = "evaluation_report.csv"
SCATTER_PLOT_FILENAME = "actual_vs_predicted.png"
RESIDUAL_PLOT_FILENAME = "residual_plot.png"
RESIDUAL_DIST_FILENAME = "residual_distribution.png"

logger = logging.getLogger(__name__)


def load_trained_model(model_path: Path = MODEL_DIR / BEST_MODEL_FILENAME) -> Pipeline:
    """Load the persisted best model from disk.

    Args:
        model_path: Path to the ``best_model.joblib`` artifact produced
            by ``train_model.py``.

    Returns:
        The deserialized sklearn pipeline.

    Raises:
        FileNotFoundError: If the model artifact does not exist, e.g.
            because training has not been run yet.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. Run train_model.py first."
        )

    logger.info("Loading trained model from %s", model_path)
    return joblib.load(model_path)


def load_validation_split() -> SplitData:
    """Rebuild the same chronological validation split used in training.

    A dedicated held-out test file is not part of this project's
    architecture, so evaluation reuses the exact same time-based split
    routine from ``train_model.py`` on the current feature dataset. This
    keeps evaluation honest: it scores the model on the same slice of
    time it was validated against during training, rather than on an
    arbitrary or re-randomized subset.

    Returns:
        A SplitData instance containing X_val/y_val (and the unused
        train partition, kept only for interface consistency).
    """
    df = load_feature_dataset()
    validate_required_columns(df, required_columns=[STORE_COLUMN, DATE_COLUMN, TARGET_COLUMN])
    df = sort_by_store_and_date(df)
    train_df, val_df, split_date = time_based_train_val_split(df)
    return build_split_data(train_df, val_df, split_date)


def generate_predictions(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Generate predictions for a feature matrix.

    Args:
        model: A fitted sklearn pipeline.
        X: Feature matrix to score.

    Returns:
        Array of predicted target values.

    Raises:
        ValueError: If the model has not been fitted (sklearn will raise
            NotFittedError internally; this is re-raised with context).
    """
    try:
        predictions = model.predict(X)
    except Exception as exc:  # sklearn raises NotFittedError, ValueError, etc.
        raise ValueError(
            "Failed to generate predictions -- the loaded model may be "
            "unfitted or incompatible with the provided features."
        ) from exc
    return predictions


def build_error_summary_table(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Build a per-run error summary table for the evaluation report.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        A one-row DataFrame summarizing error statistics, suitable for
        writing directly to CSV.
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    metrics = compute_regression_metrics(y_true, y_pred)

    summary = {
        "n_samples": len(y_true),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "r2": metrics["r2"],
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "min_residual": float(np.min(residuals)),
        "max_residual": float(np.max(residuals)),
    }
    return pd.DataFrame([summary])


def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    """Plot and save an actual-vs-predicted scatter plot.

    A perfect model would place every point on the y=x diagonal, which
    is drawn as a reference line so deviations are immediately visible.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.
        output_path: File path to save the PNG figure to.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, edgecolors="none")

    lower = min(np.min(y_true), np.min(y_pred))
    upper = max(np.max(y_true), np.max(y_pred))
    ax.plot([lower, upper], [lower, upper], color="red", linestyle="--", label="Ideal fit (y = x)")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved actual-vs-predicted plot to %s", output_path)


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> np.ndarray:
    """Plot and save a residuals-vs-predicted plot.

    Residuals scattered randomly around zero indicate a well-specified
    model; visible curvature or a funnel shape indicates the model is
    missing structure (non-linearity) or has heteroscedastic errors.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.
        output_path: File path to save the PNG figure to.

    Returns:
        The computed residuals array, for reuse by the caller (avoids
        recomputing it for the distribution plot).
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.3, s=10, edgecolors="none")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residuals vs Predicted")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved residual plot to %s", output_path)
    return residuals


def plot_residual_distribution(residuals: np.ndarray, output_path: Path) -> None:
    """Plot and save a histogram of residuals.

    A residual distribution centered near zero and roughly symmetric
    suggests unbiased errors; a skewed or off-center distribution
    indicates the model systematically over- or under-predicts.

    Args:
        residuals: Array of residual values (actual - predicted).
        output_path: File path to save the PNG figure to.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved residual distribution plot to %s", output_path)


def generate_evaluation_plots(
    y_true: pd.Series, y_pred: np.ndarray, figures_dir: Path = FIGURES_DIR
) -> None:
    """Generate and save all evaluation plots.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.
        figures_dir: Directory to save figures into (created if missing).

    Raises:
        OSError: If the figures directory cannot be created.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_actual_vs_predicted(y_true, y_pred, figures_dir / SCATTER_PLOT_FILENAME)
    residuals = plot_residuals(y_true, y_pred, figures_dir / RESIDUAL_PLOT_FILENAME)
    plot_residual_distribution(residuals, figures_dir / RESIDUAL_DIST_FILENAME)


def save_evaluation_metrics(metrics: Dict[str, float], output_dir: Path = MODEL_DIR) -> None:
    """Persist evaluation metrics as JSON.

    Args:
        metrics: Dictionary of computed regression metrics.
        output_dir: Directory to write the metrics file into.

    Raises:
        OSError: If the output directory cannot be created or written to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / EVALUATION_METRICS_FILENAME

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved evaluation metrics to %s", metrics_path)


def save_evaluation_report(report_df: pd.DataFrame, output_dir: Path = MODEL_DIR) -> None:
    """Persist the error summary table as CSV.

    Args:
        report_df: One-row DataFrame produced by
            ``build_error_summary_table``.
        output_dir: Directory to write the report file into.

    Raises:
        OSError: If the output directory cannot be created or written to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / EVALUATION_REPORT_FILENAME

    # quoting=csv.QUOTE_MINIMAL keeps the file readable by both pandas
    # and non-Python tools (e.g. spreadsheet software) without surprises.
    report_df.to_csv(report_path, index=False, quoting=csv.QUOTE_MINIMAL)
    logger.info("Saved evaluation report to %s", report_path)


def run_evaluation_pipeline() -> Dict[str, float]:
    """Run the full evaluation pipeline end-to-end.

    Steps:
        1. Load the persisted best model.
        2. Rebuild the training-time validation split (no retraining).
        3. Generate predictions.
        4. Compute RMSE, MAE, MAPE, R^2.
        5. Generate diagnostic plots (scatter, residuals, distribution).
        6. Save metrics, report, and plots to disk.

    Returns:
        Dictionary of evaluation metrics (rmse, mae, mape, r2).
    """
    configure_logging()

    model = load_trained_model()
    split = load_validation_split()

    predictions = generate_predictions(model, split.X_val)
    metrics = compute_regression_metrics(split.y_val, predictions)
    logger.info("Evaluation metrics: %s", metrics)

    report_df = build_error_summary_table(split.y_val, predictions)
    generate_evaluation_plots(split.y_val, predictions)

    save_evaluation_metrics(metrics)
    save_evaluation_report(report_df)

    return metrics


if __name__ == "__main__":
    run_evaluation_pipeline()
