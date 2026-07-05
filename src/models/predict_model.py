"""Inference pipeline for the retail sales forecasting project.

This module loads the model persisted by ``train_model.py``
(``best_model.joblib``) and applies it to new, unseen data supplied as a
CSV file. It does NOT retrain anything and does NOT redefine column
validation or preprocessing logic that already exists elsewhere in the
project -- those are imported and reused so inference-time behavior
stays identical to training/evaluation-time behavior.

Typical usage (CLI)::

    python -m src.models.predict_model --input new_stores.csv --output predictions.csv

Or programmatically::

    from src.models.predict_model import run_prediction_pipeline
    predictions_df = run_prediction_pipeline("new_stores.csv")

Do NOT modify:
    - src/config.py
    - src/data/load_data.py
    - src/data/merge_data.py
    - src/data/clean_data.py
    - src/features/build_features.py
    - src/models/train_model.py
    - src/models/evaluate_model.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------- #
# Project imports.
#
# Reuse -- rather than reimplement -- column validation, date sorting,
# model loading, and logging setup already defined elsewhere in the
# project. The model pipeline saved by train_model.py already contains
# its own preprocessing step (ColumnTransformer), so "applying the same
# preprocessing" for inference means calling `model.predict(X)` on a
# correctly shaped DataFrame -- not rebuilding a transformer here.
# --------------------------------------------------------------------------- #
from src import config  # TODO: Match this import path to your package layout if different
from src.models.evaluate_model import generate_predictions, load_trained_model
from src.models.train_model import (
    DATE_COLUMN,
    STORE_COLUMN,
    TARGET_COLUMN,
    configure_logging,
    sort_by_store_and_date,
    validate_required_columns,
)

try:
    PREDICTIONS_OUTPUT_PATH: Path = config.PREDICTIONS_OUTPUT_PATH  # TODO: Match this constant name to config.py
except AttributeError:
    # Fall back to a sensible default so the module stays importable
    # before config.py defines a dedicated predictions output path.
    PREDICTIONS_OUTPUT_PATH = Path("outputs") / "predictions.csv"  # TODO: Match this constant name to config.py

PREDICTION_COLUMN_NAME = "predicted_sales"  # TODO: Match this constant name to config.py if defined there

# Columns that must exist in *any* new inference file, regardless of
# which extra engineered features the model also expects. Store and
# Date are required because they identify each prediction row; the full
# feature-column contract is enforced implicitly by the model pipeline
# itself (see run_prediction_pipeline docstring).
REQUIRED_INFERENCE_COLUMNS: List[str] = [STORE_COLUMN, DATE_COLUMN]

logger = logging.getLogger(__name__)


def load_new_data(input_path: Path) -> pd.DataFrame:
    """Load a CSV file containing new, unseen data for inference.

    Args:
        input_path: Path to the input CSV file.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the input path does not exist.
        ValueError: If the path is not a CSV file, or the file is empty
            or malformed.
    """
    input_path = Path(input_path)

    # Fail fast with a specific, actionable error rather than letting
    # pandas raise a generic FileNotFoundError deeper in the stack.
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at '{input_path}'.")
    if input_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported file extension '{input_path.suffix}' -- expected a .csv file."
        )

    logger.info("Loading new data for prediction from %s", input_path)
    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Input file '{input_path}' is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Input file '{input_path}' could not be parsed as CSV.") from exc

    if df.empty:
        raise ValueError(f"Input file '{input_path}' contains no rows.")

    logger.info("Loaded %d rows for prediction", len(df))
    return df


def prepare_inference_features(
    df: pd.DataFrame, date_col: str = DATE_COLUMN, target_col: str = TARGET_COLUMN
) -> pd.DataFrame:
    """Prepare a feature matrix from raw inference input.

    The saved pipeline was trained without the raw date column and
    without the target column (see ``train_model.build_split_data``), so
    inference input must be shaped identically or the ColumnTransformer
    inside the pipeline will reject it. The date column is dropped only
    after being used for sorting; the target column is dropped if
    present at all (new/unseen data normally will not have it, but
    tolerating its presence lets this function also be used for
    backtesting against historical data that does include it).

    Args:
        df: Raw input DataFrame, already validated and sorted.
        date_col: Name of the date column to exclude from features.
        target_col: Name of the target column to exclude from features,
            if present.

    Returns:
        A DataFrame containing only model input features.
    """
    columns_to_drop = [
      col for col in (
         date_col,
         target_col,
         "Customers",
         "SalesLog",
      )
     if col in df.columns
    ]
    return df.drop(columns=columns_to_drop)


def attach_predictions(
    df: pd.DataFrame, predictions: pd.Series, prediction_col: str = PREDICTION_COLUMN_NAME
) -> pd.DataFrame:
    """Attach predictions to the original (unmodified) input DataFrame.

    Predictions are appended to a copy of the original input -- rather
    than returned alongside a bare array -- so the output file remains
    traceable back to the store/date (and any other identifying columns)
    that produced each prediction.

    Args:
        df: Original input DataFrame (pre-feature-preparation).
        predictions: Array-like of predicted values, same length and
            row order as ``df``.
        prediction_col: Name of the new column to store predictions in.

    Returns:
        A new DataFrame equal to ``df`` plus the prediction column.

    Raises:
        ValueError: If the number of predictions does not match the
            number of input rows.
    """
    if len(predictions) != len(df):
        raise ValueError(
            f"Prediction count ({len(predictions)}) does not match "
            f"input row count ({len(df)})."
        )
    result_df = df.copy()
    result_df[prediction_col] = predictions
    return result_df


def save_predictions(df: pd.DataFrame, output_path: Path = PREDICTIONS_OUTPUT_PATH) -> None:
    """Persist predictions to a CSV file.

    Args:
        df: DataFrame containing predictions to save.
        output_path: Destination path for the predictions CSV.

    Raises:
        OSError: If the output directory cannot be created or written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d predictions to %s", len(df), output_path)


def run_prediction_pipeline(
    input_path: Path,
    output_path: Path = PREDICTIONS_OUTPUT_PATH,
    model: Optional[Pipeline] = None,
) -> pd.DataFrame:
    """Run the full inference pipeline end-to-end.

    Steps:
        1. Load the persisted best model (unless one is injected, which
           is useful for tests or batch callers that already have it
           loaded in memory).
        2. Load the new/unseen CSV data.
        3. Validate required identifying columns are present.
        4. Sort by store and date, for consistency with training-time
           row ordering (harmless for row-independent models, and
           required if a future model in this pipeline becomes
           order-sensitive, e.g. a windowed/sequence model).
        5. Drop non-feature columns; the pipeline's own preprocessing
           step handles everything else automatically.
        6. Generate predictions.
        7. Attach predictions to the original data and save to disk.

    Args:
        input_path: Path to the CSV file with new, unseen data.
        output_path: Path to write the predictions CSV to.
        model: Optionally, a pre-loaded model pipeline. If omitted, the
            model is loaded from ``MODEL_DIR / best_model.joblib`` via
            ``evaluate_model.load_trained_model``.

    Returns:
        A DataFrame containing the original input columns plus a
        ``predicted_sales`` column.

    Raises:
        FileNotFoundError: If the input CSV or the trained model file
            does not exist.
        ValueError: If required columns are missing, the file is
            malformed, or the model fails to produce predictions.
    """
    configure_logging()

    if model is None:
        # Reuse evaluate_model's loader instead of duplicating joblib
        # load + existence-check logic here.
        model = load_trained_model()

    raw_df = load_new_data(input_path)

    # Missing columns are handled gracefully: validate_required_columns
    # raises a single, clear ValueError listing every missing column
    # instead of failing deep inside the pipeline with an opaque
    # sklearn/pandas KeyError.
    try:
        validate_required_columns(raw_df, required_columns=REQUIRED_INFERENCE_COLUMNS)
    except ValueError as exc:
        raise ValueError(f"Cannot generate predictions: {exc}") from exc

    sorted_df = sort_by_store_and_date(raw_df)
    features_df = prepare_inference_features(sorted_df)

    predictions = generate_predictions(model, features_df)
    result_df = attach_predictions(sorted_df, predictions)

    save_predictions(result_df, output_path)
    return result_df


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone script execution.

    Returns:
        Parsed arguments with ``input`` and ``output`` paths.
    """
    parser = argparse.ArgumentParser(
        description="Generate sales predictions for new, unseen data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a CSV file containing new, unseen data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PREDICTIONS_OUTPUT_PATH,
        help="Path to write the predictions CSV to.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_prediction_pipeline(input_path=args.input, output_path=args.output)
