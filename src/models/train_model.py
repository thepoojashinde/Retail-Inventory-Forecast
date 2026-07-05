"""Model training pipeline for the Rossmann retail sales forecasting project.

This module loads the feature-engineered dataset produced by
``src.features.build_features``, removes leakage-prone columns, performs
a chronological train/validation split, trains several candidate
regressors through a proper imputation + encoding pipeline, tunes the
single best candidate with ``RandomizedSearchCV`` + ``TimeSeriesSplit``,
and persists the final model, metrics, and training metadata.

Typical usage::

    python -m src.models.train_model

Or programmatically::

    from src.models.train_model import run_training_pipeline
    model = run_training_pipeline()

Do NOT modify:
    - src/config.py
    - src/data/load_data.py
    - src/data/merge_data.py
    - src/data/clean_data.py
    - src/features/build_features.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config

# --------------------------------------------------------------------------- #
# Config-driven constants. Required constants raise at import time if
# missing; optional constants fall back to safe defaults via getattr().
# --------------------------------------------------------------------------- #
try:
    FEATURES_DATA_PATH: Path = config.FEATURED_DATA_FILE
except AttributeError as exc:
    raise ImportError(
        "config.py must define FEATURED_DATA_FILE pointing to the "
        "feature-engineered dataset produced by build_features.py."
    ) from exc

try:
    MODEL_DIR: Path = config.TRAINED_MODELS_DIR
except AttributeError as exc:
    raise ImportError(
        "config.py must define TRAINED_MODELS_DIR: the directory where "
        "trained artifacts (model, metrics, metadata) are written."
    ) from exc

try:
    TARGET_COLUMN: str = config.TARGET_COLUMN
except AttributeError as exc:
    raise ImportError("config.py must define TARGET_COLUMN, e.g. 'Sales'.") from exc

try:
    DATE_COLUMN: str = config.DATE_COLUMN
except AttributeError as exc:
    raise ImportError("config.py must define DATE_COLUMN, e.g. 'Date'.") from exc

try:
    STORE_COLUMN: str = config.STORE_COLUMN
except AttributeError as exc:
    raise ImportError("config.py must define STORE_COLUMN, e.g. 'Store'.") from exc

RANDOM_STATE: int = getattr(config, "RANDOM_STATE", 42)
VALIDATION_FRACTION: float = getattr(config, "VALIDATION_FRACTION", 0.2)
CATEGORICAL_COLUMNS: List[str] = getattr(config, "CATEGORICAL_COLUMNS", [])

# Columns known in advance to leak target information (e.g. not available
# at inference time, or a direct transform of the target).
KNOWN_LEAKAGE_COLUMNS: List[str] = getattr(
    config, "KNOWN_LEAKAGE_COLUMNS", ["SalesLog", "Customers"]
)

# --------------------------------------------------------------------------- #
# Single switch between fast local iteration and a full production run.
# Every other setting downstream is derived from this one flag via
# get_active_training_config() -- no other code path changes are needed.
# --------------------------------------------------------------------------- #
DEBUG_MODE: bool = False
DEBUG_SAMPLE_SIZE: int = 150_000

BEST_MODEL_FILENAME = "best_model.joblib"
METRICS_FILENAME = "metrics.json"
METADATA_FILENAME = "training_metadata.json"

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Container for a chronological train/validation split.

    Attributes:
        X_train: Training feature matrix.
        y_train: Training target vector.
        X_val: Validation feature matrix.
        y_val: Validation target vector.
        split_date: Timestamp used as the train/validation boundary.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    split_date: pd.Timestamp


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging for the training pipeline.

    Args:
        level: Logging verbosity level.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_active_training_config() -> Dict[str, Any]:
    """Resolve DEBUG vs. production settings from a single flag.

    Centralizing this switch means every other function just reads
    values from one dict instead of scattering ``if DEBUG_MODE`` checks
    throughout the pipeline.

    Returns:
        Dictionary with sample_size (None means "use all rows"),
        n_estimators, search_n_iter, and cv_splits.
    """
    if DEBUG_MODE:
        logger.info("DEBUG MODE enabled: training on a %d-row sample with lightweight hyperparameter search.", DEBUG_SAMPLE_SIZE)
        return {
            "sample_size": DEBUG_SAMPLE_SIZE,
            "n_estimators": 50,
            "search_n_iter": 3,
            "cv_splits": 3,
        }

    logger.info("PRODUCTION MODE enabled: training on the full dataset with the complete hyperparameter search.")
    return {
        "sample_size": None,
        "n_estimators": 200,
        "search_n_iter": 20,
        "cv_splits": 5,
    }


def load_feature_dataset(path: Path = FEATURES_DATA_PATH) -> pd.DataFrame:
    """Load the feature-engineered dataset from disk.

    Args:
        path: Path to the parquet/csv file produced by build_features.py.

    Returns:
        The loaded dataset as a DataFrame.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at '{path}'. Run build_features.py first."
        )

    logger.info("Loading feature-engineered dataset from %s", path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension '{path.suffix}' for feature dataset.")

    logger.info("Loaded dataset with shape %s", df.shape)
    return df


def apply_debug_sampling(df: pd.DataFrame, sample_size: int | None) -> pd.DataFrame:
    """Optionally subsample rows for fast local iteration.

    Sampling is done uniformly at random purely for speed -- chronological
    order does not need to be preserved here because
    ``sort_by_store_and_date`` is always applied afterward, restoring
    correct time ordering before any split occurs.

    Args:
        df: Full dataset.
        sample_size: Number of rows to sample, or None to use all rows.

    Returns:
        The sampled (or original) DataFrame.
    """
    if sample_size is None or sample_size >= len(df):
        return df

    logger.info("Sampling %d rows out of %d.", sample_size, len(df))
    return df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)


def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Ensure all required columns are present before training.

    Args:
        df: The dataset to validate.
        required_columns: Columns that must be present.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in dataset: {missing}")
    logger.info("All required columns present: %s", required_columns)


def sort_by_store_and_date(
    df: pd.DataFrame, store_col: str = STORE_COLUMN, date_col: str = DATE_COLUMN
) -> pd.DataFrame:
    """Sort the dataset by store and date.

    Sorting is mandatory because the time-based split and
    ``TimeSeriesSplit`` both assume row order reflects chronological
    order; skipping this step would silently produce a leaky or
    meaningless split, especially after random debug sampling.

    Args:
        df: Input dataset.
        store_col: Name of the store identifier column.
        date_col: Name of the date column.

    Returns:
        A new DataFrame sorted by store then date, with the date column
        cast to datetime.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=[store_col, date_col]).reset_index(drop=True)
    logger.info("Sorted dataset by ['%s', '%s']", store_col, date_col)
    return df


def identify_leakage_columns(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    known_leakage_columns: List[str] = KNOWN_LEAKAGE_COLUMNS,
) -> List[str]:
    """Identify columns that would leak target information at inference time.

    Three checks are combined:

    1. Explicit known leakage columns (e.g. 'Customers', not known in
       advance for a future day, and 'SalesLog', a direct transform of
       the target).
    2. Any column whose name contains the target column's name, since
       feature-engineering steps often derive columns like
       'Sales_per_Customer' directly from the target. Columns
       containing 'lag' or 'roll' (case-insensitive) are exempted, since
       these are legitimate time-series features that only look
       backward (e.g. 'Sales_lag_7', 'Sales_rolling_mean_30').
    3. The target column itself.

    Args:
        df: Feature-engineered dataset.
        target_col: Name of the prediction target.
        known_leakage_columns: Explicit list of columns known in advance
            to leak.

    Returns:
        Sorted list of column names to exclude from the feature matrix.
    """
    leakage_columns: set[str] = set()

    for col in known_leakage_columns:
        if col in df.columns:
            leakage_columns.add(col)

    target_lower = target_col.lower()
    for col in df.columns:
        col_lower = col.lower()
        is_target_derived = target_lower in col_lower and col != target_col
        is_lag_or_rolling = "lag" in col_lower or "roll" in col_lower
        if is_target_derived and not is_lag_or_rolling:
            leakage_columns.add(col)

    leakage_columns.add(target_col)

    sorted_columns = sorted(leakage_columns)
    logger.info("Identified leakage/target column(s) to exclude from features: %s", sorted_columns)
    return sorted_columns


def time_based_train_val_split(
    df: pd.DataFrame,
    date_col: str = DATE_COLUMN,
    validation_fraction: float = VALIDATION_FRACTION,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split the dataset into train/validation sets by time, not randomly.

    A random split would let the model see future dates during training
    and be validated on past dates, leaking information and producing an
    overly optimistic estimate of real-world performance. A single
    global cutoff date is chosen so every training row occurs strictly
    before every validation row.

    Args:
        df: Dataset sorted by store and date.
        date_col: Name of the date column.
        validation_fraction: Fraction of the most recent time range to
            reserve for validation.

    Returns:
        A tuple of (train_df, val_df, split_date).

    Raises:
        ValueError: If validation_fraction is not in (0, 1), or the
            split produces an empty train or validation set.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1 (exclusive).")

    min_date, max_date = df[date_col].min(), df[date_col].max()
    split_date = max_date - validation_fraction * (max_date - min_date)

    train_df = df[df[date_col] < split_date].copy()
    val_df = df[df[date_col] >= split_date].copy()

    if train_df.empty or val_df.empty:
        raise ValueError(
            "Time-based split produced an empty train or validation set; "
            "check the date range and validation_fraction."
        )

    logger.info(
        "Time-based split at %s | train rows=%d (%s to %s) | val rows=%d (%s to %s)",
        split_date.date(),
        len(train_df),
        train_df[date_col].min().date(),
        train_df[date_col].max().date(),
        len(val_df),
        val_df[date_col].min().date(),
        val_df[date_col].max().date(),
    )
    return train_df, val_df, split_date


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    extra_drop_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into a leakage-free feature matrix X and target y.

    This is the single source of truth for which columns are excluded
    from X. It combines automatically detected leakage columns with any
    caller-supplied columns (e.g. the raw date, once it has served its
    ordering purpose).

    Args:
        df: Input dataset containing the target column.
        target_col: Name of the target column to predict.
        extra_drop_columns: Additional non-leakage columns to exclude.

    Returns:
        A tuple (X, y).

    Raises:
        ValueError: If the target column is missing.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    leakage_columns = identify_leakage_columns(df, target_col=target_col)
    columns_to_drop = set(leakage_columns) | set(extra_drop_columns or [])
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    X = df.drop(columns=columns_to_drop)
    y = df[target_col]
    logger.info("Prepared feature matrix with %d columns (dropped %d).", X.shape[1], len(columns_to_drop))
    return X, y


def build_split_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    split_date: pd.Timestamp,
    target_col: str = TARGET_COLUMN,
    date_col: str = DATE_COLUMN,
) -> SplitData:
    """Assemble X/y for train and validation into a single container.

    The raw date column is dropped from X after having served its
    purpose for sorting and splitting; a raw timestamp is not directly
    usable by these estimators without further encoding.

    Args:
        train_df: Training partition.
        val_df: Validation partition.
        split_date: The cutoff date used to create the split.
        target_col: Name of the target column.
        date_col: Name of the date column to drop from features.

    Returns:
        A populated SplitData instance.
    """
    X_train, y_train = prepare_features_and_target(train_df, target_col, extra_drop_columns=[date_col])
    X_val, y_val = prepare_features_and_target(val_df, target_col, extra_drop_columns=[date_col])
    return SplitData(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, split_date=split_date)


def detect_categorical_columns(X: pd.DataFrame) -> List[str]:
    """Detect categorical columns, preferring config over dtype inference.

    If ``config.CATEGORICAL_COLUMNS`` is non-empty, only those columns
    (that are actually present) are treated as categorical. Otherwise
    dtype (object/category/bool) is used, so this pipeline keeps working
    unchanged whenever build_features.py adds, removes, or renames
    categorical features without config.py being updated in lockstep.

    Args:
        X: Feature matrix.

    Returns:
        List of column names considered categorical.
    """
    if CATEGORICAL_COLUMNS:
        categorical_columns = [col for col in CATEGORICAL_COLUMNS if col in X.columns]
    else:
        categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    logger.info("Detected categorical columns: %s", categorical_columns)
    return categorical_columns


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing ColumnTransformer for numeric and categorical features.

    Numeric columns are median-imputed because median is robust to the
    skewed distributions typical of retail sales/count data. Categorical
    columns are imputed with the most frequent value, then one-hot
    encoded with unknown-category handling enabled so categories unseen
    during training (e.g. a new StoreType value at inference time) do
    not crash the pipeline.

    Args:
        X: Feature matrix used to determine which columns are numeric
            vs. categorical.

    Returns:
        An unfitted ColumnTransformer.
    """
    categorical_columns = detect_categorical_columns(X)
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
    return preprocessor


def build_candidate_models(preprocessor: ColumnTransformer, n_estimators: int) -> Dict[str, Pipeline]:
    """Construct the candidate model pipelines to compare.

    Each candidate shares the same preprocessing step so that the
    downstream metric comparison is attributable to the estimator, not
    to inconsistent feature handling.

    Args:
        preprocessor: Shared preprocessing ColumnTransformer.
        n_estimators: Number of trees for the RandomForest candidate;
            driven by DEBUG_MODE so debug runs stay fast.

    Returns:
        Mapping of model name to an unfitted sklearn Pipeline.
    """
    return {
        "baseline_mean": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                # DummyRegressor gives the "predict the mean" floor that
                # any real model must clear to be worth deploying.
                ("regressor", DummyRegressor(strategy="mean")),
            ]
        ),
        "linear_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]
        ),
    }


def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        Dictionary with RMSE, MAE, MAPE, and R^2.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    # MAPE is undefined for rows where y_true == 0 (e.g. closed-store
    # days with zero sales); those rows are excluded so a handful of
    # zero-sale days do not blow the metric up to infinity.
    non_zero_mask = y_true != 0
    if non_zero_mask.any():
        mape = float(mean_absolute_percentage_error(y_true[non_zero_mask], y_pred[non_zero_mask]))
    else:
        mape = float("nan")
        logger.warning("All y_true values are zero; MAPE is undefined and set to NaN.")

    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def train_and_evaluate_candidates(
    models: Dict[str, Pipeline], split: SplitData
) -> Dict[str, Dict[str, float]]:
    """Fit every candidate model and evaluate it on the validation split.

    Args:
        models: Mapping of model name to unfitted pipeline.
        split: Train/validation data container.

    Returns:
        Mapping of model name to its metrics dictionary.
    """
    results: Dict[str, Dict[str, float]] = {}
    for name, pipeline in models.items():
        logger.info("Training candidate model: %s", name)
        pipeline.fit(split.X_train, split.y_train)
        predictions = pipeline.predict(split.X_val)
        metrics = compute_regression_metrics(split.y_val, predictions)
        results[name] = metrics
        logger.info("Metrics for %s: %s", name, metrics)
    return results


def get_search_space_for_model(model_name: str) -> Dict[str, Any]:
    """Return the RandomizedSearchCV parameter grid for a given model.

    Args:
        model_name: Key identifying the model, as used in
            ``build_candidate_models``.

    Returns:
        A hyperparameter distribution dictionary, prefixed for the
        pipeline's "regressor" step.

    Raises:
        ValueError: If no search space is defined for the given model.
    """
    search_spaces: Dict[str, Dict[str, Any]] = {
        "random_forest": {
            "regressor__n_estimators": [50, 100, 200, 300, 400],
            "regressor__max_depth": [None, 8, 12, 16, 24],
            "regressor__min_samples_leaf": [1, 2, 4, 8],
        },
        "hist_gradient_boosting": {
            "regressor__max_iter": [100, 200, 300],
            "regressor__max_depth": [None, 6, 10, 15],
            "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "regressor__l2_regularization": [0.0, 0.1, 1.0],
        },
        "linear_regression": {
            "regressor__fit_intercept": [True, False],
        },
    }
    if model_name not in search_spaces:
        raise ValueError(f"No hyperparameter search space defined for model '{model_name}'.")
    return search_spaces[model_name]


def tune_best_candidate(
    pipeline: Pipeline,
    split: SplitData,
    param_distributions: Dict[str, Any],
    n_iter: int,
    cv_splits: int,
) -> Pipeline:
    """Tune the single strongest candidate with RandomizedSearchCV.

    Hyperparameter search is restricted to the one best-performing model
    from the initial comparison -- searching all four candidates would
    multiply training cost for little benefit, since three of them
    already lost the comparison. ``TimeSeriesSplit`` (not standard
    k-fold) is used so every validation fold still occurs strictly after
    its training fold, preserving the no-leakage guarantee from the
    outer split.

    Args:
        pipeline: The pipeline for the best candidate model, unfitted.
        split: Train/validation data container (search runs on the
            training partition only).
        param_distributions: Hyperparameter search space, keyed with the
            ``regressor__`` prefix.
        n_iter: Number of parameter settings sampled; driven by
            DEBUG_MODE.
        cv_splits: Number of TimeSeriesSplit folds; driven by DEBUG_MODE.

    Returns:
        The refit pipeline with the best-found hyperparameters.
    """
    logger.info("Starting RandomizedSearchCV (n_iter=%d, cv_splits=%d).", n_iter, cv_splits)
    time_series_cv = TimeSeriesSplit(n_splits=cv_splits)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=time_series_cv,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(split.X_train, split.y_train)
    logger.info("Best params found: %s", search.best_params_)
    logger.info("Best CV score (neg RMSE): %.4f", search.best_score_)
    return search.best_estimator_


def select_best_model(results: Dict[str, Dict[str, float]]) -> str:
    """Select the best model name by lowest validation RMSE.

    RMSE is the primary selection criterion because it is in the same
    units as the target and penalizes large errors more heavily, which
    matters for sales forecasting where large misses are costlier than
    small ones.

    Args:
        results: Mapping of model name to metrics dictionary.

    Returns:
        The name of the best-performing model.

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        raise ValueError("Cannot select a best model from empty results.")
    best_name = min(results, key=lambda name: results[name]["rmse"])
    logger.info("Selected best model: %s (RMSE=%.4f)", best_name, results[best_name]["rmse"])
    return best_name


def save_training_artifacts(
    model: Pipeline,
    metrics: Dict[str, Dict[str, float]],
    metadata: Dict[str, Any],
    output_dir: Path = MODEL_DIR,
) -> None:
    """Persist the trained model, metrics, and training metadata to disk.

    Args:
        model: The final trained (and possibly tuned) pipeline to persist.
        metrics: Metrics for every candidate model, for auditability.
        metadata: Training run metadata (split date, feature list, etc.).
        output_dir: Directory to write artifacts into.

    Raises:
        OSError: If the output directory cannot be created or written to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / BEST_MODEL_FILENAME
    metrics_path = output_dir / METRICS_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    joblib.dump(model, model_path)
    logger.info("Saved best model to %s", model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved training metadata to %s", metadata_path)


def run_training_pipeline() -> Pipeline:
    """Run the full training pipeline end-to-end.

    Steps:
        1. Resolve DEBUG_MODE vs. production settings.
        2. Load the feature-engineered dataset (optionally sampled).
        3. Validate required columns.
        4. Sort by store and date.
        5. Perform a time-based train/validation split.
        6. Build a leakage-free feature matrix via an auto-detecting
           preprocessing pipeline (median/most-frequent imputation +
           one-hot encoding).
        7. Train and compare baseline, linear, random forest, and
           HistGradientBoosting models.
        8. Tune the best candidate with RandomizedSearchCV + TimeSeriesSplit.
        9. Persist the final model, metrics, and metadata.

    Returns:
        The final trained (and tuned, if applicable) model pipeline.
    """
    configure_logging()
    training_config = get_active_training_config()

    df = load_feature_dataset()
    df = apply_debug_sampling(df, training_config["sample_size"])
    validate_required_columns(df, required_columns=[STORE_COLUMN, DATE_COLUMN, TARGET_COLUMN])
    df = sort_by_store_and_date(df)

    train_df, val_df, split_date = time_based_train_val_split(df)
    split = build_split_data(train_df, val_df, split_date)

    preprocessor = build_preprocessor(split.X_train)
    candidate_models = build_candidate_models(preprocessor, n_estimators=training_config["n_estimators"])
    results = train_and_evaluate_candidates(candidate_models, split)

    best_model_name = select_best_model(results)

    # Only the winning candidate is tuned; a DummyRegressor has nothing
    # meaningful to search over, so it is shipped as-is if it somehow wins.
    if best_model_name == "baseline_mean":
        logger.warning(
            "Baseline model won the comparison; skipping hyperparameter "
            "tuning and shipping the baseline as the production model."
        )
        final_model = candidate_models[best_model_name]
    else:
        search_space = get_search_space_for_model(best_model_name)
        final_model = tune_best_candidate(
            candidate_models[best_model_name],
            split,
            search_space,
            n_iter=training_config["search_n_iter"],
            cv_splits=training_config["cv_splits"],
        )
        # Re-evaluate after tuning so persisted metrics reflect the final
        # production model, not just the untuned candidate.
        tuned_predictions = final_model.predict(split.X_val)
        results[f"{best_model_name}_tuned"] = compute_regression_metrics(split.y_val, tuned_predictions)
        best_model_name = f"{best_model_name}_tuned"

    metadata = {
        "debug_mode": DEBUG_MODE,
        "target_column": TARGET_COLUMN,
        "date_column": DATE_COLUMN,
        "store_column": STORE_COLUMN,
        "split_date": split_date,
        "train_rows": len(split.X_train),
        "validation_rows": len(split.X_val),
        "feature_columns": list(split.X_train.columns),
        "categorical_columns": detect_categorical_columns(split.X_train),
        "excluded_leakage_columns": identify_leakage_columns(df),
        "candidate_models": list(candidate_models.keys()),
        "selected_model": best_model_name,
        "random_state": RANDOM_STATE,
        "training_config": training_config,
    }

    save_training_artifacts(final_model, results, metadata)
    logger.info("Training pipeline finished successfully.")
    return final_model


if __name__ == "__main__":
    run_training_pipeline()