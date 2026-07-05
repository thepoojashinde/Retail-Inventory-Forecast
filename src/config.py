"""
config.py
=========
Centralized configuration for the Retail Inventory Demand Forecasting project.

This module defines all file paths, directory locations, and global constants
used across the pipeline (data loading, feature engineering, modeling, and the
Streamlit app). No other module should hardcode a path — everything is
resolved relative to PROJECT_ROOT so the project runs identically on any
machine or deployment environment.
"""

from pathlib import Path

# -------------------------------------------------------------------
# Root paths
# -------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# -------------------------------------------------------------------
# Data directories
# -------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

# Expected raw source files (Rossmann Store Sales dataset)
RAW_TRAIN_FILE: Path = RAW_DATA_DIR / "train.csv"
RAW_STORE_FILE: Path = RAW_DATA_DIR / "store.csv"
RAW_TEST_FILE: Path = RAW_DATA_DIR / "test.csv"

# Intermediate / processed artifacts
MERGED_DATA_FILE: Path = INTERIM_DATA_DIR / "merged_data.csv"
CLEANED_DATA_FILE: Path = INTERIM_DATA_DIR / "cleaned_data.csv"
FEATURED_DATA_FILE: Path = PROCESSED_DATA_DIR / "featured_data.csv"
TRAIN_SPLIT_FILE: Path = PROCESSED_DATA_DIR / "train_split.csv"
VALIDATION_SPLIT_FILE: Path = PROCESSED_DATA_DIR / "validation_split.csv"
TEST_SPLIT_FILE: Path = PROCESSED_DATA_DIR / "test_split.csv"

# -------------------------------------------------------------------
# Model directories
# -------------------------------------------------------------------
MODELS_DIR: Path = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR: Path = MODELS_DIR / "trained"
MODEL_METRICS_DIR: Path = MODELS_DIR / "metrics"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
PREDICTIONS_OUTPUT_PATH: Path = OUTPUTS_DIR / "predictions.csv"

PRODUCTION_MODEL_FILE: Path = TRAINED_MODELS_DIR / "production_model.joblib"
MODEL_REGISTRY_FILE: Path = MODEL_METRICS_DIR / "model_registry.json"

# -------------------------------------------------------------------
# Reports directory
# -------------------------------------------------------------------
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------
RANDOM_STATE: int = 42

# -------------------------------------------------------------------
# Time-based train/validation/test split configuration
# NOTE: This is a time series problem — splits are date-based, never random.
# -------------------------------------------------------------------
DATE_COLUMN: str = "Date"
TARGET_COLUMN: str = "Sales"
STORE_COLUMN: str = "Store"

VALIDATION_SPLIT_DAYS: int = 42   # ~6 weeks held out for validation
TEST_SPLIT_DAYS: int = 42         # ~6 weeks held out for final testing

# -------------------------------------------------------------------
# Feature engineering configuration
# -------------------------------------------------------------------
LAG_DAYS: list[int] = [1, 7, 14, 30]
ROLLING_WINDOWS: list[int] = [7, 30]

# -------------------------------------------------------------------
# Categorical features
# -------------------------------------------------------------------
CATEGORICAL_COLUMNS = [
    "StoreType",
    "Assortment",
    "StateHoliday",
    "PromoInterval",
]

# -------------------------------------------------------------------
# Model training configuration
# -------------------------------------------------------------------
CV_N_SPLITS: int = 5

MODEL_CANDIDATES: list[str] = [
    "baseline_naive",
    "linear_regression",
    "random_forest",
    "gradient_boosting",
]

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# -------------------------------------------------------------------
# Streamlit app configuration
# -------------------------------------------------------------------
APP_TITLE: str = "Retail Inventory Demand Forecasting & Business Analytics"
APP_LAYOUT: str = "wide"


def ensure_directories() -> None:
    """
    Create all required project directories if they do not already exist.
    Safe to call multiple times (idempotent). Intended to be invoked once
    at the start of the pipeline or app entry point.
    """
    required_dirs = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        TRAINED_MODELS_DIR,
        MODEL_METRICS_DIR,
        FIGURES_DIR,
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
