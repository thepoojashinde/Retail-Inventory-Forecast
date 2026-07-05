"""
load_data.py
============
Responsible for loading the raw Rossmann Store Sales CSV files
(``train.csv`` and ``store.csv``) from disk into pandas DataFrames.

This module performs no cleaning or transformation — its single
responsibility is to get raw data safely and correctly into memory,
failing loudly and early if the source files are missing or malformed.
Downstream cleaning/merging logic lives in ``merge_data.py`` and
``clean_data.py``.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import LOG_FORMAT, LOG_LEVEL, RAW_STORE_FILE, RAW_TRAIN_FILE

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# Using logging instead of print() so log verbosity/output can be controlled
# centrally (e.g. redirected to a file or silenced) once this module is used
# inside the larger pipeline, without touching this file again.
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected schemas
# ---------------------------------------------------------------------------
# Declared as module-level constants (not hardcoded inline) so schema
# expectations are visible at a glance and easy to update if the dataset
# version changes.
REQUIRED_TRAIN_COLUMNS: list[str] = [
    "Store",
    "DayOfWeek",
    "Date",
    "Sales",
    "Customers",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
]

REQUIRED_STORE_COLUMNS: list[str] = [
    "Store",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "PromoInterval",
]


def _validate_file_exists(file_path: Path) -> None:
    """Validate that a required raw data file exists on disk.

    Args:
        file_path: Path to the file expected to exist.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    if not file_path.exists():
        # Fail fast with an actionable message rather than letting pandas
        # raise a less informative low-level error further downstream.
        raise FileNotFoundError(
            f"Required data file not found: '{file_path}'. "
            "Please download the Rossmann Store Sales dataset from Kaggle "
            "and place it under 'data/raw/'."
        )
    logger.debug("Validated file exists: %s", file_path)


def _validate_required_columns(
    df: pd.DataFrame, required_columns: list[str], source_name: str
) -> None:
    """Validate that a DataFrame contains all required columns.

    Args:
        df: DataFrame to validate.
        required_columns: Column names that must be present.
        source_name: Human-readable name of the source file, used in
            error messages (e.g. "train.csv").

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"'{source_name}' is missing required column(s): {missing_columns}. "
            f"Expected schema: {required_columns}."
        )
    logger.debug("Validated required columns present in %s", source_name)


def load_train_data(file_path: Path = RAW_TRAIN_FILE) -> pd.DataFrame:
    """Load and validate the raw training sales data.

    Args:
        file_path: Path to ``train.csv``. Defaults to the path configured
            in ``src.config.RAW_TRAIN_FILE``.

    Returns:
        A pandas DataFrame containing the raw training data, with the
        ``StateHoliday`` column cast to string (it is mixed-type in the
        source CSV: numeric 0 vs. categorical labels like 'a', 'b', 'c').

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If required columns are missing from the file.
    """
    logger.info("Loading training data from '%s'", file_path)
    _validate_file_exists(file_path)

    # dtype is fixed for StateHoliday up front to avoid pandas' mixed-type
    # inference warning, since the raw column contains both 0 (int) and
    # 'a'/'b'/'c' (str) values that must be treated as one categorical type.
    df = pd.read_csv(file_path, dtype={"StateHoliday": str}, low_memory=False)

    _validate_required_columns(df, REQUIRED_TRAIN_COLUMNS, source_name="train.csv")
    logger.info("Loaded train.csv with shape %s", df.shape)
    return df


def load_store_data(file_path: Path = RAW_STORE_FILE) -> pd.DataFrame:
    """Load and validate the raw store metadata.

    Args:
        file_path: Path to ``store.csv``. Defaults to the path configured
            in ``src.config.RAW_STORE_FILE``.

    Returns:
        A pandas DataFrame containing the raw store metadata (one row
        per store).

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If required columns are missing from the file.
    """
    logger.info("Loading store metadata from '%s'", file_path)
    _validate_file_exists(file_path)

    df = pd.read_csv(file_path, low_memory=False)

    _validate_required_columns(df, REQUIRED_STORE_COLUMNS, source_name="store.csv")
    logger.info("Loaded store.csv with shape %s", df.shape)
    return df


def load_raw_data(
    train_file_path: Path = RAW_TRAIN_FILE,
    store_file_path: Path = RAW_STORE_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both raw source files required for this project.

    Convenience wrapper so downstream callers (e.g. ``merge_data.py`` or
    the pipeline orchestrator) can retrieve both DataFrames with a single
    call rather than importing and calling each loader individually.

    Args:
        train_file_path: Path to ``train.csv``.
        store_file_path: Path to ``store.csv``.

    Returns:
        A tuple of ``(train_df, store_df)``.

    Raises:
        FileNotFoundError: If either source file is missing.
        ValueError: If either source file is missing required columns.
    """
    train_df = load_train_data(train_file_path)
    store_df = load_store_data(store_file_path)
    return train_df, store_df


if __name__ == "__main__":
    # Manual smoke-test entry point: `python -m src.data.load_data`
    # Not part of the pipeline — useful for quickly verifying that raw
    # files are in place and schema-valid during local development.
    train_data, store_data = load_raw_data()
    logger.info("Smoke test passed. train=%s, store=%s", train_data.shape, store_data.shape)
