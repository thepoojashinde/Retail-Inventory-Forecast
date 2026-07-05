"""
clean_data.py
=============
Responsible for cleaning the merged Rossmann sales + store dataset into
an analysis-ready, model-ready state: correcting data types, resolving
missing values with explicit business justification, and taming
outliers without discarding information.

This module deliberately stops short of feature engineering (lags,
rolling stats, encodings) — that responsibility lives in
``src/features/build_features.py``. Keeping "clean" and "engineer"
separate means a bug in feature logic never forces re-validation of
basic data hygiene, and vice versa.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import LOG_FORMAT, LOG_LEVEL, MERGED_DATA_FILE, PROCESSED_DATA_DIR

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output location
# ---------------------------------------------------------------------------
# NOTE: src.config currently defines CLEANED_DATA_FILE under the *interim*
# directory (used for a lighter intermediate save). This module's explicit
# requirement is to persist the fully cleaned dataset under *processed*,
# since after this step the data is genuinely model-ready. The path is
# still built entirely from src.config.PROCESSED_DATA_DIR rather than a
# hardcoded string, to keep a single source of truth for the data root.
CLEANED_OUTPUT_FILE: Path = PROCESSED_DATA_DIR / "cleaned_data.csv"

# Columns that must never contain missing values after cleaning completes.
# Used by the post-clean validation gate.
CRITICAL_NON_NULL_COLUMNS: list[str] = [
    "Store",
    "Date",
    "Sales",
    "Open",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
]

# Upper percentile used to cap (winsorize) extreme sales values.
# 99th percentile is a conventional, defensible choice: it limits the
# influence of the top 1% of extreme values without discarding rows,
# and preserves every store's time series continuity.
SALES_OUTLIER_UPPER_PERCENTILE: float = 0.99


def generate_data_quality_report(df: pd.DataFrame, stage: str) -> dict[str, Any]:
    """Generate and log a concise data quality report.

    Captures row count, per-column missing value counts, duplicate row
    count, and dtypes — enough to sanity-check the dataset at a glance
    before and after cleaning, without producing an overwhelming dump.

    Args:
        df: DataFrame to summarize.
        stage: Label for this checkpoint (e.g. "pre-clean",
            "post-clean"), used in log output.

    Returns:
        A dictionary containing the report fields, so callers can also
        persist or compare reports programmatically rather than only
        reading logs.
    """
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].to_dict()

    report: dict[str, Any] = {
        "stage": stage,
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_duplicate_rows": int(df.duplicated().sum()),
        "missing_values_by_column": missing_counts,
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    logger.info(
        "[%s] rows=%d, columns=%d, duplicate_rows=%d, columns_with_missing=%d",
        stage,
        report["n_rows"],
        report["n_columns"],
        report["n_duplicate_rows"],
        len(missing_counts),
    )
    if missing_counts:
        logger.info("[%s] missing value counts: %s", stage, missing_counts)

    return report


def remove_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows from the dataset.

    Unlike outliers (which may represent real, informative extreme
    events), an exact duplicate row is a data entry/ETL artifact, not
    information — the same store cannot have two identical records for
    the same day. Dropping these is therefore a correction, not a loss
    of signal.

    Args:
        df: DataFrame to de-duplicate.

    Returns:
        DataFrame with exact duplicate rows removed.
    """
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        logger.warning("Removing %d exact duplicate row(s)", duplicate_count)
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        logger.info("No exact duplicate rows found")
    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to their correct, memory-efficient dtypes.

    Args:
        df: DataFrame with raw/merged dtypes (typically everything
            read from CSV as int64/float64/object).

    Returns:
        DataFrame with corrected dtypes: ``Date`` as datetime, boolean
        flag columns as int8, and categorical text columns as
        pandas ``category`` dtype.
    """
    df = df.copy()

    # Date must be a real datetime for any time-based split, lag, or
    # rolling-window logic downstream to work correctly.
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")

    # Binary flag columns: stored as int8 rather than int64 to reduce
    # memory footprint on a dataset with ~1M rows, with no loss of
    # information (values are strictly 0/1).
    binary_flag_columns = ["Open", "Promo", "SchoolHoliday", "Promo2"]
    for col in binary_flag_columns:
        if col in df.columns:
            df[col] = df[col].astype("int8")

    # Low-cardinality text columns: category dtype is both more memory
    # efficient and communicates intent (these are categorical labels,
    # not free text) to anyone reading the schema later.
    categorical_columns = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Store is an identifier, not a continuous quantity — kept as int
    # but explicitly noted here so it is never accidentally scaled or
    # treated as numeric in later feature engineering.
    df["Store"] = df["Store"].astype("int32")

    logger.info("Data types converted")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve missing values using business-justified imputation rules.

    Every imputation choice below reflects the *reason* a value is
    missing in the Rossmann dataset, rather than a one-size-fits-all
    strategy (e.g. blanket mean-fill), since the mechanism behind each
    column's missingness differs.

    Args:
        df: DataFrame after dtype conversion, prior to missing-value
            handling.

    Returns:
        DataFrame with all known missing-value patterns resolved.
    """
    df = df.copy()

    # --- CompetitionDistance -------------------------------------------------
    # Missing here almost always means "no competitor has been recorded
    # nearby" rather than "unknown distance for an existing competitor".
    # Imputing with the mean/median would fabricate a plausible-looking
    # nearby competitor that doesn't exist, distorting any competition-
    # proximity feature. Instead, we fill with a value larger than any
    # observed distance, so the model reads it as "effectively no
    # competition pressure" — consistent with the missingness mechanism.
    if df["CompetitionDistance"].isnull().any():
        max_observed_distance = df["CompetitionDistance"].max()
        fill_value = (
            max_observed_distance * 1.5 if pd.notnull(max_observed_distance) else 200_000.0
        )
        n_missing = int(df["CompetitionDistance"].isnull().sum())
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(fill_value)
        logger.info(
            "Filled %d missing 'CompetitionDistance' value(s) with %.1f "
            "(> max observed distance, representing 'no known nearby competition')",
            n_missing,
            fill_value,
        )

    # --- CompetitionOpenSinceMonth / CompetitionOpenSinceYear ---------------
    # Missing precisely when no competition-open date is known (often
    # paired with missing/large CompetitionDistance). Filling with 0
    # is a deliberate sentinel meaning "not applicable" rather than a
    # real calendar month/year — any feature using these columns must
    # treat 0 as a flag, not a date component.
    for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]:
        if col in df.columns and df[col].isnull().any():
            n_missing = int(df[col].isnull().sum())
            df[col] = df[col].fillna(0).astype("int32")
            logger.info(
                "Filled %d missing '%s' value(s) with 0 (sentinel for "
                "'no recorded competition open date')",
                n_missing,
                col,
            )

    # --- Promo2SinceWeek / Promo2SinceYear / PromoInterval ------------------
    # These are missing exactly when Promo2 == 0, i.e. the store does
    # not participate in the recurring "Promo2" campaign at all. This
    # is structural missingness, not lost information — there is no
    # "true" start week/year to estimate. Fill with 0 / "None" so the
    # column is complete without inventing a fictitious promo start.
    for col in ["Promo2SinceWeek", "Promo2SinceYear"]:
        if col in df.columns and df[col].isnull().any():
            n_missing = int(df[col].isnull().sum())
            df[col] = df[col].fillna(0).astype("int32")
            logger.info(
                "Filled %d missing '%s' value(s) with 0 (store does not "
                "participate in Promo2)",
                n_missing,
                col,
            )

    if "PromoInterval" in df.columns and df["PromoInterval"].isnull().any():
        n_missing = int(df["PromoInterval"].isnull().sum())
        if "None" not in df["PromoInterval"].cat.categories:
            df["PromoInterval"] = df["PromoInterval"].cat.add_categories(["None"])
        df["PromoInterval"] = df["PromoInterval"].fillna("None")
        logger.info(
            "Filled %d missing 'PromoInterval' value(s) with 'None' "
            "(store does not participate in Promo2)",
            n_missing,
        )

    return df


def flag_closed_store_days(df: pd.DataFrame) -> pd.DataFrame:
    """Add an explicit flag distinguishing structural zero-sales days.

    Days where ``Open == 0`` have ``Sales == 0`` by construction, not
    because demand was zero — the store simply wasn't trading. Treating
    these as ordinary low-sales observations would distort both outlier
    handling and any future demand model. Rather than dropping these
    rows (which would break daily time-series continuity per store),
    we tag them so downstream steps can decide how to treat them.

    Args:
        df: DataFrame with an ``Open`` column.

    Returns:
        DataFrame with an added boolean column, ``IsClosedDay``.
    """
    df = df.copy()
    df["IsClosedDay"] = df["Open"] == 0
    n_closed = int(df["IsClosedDay"].sum())
    logger.info(
        "Flagged %d closed-store day(s) (%.2f%% of rows) as structural zeros",
        n_closed,
        100 * n_closed / len(df) if len(df) else 0.0,
    )
    return df


def handle_outliers(
    df: pd.DataFrame, upper_percentile: float = SALES_OUTLIER_UPPER_PERCENTILE
) -> pd.DataFrame:
    """Cap extreme sales values instead of dropping them.

    Extreme sales days (e.g. major promotions, holiday rushes) are
    real, informative business events — deleting them would teach a
    downstream model to never expect a demand spike, which is exactly
    the scenario inventory forecasting needs to anticipate. Winsorizing
    (capping) at a high percentile limits the leverage of a handful of
    extreme points on model training while preserving every row and the
    store's daily time series continuity.

    Closed-store days (``IsClosedDay``) are excluded from the
    percentile calculation and from capping — their zero sales are
    structural, not part of the demand distribution being modeled.

    Args:
        df: DataFrame with ``Sales`` and ``IsClosedDay`` columns.
        upper_percentile: Percentile threshold used as the sales cap,
            computed only over open-store days.

    Returns:
        DataFrame with ``Sales`` capped at the computed threshold for
        open-store days. Closed-store days are left untouched (already
        zero).
    """
    df = df.copy()
    open_day_sales = df.loc[~df["IsClosedDay"], "Sales"]

    if open_day_sales.empty:
        logger.warning("No open-store days found; skipping outlier capping")
        return df

    cap_value = float(open_day_sales.quantile(upper_percentile))
    n_capped = int((open_day_sales > cap_value).sum())

    df.loc[~df["IsClosedDay"], "Sales"] = open_day_sales.clip(upper=cap_value)

    logger.info(
        "Capped %d open-store-day 'Sales' value(s) above the %.0fth percentile "
        "(%.2f)",
        n_capped,
        upper_percentile * 100,
        cap_value,
    )
    return df


def validate_cleaned_data(df: pd.DataFrame) -> None:
    """Run post-cleaning quality gates before the data is persisted.

    Args:
        df: Fully cleaned DataFrame.

    Raises:
        ValueError: If any critical column still contains missing
            values, if duplicate (Store, Date) records remain, or if
            ``Sales`` contains negative values.
    """
    missing_critical = {
        col: int(df[col].isnull().sum())
        for col in CRITICAL_NON_NULL_COLUMNS
        if col in df.columns and df[col].isnull().any()
    }
    if missing_critical:
        raise ValueError(
            f"Post-clean validation failed: critical columns still contain "
            f"missing values: {missing_critical}"
        )

    duplicate_keys = int(df.duplicated(subset=["Store", "Date"]).sum())
    if duplicate_keys > 0:
        raise ValueError(
            f"Post-clean validation failed: {duplicate_keys} duplicate "
            "(Store, Date) record(s) remain after cleaning."
        )

    if (df["Sales"] < 0).any():
        n_negative = int((df["Sales"] < 0).sum())
        raise ValueError(
            f"Post-clean validation failed: {n_negative} negative 'Sales' "
            "value(s) found."
        )

    logger.info("Post-clean validation passed")


def save_cleaned_data(df: pd.DataFrame, output_path: Path = CLEANED_OUTPUT_FILE) -> None:
    """Persist the cleaned DataFrame to the processed data directory.

    Args:
        df: Cleaned DataFrame to save.
        output_path: Destination path. Defaults to
            ``CLEANED_OUTPUT_FILE`` (under ``PROCESSED_DATA_DIR``).

    Raises:
        OSError: If the file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(output_path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write cleaned data to '{output_path}': {exc}") from exc
    logger.info("Saved cleaned data to '%s' (%s)", output_path, df.shape)


def run_cleaning(
    input_path: Path = MERGED_DATA_FILE,
    output_path: Path = CLEANED_OUTPUT_FILE,
    save_output: bool = True,
) -> pd.DataFrame:
    """Orchestrate the full cleaning workflow end-to-end.

    Loads merged data, reports quality before cleaning, applies dtype
    conversion, missing-value handling, closed-day flagging and outlier
    capping, validates the result, reports quality after cleaning, and
    optionally saves the output.

    Args:
        input_path: Path to the merged dataset produced by
            ``merge_data.py``.
        output_path: Destination path for the cleaned dataset.
        save_output: Whether to persist the cleaned DataFrame to disk.

    Returns:
        The fully cleaned DataFrame.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ValueError: If post-clean validation fails.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Merged data file not found: '{input_path}'. Run merge_data.py first."
        )

    df = pd.read_csv(input_path, dtype={"StateHoliday": str}, low_memory=False)
    generate_data_quality_report(df, stage="pre-clean")

    df = remove_duplicate_records(df)
    df = convert_data_types(df)
    df = handle_missing_values(df)
    df = flag_closed_store_days(df)
    df = handle_outliers(df)

    validate_cleaned_data(df)
    generate_data_quality_report(df, stage="post-clean")

    if save_output:
        save_cleaned_data(df, output_path)

    return df


if __name__ == "__main__":
    # Manual smoke-test entry point: `python -m src.data.clean_data`
    cleaned_data = run_cleaning()
    logger.info("Smoke test passed. cleaned_data shape=%s", cleaned_data.shape)
