"""
merge_data.py
=============
Responsible for merging the raw training sales data with store-level
metadata into a single, analysis-ready DataFrame.

This module performs no cleaning beyond what is strictly required to
merge safely (e.g. verifying join-key uniqueness) — general cleaning
(missing values, outliers, dtype normalization) lives in
``clean_data.py``. Keeping "merge" and "clean" separate mirrors how a
real data engineering pipeline stages transformations, and makes each
step independently testable.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import LOG_FORMAT, LOG_LEVEL, MERGED_DATA_FILE
from src.data.load_data import load_raw_data

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# The join key linking daily sales records to store metadata.
# Declared as a constant rather than a magic string repeated inline.
MERGE_KEY: str = "Store"


def _validate_merge_key_present(df: pd.DataFrame, source_name: str) -> None:
    """Validate that the merge key column exists in a DataFrame.

    Args:
        df: DataFrame expected to contain the merge key.
        source_name: Human-readable name of the DataFrame's source,
            used in error messages (e.g. "train_df").

    Raises:
        ValueError: If ``MERGE_KEY`` is not a column in ``df``.
    """
    if MERGE_KEY not in df.columns:
        raise ValueError(
            f"'{source_name}' is missing the merge key column '{MERGE_KEY}'. "
            f"Available columns: {list(df.columns)}."
        )


def _validate_store_uniqueness(store_df: pd.DataFrame) -> None:
    """Validate that store metadata has exactly one row per store.

    A left join against a non-unique key would silently duplicate sales
    rows, quietly corrupting downstream aggregates. This check turns
    that risk into a loud, early failure instead.

    Args:
        store_df: Store metadata DataFrame.

    Raises:
        ValueError: If duplicate ``Store`` IDs are found in ``store_df``.
    """
    duplicate_count = int(store_df[MERGE_KEY].duplicated().sum())
    if duplicate_count > 0:
        duplicate_ids = store_df.loc[
            store_df[MERGE_KEY].duplicated(keep=False), MERGE_KEY
        ].unique()
        raise ValueError(
            f"store.csv contains {duplicate_count} duplicate '{MERGE_KEY}' "
            f"entries (duplicate IDs: {sorted(duplicate_ids.tolist())[:10]}...). "
            "Expected exactly one row per store."
        )


def _log_unmatched_stores(train_df: pd.DataFrame, store_df: pd.DataFrame) -> None:
    """Log a warning if any stores in the sales data lack metadata.

    This does not raise an exception because a left join handles the
    situation gracefully (metadata columns become NaN, to be resolved
    in ``clean_data.py``) — but it is important to surface visibly,
    since silent NaNs from a bad join are a classic source of hidden
    data quality bugs.

    Args:
        train_df: Raw training sales data.
        store_df: Store metadata.
    """
    train_store_ids = set(train_df[MERGE_KEY].unique())
    known_store_ids = set(store_df[MERGE_KEY].unique())
    unmatched = train_store_ids - known_store_ids

    if unmatched:
        logger.warning(
            "%d store ID(s) present in train.csv have no matching metadata "
            "in store.csv: %s",
            len(unmatched),
            sorted(unmatched)[:10],
        )
    else:
        logger.info("All store IDs in train.csv have matching metadata.")


def merge_train_and_store(
    train_df: pd.DataFrame, store_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge daily sales records with store-level metadata.

    Performs a left join so that every sales record is preserved even
    if store metadata is (unexpectedly) incomplete — dropping sales
    rows during a merge would silently bias the dataset.

    Args:
        train_df: Raw training sales data (one row per store per day).
        store_df: Store metadata (one row per store).

    Returns:
        A merged DataFrame with the same number of rows as ``train_df``,
        enriched with store metadata columns.

    Raises:
        ValueError: If the merge key is missing from either input, if
            ``store_df`` contains duplicate store IDs, or if the merge
            unexpectedly changes the row count of ``train_df``.
    """
    _validate_merge_key_present(train_df, source_name="train_df")
    _validate_merge_key_present(store_df, source_name="store_df")
    _validate_store_uniqueness(store_df)
    _log_unmatched_stores(train_df, store_df)

    expected_row_count = len(train_df)
    logger.info(
        "Merging train_df (%d rows) with store_df (%d rows) on '%s'",
        expected_row_count,
        len(store_df),
        MERGE_KEY,
    )

    merged_df = train_df.merge(store_df, on=MERGE_KEY, how="left")

    # A left join on a verified-unique key must preserve row count exactly.
    # If it doesn't, something upstream (e.g. an unexpected duplicate that
    # slipped past validation) has silently corrupted the data.
    if len(merged_df) != expected_row_count:
        raise ValueError(
            f"Merge changed row count unexpectedly: expected "
            f"{expected_row_count}, got {len(merged_df)}. Aborting to "
            "prevent silent data corruption."
        )

    logger.info("Merge complete. Resulting shape: %s", merged_df.shape)
    return merged_df


def save_merged_data(
    df: pd.DataFrame, output_path: Path = MERGED_DATA_FILE
) -> None:
    """Persist the merged DataFrame to the interim data directory.

    Args:
        df: Merged DataFrame to save.
        output_path: Destination path. Defaults to the path configured
            in ``src.config.MERGED_DATA_FILE``.

    Raises:
        OSError: If the file cannot be written (e.g. permissions issue,
            disk full).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(output_path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write merged data to '{output_path}': {exc}") from exc
    logger.info("Saved merged data to '%s' (%s)", output_path, df.shape)


def run_merge(save_output: bool = True) -> pd.DataFrame:
    """Orchestrate the full load-then-merge step.

    Convenience entry point for the pipeline: loads raw train/store
    data and returns the merged result, optionally persisting it to
    disk.

    Args:
        save_output: Whether to write the merged DataFrame to the
            interim data directory. Defaults to ``True``.

    Returns:
        The merged DataFrame.
    """
    train_df, store_df = load_raw_data()
    merged_df = merge_train_and_store(train_df, store_df)

    if save_output:
        save_merged_data(merged_df)

    return merged_df


if __name__ == "__main__":
    # Manual smoke-test entry point: `python -m src.data.merge_data`
    merged_data = run_merge()
    logger.info("Smoke test passed. merged_data shape=%s", merged_data.shape)
