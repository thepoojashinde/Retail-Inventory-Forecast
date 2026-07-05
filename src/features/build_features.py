"""
build_features.py
==================
Responsible for turning the cleaned Rossmann sales dataset into a
model-ready feature set: calendar signals, competition/promo duration
signals, autoregressive lag and rolling-window statistics, a
log-transformed target, and cyclical encodings.

This module performs no model training and no train/validation/test
splitting — it only constructs columns. Splitting strategy and model
fitting live in later pipeline stages, so that feature logic can be
unit-tested and reused independently of any particular model.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATE_COLUMN,
    FEATURED_DATA_FILE,
    LAG_DAYS,
    LOG_FORMAT,
    LOG_LEVEL,
    ROLLING_WINDOWS,
    TARGET_COLUMN,
)
from src.data.clean_data import CLEANED_OUTPUT_FILE

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Columns the cleaned dataset must contain before feature engineering
# can safely proceed. Declared once so validation and documentation
# stay in sync.
REQUIRED_INPUT_COLUMNS: list[str] = [
    "Store",
    DATE_COLUMN,
    TARGET_COLUMN,
    "Open",
    "Promo2",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "PromoInterval",
]


def validate_input_data(df: pd.DataFrame) -> None:
    """Validate that the dataset is ready for feature engineering.

    Args:
        df: Cleaned DataFrame, as produced by ``clean_data.py``.

    Raises:
        ValueError: If required columns are missing, or if the date
            column cannot be interpreted as a datetime.
    """
    missing_columns = sorted(set(REQUIRED_INPUT_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Cannot build features: missing required column(s) {missing_columns}. "
            "Run clean_data.py first."
        )

    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COLUMN]):
        raise ValueError(
            f"'{DATE_COLUMN}' column must be a datetime dtype before feature "
            f"engineering; got {df[DATE_COLUMN].dtype}. Run clean_data.py first."
        )

    logger.info("Input data validated for feature engineering (%d rows)", len(df))


def sort_by_store_and_date(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the dataset by store and date, ascending.

    Lag and rolling-window features are only meaningful if each store's
    rows are in strict chronological order — an unsorted frame would
    silently produce lags/rolling stats that mix unrelated time points.

    Args:
        df: DataFrame to sort.

    Returns:
        DataFrame sorted by ``Store`` then ``Date``, with a fresh
        index.
    """
    df = df.sort_values(["Store", DATE_COLUMN]).reset_index(drop=True)
    logger.info("Sorted data by Store and %s", DATE_COLUMN)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-derived features capturing seasonality patterns.

    Retail demand is strongly seasonal at multiple scales: day-of-week
    (weekend footfall differs from weekdays), week-of-year (holiday
    season build-up), and month boundaries (salary-driven shopping
    spikes around month start/end). These features let a tree-based
    model split on seasonality directly instead of inferring it
    indirectly from the raw date.

    Args:
        df: DataFrame containing the date column.

    Returns:
        DataFrame with added columns: ``Year``, ``Month``,
        ``WeekOfYear``, ``Day``, ``Weekday``, ``IsWeekend``,
        ``IsMonthStart``, ``IsMonthEnd``.
    """
    df = df.copy()
    date_col = df[DATE_COLUMN]

    df["Year"] = date_col.dt.year
    df["Month"] = date_col.dt.month
    df["WeekOfYear"] = date_col.dt.isocalendar().week.astype("int32")
    df["Day"] = date_col.dt.day

    # Weekday: 0=Monday..6=Sunday, derived directly from Date rather than
    # trusting the dataset's own 'DayOfWeek' column, so this feature set
    # is self-consistent even if used on new/unseen data without that column.
    df["Weekday"] = date_col.dt.dayofweek
    df["IsWeekend"] = (df["Weekday"] >= 5).astype("int8")

    # Month start/end: salary disbursement and bill cycles commonly
    # concentrate discretionary retail spending around these boundaries.
    df["IsMonthStart"] = date_col.dt.is_month_start.astype("int8")
    df["IsMonthEnd"] = date_col.dt.is_month_end.astype("int8")

    logger.info("Added calendar features")
    return df


def add_competition_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a feature measuring how long a nearby competitor has been open.

    A newly opened competitor typically causes an initial sales dip that
    partially recovers as customers settle into new habits; a long-
    established competitor's effect is usually already "baked into"
    baseline sales. Duration-since-opening lets the model distinguish
    these regimes instead of only knowing that competition exists.

    Args:
        df: DataFrame with ``CompetitionOpenSinceMonth``,
            ``CompetitionOpenSinceYear``, ``Year``, and ``Month``
            columns.

    Returns:
        DataFrame with an added ``CompetitionOpenMonths`` column
        (months elapsed since competitor opened; 0 where no
        competition-open date is known, matching the sentinel used
        during cleaning).
    """
    df = df.copy()

    months_elapsed = (df["Year"] - df["CompetitionOpenSinceYear"]) * 12 + (
        df["Month"] - df["CompetitionOpenSinceMonth"]
    )

    # A sentinel of 0/0 (no known competition-open date, set during
    # cleaning) would otherwise produce a large, meaningless positive
    # number here. Treat that case explicitly as "no competition
    # duration effect" rather than a real elapsed time.
    no_known_competition_date = (df["CompetitionOpenSinceYear"] == 0) | (
        df["CompetitionOpenSinceMonth"] == 0
    )
    months_elapsed = months_elapsed.where(~no_known_competition_date, 0)

    # Negative values would occur if a competition-open date is recorded
    # in the future relative to the current row (data artifact); clip
    # to 0 rather than allowing a nonsensical negative duration.
    df["CompetitionOpenMonths"] = months_elapsed.clip(lower=0).astype("int32")

    logger.info("Added competition duration feature (CompetitionOpenMonths)")
    return df


def add_promo_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features capturing recurring-promotion timing and duration.

    Promo2 is a recurring campaign (e.g. relaunched every few months in
    a fixed set of calendar months) rather than a one-off event. Two
    signals matter for demand: (1) how long the store has been enrolled
    in the program (promo effects often taper with novelty), and (2)
    whether the *current* month is one of the store's active promo
    months at all — a store not in an active promo month should not
    receive a promo-driven demand boost in the model's reasoning.

    Args:
        df: DataFrame with ``Promo2``, ``Promo2SinceWeek``,
            ``Promo2SinceYear``, ``PromoInterval``, ``Year``,
            ``WeekOfYear``, and ``Month`` columns.

    Returns:
        DataFrame with added columns ``Promo2DurationWeeks`` and
        ``IsPromoMonth``.
    """
    df = df.copy()

    weeks_elapsed = (df["Year"] - df["Promo2SinceYear"]) * 52 + (
        df["WeekOfYear"] - df["Promo2SinceWeek"]
    )

    # Stores not enrolled in Promo2 (or with the 0/0 sentinel from
    # cleaning) have no meaningful "duration since start" — force to 0
    # rather than propagating a spurious large number.
    not_enrolled = (df["Promo2"] == 0) | (df["Promo2SinceYear"] == 0) | (
        df["Promo2SinceWeek"] == 0
    )
    weeks_elapsed = weeks_elapsed.where(~not_enrolled, 0)
    df["Promo2DurationWeeks"] = weeks_elapsed.clip(lower=0).astype("int32")

    # PromoInterval lists the recurring months a store's Promo2 is active
    # in (e.g. "Jan,Apr,Jul,Oct"). Precompute a set of active months per
    # unique interval string once, then map it across rows — far cheaper
    # than re-parsing the string on every row.
    # fillna before astype(str): on some pandas versions astype(str) leaves
    # NaN as a float rather than the string "nan", which would break the
    # dict lookup below. Filling first guarantees every value is a
    # genuine string to split or match against "None".
    interval_series = df["PromoInterval"].fillna("None").astype(str)
    interval_month_sets = {
        value: set(value.split(",")) if value not in ("None", "") else set()
        for value in interval_series.unique()
    }
    month_abbreviations = df[DATE_COLUMN].dt.strftime("%b")
    active_months_per_row = interval_series.map(interval_month_sets)

    df["IsPromoMonth"] = [
        1 if (is_enrolled == 1 and month in active_months) else 0
        for is_enrolled, month, active_months in zip(
            df["Promo2"], month_abbreviations, active_months_per_row
        )
    ]
    df["IsPromoMonth"] = df["IsPromoMonth"].astype("int8")

    logger.info("Added promo duration features (Promo2DurationWeeks, IsPromoMonth)")
    return df


def add_lag_features(
    df: pd.DataFrame, lags: list[int] = LAG_DAYS, target_column: str = TARGET_COLUMN
) -> pd.DataFrame:
    """Add per-store lagged sales features.

    Retail demand is strongly autocorrelated — yesterday's, last week's,
    and last month's sales are among the best predictors of today's
    sales. Lags are computed per store (via ``groupby``) so that one
    store's history never leaks into another store's lag values.

    Args:
        df: DataFrame sorted by ``Store`` then date, containing the
            target column.
        lags: List of lag offsets in days. Defaults to
            ``src.config.LAG_DAYS``.
        target_column: Column to lag. Defaults to
            ``src.config.TARGET_COLUMN``.

    Returns:
        DataFrame with one new column per lag: ``SalesLag_{n}``.
    """
    df = df.copy()
    grouped_target = df.groupby("Store")[target_column]

    for lag in lags:
        df[f"{target_column}Lag_{lag}"] = grouped_target.shift(lag)

    logger.info("Added lag features for lags=%s", lags)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] = ROLLING_WINDOWS,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Add per-store rolling mean and standard deviation features.

    Rolling mean captures the recent demand *level* (short-term trend),
    while rolling standard deviation captures recent demand *volatility*
    — a store with high recent variance is a stronger candidate for
    safety-stock buffers than one with the same average but stable
    demand. Values are shifted by one day before rolling so that the
    statistic for a given day never includes that day's own sales
    (which would leak the prediction target into its own feature).

    Args:
        df: DataFrame sorted by ``Store`` then date, containing the
            target column.
        windows: List of rolling window sizes in days. Defaults to
            ``src.config.ROLLING_WINDOWS``.
        target_column: Column to compute rolling statistics on.
            Defaults to ``src.config.TARGET_COLUMN``.

    Returns:
        DataFrame with two new columns per window:
        ``SalesRollingMean_{n}`` and ``SalesRollingStd_{n}``.
    """
    df = df.copy()
    grouped_target = df.groupby("Store")[target_column]

    for window in windows:
        # shift(1) excludes the current day from its own rolling window —
        # without this, the feature would directly encode part of the
        # value the model is trying to predict (target leakage).
        shifted = grouped_target.transform(lambda s: s.shift(1))
        df[f"{target_column}RollingMean_{window}"] = shifted.groupby(
            df["Store"]
        ).transform(lambda s: s.rolling(window, min_periods=1).mean())
        df[f"{target_column}RollingStd_{window}"] = shifted.groupby(
            df["Store"]
        ).transform(lambda s: s.rolling(window, min_periods=1).std())

    logger.info("Added rolling mean/std features for windows=%s", windows)
    return df


def add_log_sales(df: pd.DataFrame, column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Add a log1p-transformed version of the target column.

    Retail sales distributions are right-skewed (many typical days, a
    few very high-demand days). Modeling ``log1p(Sales)`` instead of raw
    ``Sales`` stabilizes variance and reduces the disproportionate
    influence of high-sales days on error-minimizing models — a common,
    well-justified transformation for this kind of target.
    ``log1p`` (rather than plain ``log``) is used specifically because
    ``Sales`` can legitimately be 0 (closed-store days), and ``log(0)``
    is undefined.

    Args:
        df: DataFrame containing the target column.
        column: Name of the column to transform. Defaults to
            ``src.config.TARGET_COLUMN``.

    Returns:
        DataFrame with an added ``{column}Log`` column.
    """
    df = df.copy()
    df[f"{column}Log"] = np.log1p(df[column])
    logger.info("Added log1p-transformed target column: %sLog", column)
    return df


def add_cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Add sine/cosine cyclical encodings for month and weekday.

    Calendar features are cyclical, not ordinal: December (12) is
    adjacent to January (1), and Sunday is adjacent to Monday, but a raw
    integer encoding implies December and January are the two most
    *distant* values. Sine/cosine encoding maps each cycle onto a circle
    so that adjacency in time is preserved as numerical closeness,
    which matters especially for distance-based or linear models.

    Args:
        df: DataFrame containing ``Month`` (1-12) and ``Weekday``
            (0-6) columns.

    Returns:
        DataFrame with added columns: ``MonthSin``, ``MonthCos``,
        ``WeekdaySin``, ``WeekdayCos``.
    """
    df = df.copy()

    df["MonthSin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["MonthCos"] = np.cos(2 * np.pi * df["Month"] / 12)

    df["WeekdaySin"] = np.sin(2 * np.pi * df["Weekday"] / 7)
    df["WeekdayCos"] = np.cos(2 * np.pi * df["Weekday"] / 7)

    logger.info("Added cyclical encodings for Month and Weekday")
    return df


def save_featured_data(df: pd.DataFrame, output_path: Path = FEATURED_DATA_FILE) -> None:
    """Persist the feature-engineered DataFrame to the processed data directory.

    Args:
        df: Feature-engineered DataFrame to save.
        output_path: Destination path. Defaults to
            ``src.config.FEATURED_DATA_FILE``.

    Raises:
        OSError: If the file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(output_path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write featured data to '{output_path}': {exc}") from exc
    logger.info("Saved featured data to '%s' (%s)", output_path, df.shape)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering sequence to a cleaned dataset.

    Args:
        df: Cleaned DataFrame, as produced by ``clean_data.py``.

    Returns:
        DataFrame enriched with calendar, competition, promo, lag,
        rolling, log-target, and cyclical-encoding features.

    Raises:
        ValueError: If the input data fails validation.
    """
    validate_input_data(df)

    df = sort_by_store_and_date(df)
    df = add_calendar_features(df)
    df = add_competition_duration_features(df)
    df = add_promo_duration_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_log_sales(df)
    df = add_cyclical_encoding(df)

    # Lag and rolling features are undefined for each store's earliest
    # rows (no prior history exists yet). These NaNs are left in place
    # deliberately rather than dropped here: removing rows is a
    # train/test-splitting decision, not a feature-construction one, and
    # belongs in the modeling stage where the usable date range is
    # actually decided.
    n_incomplete_history_rows = int(
        df[[f"{TARGET_COLUMN}Lag_{lag}" for lag in LAG_DAYS]].isnull().any(axis=1).sum()
    )
    logger.info(
        "%d row(s) have incomplete lag history (expected for each store's "
        "earliest dates); left as NaN for the modeling stage to handle",
        n_incomplete_history_rows,
    )

    logger.info("Feature engineering complete. Final shape: %s", df.shape)
    return df


def run_feature_engineering(
    input_path: Path = CLEANED_OUTPUT_FILE,
    output_path: Path = FEATURED_DATA_FILE,
    save_output: bool = True,
) -> pd.DataFrame:
    """Orchestrate the full feature engineering workflow end-to-end.

    Loads the cleaned dataset, applies all feature engineering steps,
    and optionally persists the result.

    Args:
        input_path: Path to the cleaned dataset produced by
            ``clean_data.py``.
        output_path: Destination path for the featured dataset.
        save_output: Whether to persist the featured DataFrame to disk.

    Returns:
        The feature-engineered DataFrame.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ValueError: If the input data fails validation.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned data file not found: '{input_path}'. Run clean_data.py first."
        )

    df = pd.read_csv(input_path, parse_dates=[DATE_COLUMN], low_memory=False)
    df = build_features(df)

    if save_output:
        save_featured_data(df, output_path)

    return df


if __name__ == "__main__":
    # Manual smoke-test entry point: `python -m src.features.build_features`
    featured_data = run_feature_engineering()
    logger.info("Smoke test passed. featured_data shape=%s", featured_data.shape)
