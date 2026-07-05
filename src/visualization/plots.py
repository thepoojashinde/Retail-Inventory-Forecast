"""Reusable Plotly visualization functions for retail sales forecasting.

Every function in this module builds and RETURNS a ``plotly.graph_objects
.Figure`` -- none of them call ``.show()``, write files, or depend on a
UI framework. This keeps the module usable from a notebook, a script, a
web backend, or a Streamlit/Dash app (the caller decides how to render
the returned figure) without coupling this project to any specific
front end.

None of these functions load data, train models, or duplicate metric
computation: callers are expected to pass in already-prepared
DataFrames, arrays, or fitted model objects (e.g. from
``train_model.py`` / ``evaluate_model.py``), consistent with this
module's single responsibility -- visualization only.

Do NOT modify:
    - src/config.py
    - src/data/load_data.py
    - src/data/merge_data.py
    - src/data/clean_data.py
    - src/features/build_features.py
    - src/models/train_model.py
    - src/models/evaluate_model.py
    - src/models/predict_model.py
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------- #
# Project imports.
#
# Column names are pulled from config.py so this module never hardcodes
# a schema that could drift from the rest of the pipeline. Names that
# are not certain to exist in config.py yet are guarded with getattr
# and flagged for renaming.
# --------------------------------------------------------------------------- #
from src import config  # TODO: Match this import path to your package layout if different

try:
    DATE_COLUMN: str = config.DATE_COLUMN  # TODO: Match this constant name to config.py
except AttributeError as exc:
    raise ImportError("config.py must define DATE_COLUMN, e.g. 'Date'.") from exc

try:
    STORE_COLUMN: str = config.STORE_COLUMN  # TODO: Match this constant name to config.py
except AttributeError as exc:
    raise ImportError("config.py must define STORE_COLUMN, e.g. 'Store'.") from exc

try:
    TARGET_COLUMN: str = config.TARGET_COLUMN  # TODO: Match this constant name to config.py
except AttributeError as exc:
    raise ImportError("config.py must define TARGET_COLUMN, e.g. 'Sales'.") from exc

PROMO_COLUMN: str = getattr(config, "PROMO_COLUMN", "Promo")  # TODO: Match this constant name to config.py
STORE_TYPE_COLUMN: str = getattr(
    config, "STORE_TYPE_COLUMN", "StoreType"  # TODO: Match this constant name to config.py
)
ASSORTMENT_COLUMN: str = getattr(
    config, "ASSORTMENT_COLUMN", "Assortment"  # TODO: Match this constant name to config.py
)
DEFAULT_TOP_N_STORES: int = getattr(config, "TOP_N_STORES", 10)  # TODO: Match this constant name to config.py

logger = logging.getLogger(__name__)

# A shared template keeps every figure in the app visually consistent
# without repeating layout kwargs in every single plotting function.
_PLOTLY_TEMPLATE = "plotly_white"


def _validate_columns_exist(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Validate that required columns exist in a DataFrame.

    Centralizing this check means every plotting function fails with the
    same clear, actionable error message instead of an opaque KeyError
    raised deep inside a groupby or plotly call.

    Args:
        df: DataFrame to validate.
        required_columns: Column names that must be present.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) for this plot: {missing}")


def _aggregate_sales_over_time(
    df: pd.DataFrame,
    freq: str,
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Resample and sum sales over a given time frequency.

    A shared helper backs the monthly/weekly/daily trend functions
    because they differ only in the resampling frequency string -- three
    near-identical implementations would be a maintenance hazard if the
    aggregation logic (e.g. sum vs. mean) ever needs to change.

    Args:
        df: Input dataset containing date and target columns.
        freq: Pandas offset alias (e.g. "D", "W", "M").
        date_col: Name of the date column.
        target_col: Name of the sales/target column.

    Returns:
        A DataFrame with columns [date_col, target_col] aggregated at
        the requested frequency, sorted chronologically.
    """
    _validate_columns_exist(df, [date_col, target_col])
    working_df = df[[date_col, target_col]].copy()
    working_df[date_col] = pd.to_datetime(working_df[date_col])

    aggregated = (
        working_df.set_index(date_col)
        .resample(freq)[target_col]
        .sum()
        .reset_index()
        .sort_values(date_col)
    )
    return aggregated


def _sales_trend_figure(aggregated: pd.DataFrame, date_col: str, target_col: str, title: str) -> go.Figure:
    """Build a line chart figure from pre-aggregated time-series sales.

    Args:
        aggregated: DataFrame with [date_col, target_col] columns.
        date_col: Name of the date column.
        target_col: Name of the sales/target column.
        title: Figure title.

    Returns:
        A Plotly line chart Figure.
    """
    fig = go.Figure(
        data=go.Scatter(
            x=aggregated[date_col],
            y=aggregated[target_col],
            mode="lines",
            line=dict(width=2),
            name="Sales",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Total Sales",
        template=_PLOTLY_TEMPLATE,
    )
    return fig


def plot_monthly_sales_trend(
    df: pd.DataFrame, date_col: str = DATE_COLUMN, target_col: str = TARGET_COLUMN
) -> go.Figure:
    """Plot total sales aggregated by month.

    Args:
        df: Dataset containing date and target columns.
        date_col: Name of the date column.
        target_col: Name of the sales/target column.

    Returns:
        A Plotly line chart Figure of monthly sales totals.
    """
    aggregated = _aggregate_sales_over_time(df, freq="ME", date_col=date_col, target_col=target_col)
    return _sales_trend_figure(aggregated, date_col, target_col, title="Monthly Sales Trend")


def plot_weekly_sales_trend(
    df: pd.DataFrame, date_col: str = DATE_COLUMN, target_col: str = TARGET_COLUMN
) -> go.Figure:
    """Plot total sales aggregated by week.

    Args:
        df: Dataset containing date and target columns.
        date_col: Name of the date column.
        target_col: Name of the sales/target column.

    Returns:
        A Plotly line chart Figure of weekly sales totals.
    """
    aggregated = _aggregate_sales_over_time(df, freq="W", date_col=date_col, target_col=target_col)
    return _sales_trend_figure(aggregated, date_col, target_col, title="Weekly Sales Trend")


def plot_daily_sales_trend(
    df: pd.DataFrame, date_col: str = DATE_COLUMN, target_col: str = TARGET_COLUMN
) -> go.Figure:
    """Plot total sales aggregated by day.

    Args:
        df: Dataset containing date and target columns.
        date_col: Name of the date column.
        target_col: Name of the sales/target column.

    Returns:
        A Plotly line chart Figure of daily sales totals.
    """
    aggregated = _aggregate_sales_over_time(df, freq="D", date_col=date_col, target_col=target_col)
    return _sales_trend_figure(aggregated, date_col, target_col, title="Daily Sales Trend")


def plot_promotion_impact(
    df: pd.DataFrame, promo_col: str = PROMO_COLUMN, target_col: str = TARGET_COLUMN
) -> go.Figure:
    """Plot a box plot comparing sales distributions with vs. without promotion.

    A box plot (rather than a simple bar of means) is used because it
    also shows spread and outliers, which matters for deciding whether
    an observed promo uplift is a robust effect or driven by a few
    unusually high-sales days.

    Args:
        df: Dataset containing the promotion flag and target columns.
        promo_col: Name of the binary promotion indicator column.
        target_col: Name of the sales/target column.

    Returns:
        A Plotly box plot Figure grouped by promotion status.
    """
    _validate_columns_exist(df, [promo_col, target_col])

    fig = go.Figure()
    for promo_value, label in ((0, "No Promotion"), (1, "Promotion")):
        subset = df[df[promo_col] == promo_value][target_col]
        fig.add_trace(go.Box(y=subset, name=label, boxmean=True))

    fig.update_layout(
        title="Promotion Impact on Sales",
        yaxis_title="Sales",
        template=_PLOTLY_TEMPLATE,
    )
    return fig


def _plot_sales_by_category(df: pd.DataFrame, category_col: str, target_col: str, title: str) -> go.Figure:
    """Build a bar chart of average sales grouped by a categorical column.

    Shared by store-type and assortment breakdowns since both are the
    same "group, average, bar chart" operation applied to a different
    column.

    Args:
        df: Input dataset.
        category_col: Categorical column to group by.
        target_col: Name of the sales/target column.
        title: Figure title.

    Returns:
        A Plotly bar chart Figure of mean sales per category.
    """
    _validate_columns_exist(df, [category_col, target_col])

    grouped = (
        df.groupby(category_col)[target_col]
        .mean()
        .reset_index()
        .sort_values(target_col, ascending=False)
    )

    fig = go.Figure(
        data=go.Bar(x=grouped[category_col].astype(str), y=grouped[target_col])
    )
    fig.update_layout(
        title=title,
        xaxis_title=category_col,
        yaxis_title="Average Sales",
        template=_PLOTLY_TEMPLATE,
    )
    return fig


def plot_sales_by_store_type(
    df: pd.DataFrame, store_type_col: str = STORE_TYPE_COLUMN, target_col: str = TARGET_COLUMN
) -> go.Figure:
    """Plot average sales by store type.

    Args:
        df: Dataset containing store type and target columns.
        store_type_col: Name of the store type column.
        target_col: Name of the sales/target column.

    Returns:
        A Plotly bar chart Figure.
    """
    return _plot_sales_by_category(df, store_type_col, target_col, title="Average Sales by Store Type")


def plot_sales_by_assortment(
    df: pd.DataFrame, assortment_col: str = ASSORTMENT_COLUMN, target_col: str = TARGET_COLUMN
) -> go.Figure:
    """Plot average sales by assortment type.

    Args:
        df: Dataset containing assortment and target columns.
        assortment_col: Name of the assortment column.
        target_col: Name of the sales/target column.

    Returns:
        A Plotly bar chart Figure.
    """
    return _plot_sales_by_category(df, assortment_col, target_col, title="Average Sales by Assortment")


def plot_correlation_heatmap(df: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> go.Figure:
    """Plot a correlation heatmap for numeric features.

    Args:
        df: Input dataset.
        numeric_columns: Optional explicit list of columns to include.
            If omitted, all numeric columns in ``df`` are used, since
            correlation is only defined for numeric data.

    Returns:
        A Plotly heatmap Figure of pairwise Pearson correlations.

    Raises:
        ValueError: If fewer than two numeric columns are available.
    """
    if numeric_columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        _validate_columns_exist(df, numeric_columns)
        numeric_df = df[numeric_columns]

    if numeric_df.shape[1] < 2:
        raise ValueError("At least two numeric columns are required to compute a correlation heatmap.")

    corr_matrix = numeric_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmid=0,  # Center the colorscale at zero so positive/negative correlations are visually symmetric.
            colorbar=dict(title="Correlation"),
        )
    )
    fig.update_layout(title="Feature Correlation Heatmap", template=_PLOTLY_TEMPLATE)
    return fig


def plot_top_performing_stores(
    df: pd.DataFrame,
    store_col: str = STORE_COLUMN,
    target_col: str = TARGET_COLUMN,
    top_n: int = DEFAULT_TOP_N_STORES,
) -> go.Figure:
    """Plot the top-performing stores by total sales.

    Total (not average) sales is used as the ranking metric because it
    reflects overall business contribution, which is usually what
    "top performing" means in a retail context; a low-traffic store
    with a high average ticket could otherwise misleadingly outrank a
    high-volume flagship store.

    Args:
        df: Dataset containing store and target columns.
        store_col: Name of the store identifier column.
        target_col: Name of the sales/target column.
        top_n: Number of top stores to display.

    Returns:
        A Plotly bar chart Figure of the top N stores by total sales.

    Raises:
        ValueError: If top_n is not a positive integer.
    """
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")
    _validate_columns_exist(df, [store_col, target_col])

    top_stores = (
        df.groupby(store_col)[target_col]
        .sum()
        .reset_index()
        .sort_values(target_col, ascending=False)
        .head(top_n)
    )

    fig = go.Figure(
        data=go.Bar(x=top_stores[store_col].astype(str), y=top_stores[target_col])
    )
    fig.update_layout(
        title=f"Top {top_n} Performing Stores",
        xaxis_title="Store",
        yaxis_title="Total Sales",
        template=_PLOTLY_TEMPLATE,
    )
    return fig


def extract_feature_importance(
    model: Pipeline,
    preprocessor_step_name: str = "preprocessor",
    regressor_step_name: str = "regressor",
) -> Tuple[List[str], np.ndarray]:
    """Extract feature names and importances from a fitted pipeline.

    This reuses the already-fitted pipeline object produced by
    ``train_model.py`` rather than recomputing or re-deriving feature
    names independently, which could silently drift out of sync with
    what the model actually saw.

    Args:
        model: A fitted sklearn Pipeline containing a ColumnTransformer
            step and a final regressor step.
        preprocessor_step_name: Name of the preprocessing step in the
            pipeline.
        regressor_step_name: Name of the final estimator step in the
            pipeline.

    Returns:
        A tuple of (feature_names, importances).

    Raises:
        ValueError: If the pipeline's regressor exposes neither
            ``feature_importances_`` nor ``coef_``, or if the named
            steps are not found in the pipeline.
    """
    try:
        preprocessor = model.named_steps[preprocessor_step_name]
        regressor = model.named_steps[regressor_step_name]
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"Model must be a fitted Pipeline with steps "
            f"'{preprocessor_step_name}' and '{regressor_step_name}'."
        ) from exc

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception as exc:  # pragma: no cover - depends on sklearn version/transformer support
        raise ValueError(
            "Could not extract feature names from the preprocessor; "
            "ensure it implements get_feature_names_out()."
        ) from exc

    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    elif hasattr(regressor, "coef_"):
        # Linear models expose coefficients, not importances; absolute
        # value is used so the plot ranks by magnitude of effect,
        # matching how tree-based importances are interpreted.
        importances = np.abs(np.ravel(regressor.coef_))
    else:
        raise ValueError(
            f"Regressor of type {type(regressor).__name__} exposes neither "
            "'feature_importances_' nor 'coef_'."
        )

    return feature_names, np.asarray(importances)


def plot_feature_importance(
    feature_names: Sequence[str], importances: Sequence[float], top_n: int = 20
) -> go.Figure:
    """Plot a horizontal bar chart of feature importances.

    Args:
        feature_names: Names corresponding to each importance value.
        importances: Importance (or absolute coefficient) values.
        top_n: Number of top features to display.

    Returns:
        A Plotly horizontal bar chart Figure, sorted ascending so the
        most important feature renders at the top.

    Raises:
        ValueError: If feature_names and importances have different
            lengths, or top_n is not positive.
    """
    if len(feature_names) != len(importances):
        raise ValueError("feature_names and importances must have the same length.")
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)  # ascending so Plotly renders the largest bar on top
    )

    fig = go.Figure(
        data=go.Bar(
            x=importance_df["importance"],
            y=importance_df["feature"],
            orientation="h",
        )
    )
    fig.update_layout(
        title=f"Top {min(top_n, len(feature_names))} Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template=_PLOTLY_TEMPLATE,
    )
    return fig


def plot_actual_vs_predicted(y_true: Sequence[float], y_pred: Sequence[float]) -> go.Figure:
    """Plot actual vs. predicted values with a y=x reference line.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        A Plotly scatter Figure with an ideal-fit reference line.

    Raises:
        ValueError: If y_true and y_pred have different lengths.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have the same length.")

    lower = float(min(y_true_arr.min(), y_pred_arr.min()))
    upper = float(max(y_true_arr.max(), y_pred_arr.max()))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true_arr,
            y=y_pred_arr,
            mode="markers",
            marker=dict(size=5, opacity=0.4),
            name="Predictions",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Ideal fit (y = x)",
        )
    )
    fig.update_layout(
        title="Actual vs Predicted",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        template=_PLOTLY_TEMPLATE,
    )
    return fig


def plot_residual_distribution(y_true: Sequence[float], y_pred: Sequence[float]) -> go.Figure:
    """Plot a histogram of residuals (actual - predicted).

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        A Plotly histogram Figure of residuals.

    Raises:
        ValueError: If y_true and y_pred have different lengths.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have the same length.")

    residuals = y_true_arr - y_pred_arr

    fig = go.Figure(data=go.Histogram(x=residuals, nbinsx=50))
    fig.add_vline(x=0, line_color="red", line_dash="dash")
    fig.update_layout(
        title="Residual Distribution",
        xaxis_title="Residual",
        yaxis_title="Frequency",
        template=_PLOTLY_TEMPLATE,
    )
    return fig


def plot_kpi_cards(metrics: Dict[str, float]) -> go.Figure:
    """Plot a row of KPI indicator cards for headline metrics.

    Plotly's ``Indicator`` trace (rather than a bar/line chart) is used
    because KPI cards are meant to be read as single at-a-glance numbers,
    not compared visually along an axis.

    Args:
        metrics: Mapping of metric name to value, e.g.
            {"RMSE": 812.4, "MAE": 601.2, "MAPE": 0.11, "R2": 0.87}.

    Returns:
        A Plotly Figure containing one indicator per metric, arranged
        in a single row.

    Raises:
        ValueError: If metrics is empty.
    """
    if not metrics:
        raise ValueError("metrics must contain at least one entry to build KPI cards.")

    n_metrics = len(metrics)
    # Each indicator gets an equal horizontal slice of the [0, 1] domain
    # so the cards render evenly spaced regardless of how many there are.
    domain_width = 1.0 / n_metrics

    fig = go.Figure()
    for i, (name, value) in enumerate(metrics.items()):
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=value,
                title={"text": name},
                domain={"x": [i * domain_width, (i + 1) * domain_width], "y": [0, 1]},
            )
        )

    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        grid={"rows": 1, "columns": n_metrics, "pattern": "independent"},
        title="Key Performance Indicators",
    )
    return fig
