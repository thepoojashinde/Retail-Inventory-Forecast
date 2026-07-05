"""Streamlit dashboard for the Retail Sales Forecasting project.

This app is a thin presentation layer only: every data load, model
load, prediction, and metric computation is delegated to the existing
project modules (``build_features``, ``train_model``, ``evaluate_model``,
``predict_model``, ``plots``). No training, preprocessing, or metric
logic is implemented here -- this file only arranges Streamlit widgets
around calls into those modules.

Run with::

    streamlit run app.py

Do NOT modify:
    - src/config.py
    - src/data/load_data.py
    - src/data/merge_data.py
    - src/data/clean_data.py
    - src/features/build_features.py
    - src/models/train_model.py
    - src/models/evaluate_model.py
    - src/models/predict_model.py
    - src/visualization/plots.py
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Project imports -- all business logic lives in these modules already.
# app.py only calls into them and renders the results.
# --------------------------------------------------------------------------- #
from src.models.train_model import (
    DATE_COLUMN,
    MODEL_DIR,
    STORE_COLUMN,
    TARGET_COLUMN,
    load_feature_dataset,
)
from src.models.evaluate_model import (
    EVALUATION_METRICS_FILENAME,
    load_trained_model,
    load_validation_split,
    generate_predictions,
)
from src.models.predict_model import (
    PREDICTIONS_OUTPUT_PATH,
    REQUIRED_INFERENCE_COLUMNS,
    run_prediction_pipeline,
)
from src.visualization import plots

logger = logging.getLogger(__name__)

PAGE_HOME = "🏠 Home"
PAGE_EDA = "📊 Exploratory Data Analysis"
PAGE_MODEL_PERFORMANCE = "🤖 Model Performance"
PAGE_PREDICTIONS = "📈 Predictions"
PAGE_INSIGHTS = "💼 Business Insights"
PAGE_ABOUT = "ℹ️ About"

PAGES = [PAGE_HOME, PAGE_EDA, PAGE_MODEL_PERFORMANCE, PAGE_PREDICTIONS, PAGE_INSIGHTS, PAGE_ABOUT]


# --------------------------------------------------------------------------- #
# Cached data/model accessors.
#
# Streamlit reruns the whole script on every interaction, so loading the
# dataset or the model without caching would re-read from disk on every
# click. st.cache_data/st.cache_resource make these effectively
# singletons for the life of the session, without changing any of the
# underlying loading logic (still delegated to train_model/evaluate_model).
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_feature_dataset() -> pd.DataFrame:
    """Load the feature-engineered dataset via train_model's loader.

    Returns:
        The cached feature dataset.
    """
    return load_feature_dataset()


@st.cache_resource(show_spinner=False)
def get_trained_model():
    """Load the persisted best model via evaluate_model's loader.

    Returns:
        The cached, fitted sklearn pipeline.
    """
    return load_trained_model()


@st.cache_data(show_spinner=False)
def get_validation_predictions() -> pd.DataFrame:
    """Build a small DataFrame of validation actuals vs. predictions.

    Reuses evaluate_model's existing split-loading and prediction
    helpers rather than reimplementing validation-set scoring here.

    Returns:
        DataFrame with 'actual' and 'predicted' columns.
    """
    model = get_trained_model()
    split = load_validation_split()
    predictions = generate_predictions(model, split.X_val)
    return pd.DataFrame({"actual": split.y_val.values, "predicted": predictions})


def load_evaluation_metrics(model_dir: Path = MODEL_DIR) -> Dict[str, float]:
    """Load the evaluation metrics JSON previously saved by evaluate_model.py.

    Args:
        model_dir: Directory containing the evaluation metrics file.

    Returns:
        Dictionary of metric name to value.

    Raises:
        FileNotFoundError: If evaluate_model.py has not been run yet.
    """
    metrics_path = Path(model_dir) / EVALUATION_METRICS_FILENAME
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Evaluation metrics not found at '{metrics_path}'. Run evaluate_model.py first."
        )
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def configure_page() -> None:
    """Apply global Streamlit page configuration and light styling."""
    st.set_page_config(
        page_title="Retail Sales Forecasting",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # A small CSS tweak for tighter, more "dashboard-like" metric cards;
    # kept minimal and inline so the app has zero extra static assets.
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background-color: rgba(240, 242, 246, 0.6);
            border-radius: 8px;
            padding: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render the sidebar navigation and return the selected page.

    Returns:
        The name of the page selected by the user.
    """
    st.sidebar.title("🛒 Sales Forecasting")
    st.sidebar.caption("Retail Sales Forecasting Dashboard")
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("Built with Streamlit, scikit-learn & Plotly")
    return page


# --------------------------------------------------------------------------- #
# Page renderers.
# --------------------------------------------------------------------------- #
def render_home_page() -> None:
    """Render the Home page: overview, business problem, dataset summary, KPIs."""
    st.title("🛒 Retail Sales Forecasting")
    st.subheader("Predicting daily store sales to support inventory and staffing decisions")

    with st.expander("📌 Business Problem", expanded=True):
        st.markdown(
            "Retail chains need reliable short-term sales forecasts per store to plan "
            "inventory, staffing, and promotions. Under-forecasting leads to stockouts; "
            "over-forecasting ties up capital in excess inventory. This project builds "
            "an end-to-end pipeline -- from raw transactional data to a tuned regression "
            "model -- that predicts daily sales at the store level."
        )

    try:
        df = get_feature_dataset()
    except FileNotFoundError as exc:
        st.error(f"Could not load the feature dataset: {exc}")
        return

    st.markdown("### Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Stores", f"{df[STORE_COLUMN].nunique():,}" if STORE_COLUMN in df.columns else "N/A")
    if DATE_COLUMN in df.columns:
        date_series = pd.to_datetime(df[DATE_COLUMN])
        col3.metric("Date Range", f"{date_series.min().date()} → {date_series.max().date()}")
    else:
        col3.metric("Date Range", "N/A")
    col4.metric("Features", f"{df.shape[1]:,}")

    st.markdown("### Key Performance Indicators")
    try:
        metrics = load_evaluation_metrics()
        st.plotly_chart(plots.plot_kpi_cards(metrics), use_container_width=True)
    except FileNotFoundError as exc:
        st.info(f"KPIs unavailable: {exc}")

    with st.expander("Preview raw feature data"):
        st.dataframe(df.head(50), use_container_width=True)


def render_eda_page() -> None:
    """Render the Exploratory Data Analysis page."""
    st.title("📊 Exploratory Data Analysis")

    try:
        df = get_feature_dataset()
    except FileNotFoundError as exc:
        st.error(f"Could not load the feature dataset: {exc}")
        return

    tab_trends, tab_promo, tab_store, tab_corr, tab_top = st.tabs(
        ["Sales Trends", "Promotion Impact", "Store Type", "Correlation", "Top Stores"]
    )

    with tab_trends:
        trend_choice = st.radio(
            "Aggregation level", ["Monthly", "Weekly", "Daily"], horizontal=True
        )
        with st.spinner("Building trend chart..."):
            if trend_choice == "Monthly":
                fig = plots.plot_monthly_sales_trend(df)
            elif trend_choice == "Weekly":
                fig = plots.plot_weekly_sales_trend(df)
            else:
                fig = plots.plot_daily_sales_trend(df)
        st.plotly_chart(fig, use_container_width=True)

    with tab_promo:
        try:
            with st.spinner("Building promotion impact chart..."):
                fig = plots.plot_promotion_impact(df)
            st.plotly_chart(fig, use_container_width=True)
        except ValueError as exc:
            st.warning(f"Promotion impact chart unavailable: {exc}")

    with tab_store:
        col_left, col_right = st.columns(2)
        with col_left:
            try:
                fig = plots.plot_sales_by_store_type(df)
                st.plotly_chart(fig, use_container_width=True)
            except ValueError as exc:
                st.warning(f"Store type chart unavailable: {exc}")
        with col_right:
            try:
                fig = plots.plot_sales_by_assortment(df)
                st.plotly_chart(fig, use_container_width=True)
            except ValueError as exc:
                st.warning(f"Assortment chart unavailable: {exc}")

    with tab_corr:
        try:
            fig = plots.plot_correlation_heatmap(df)
            st.plotly_chart(fig, use_container_width=True)
        except ValueError as exc:
            st.warning(f"Correlation heatmap unavailable: {exc}")

    with tab_top:
        top_n = st.slider("Number of stores to display", min_value=5, max_value=30, value=10)
        try:
            fig = plots.plot_top_performing_stores(df, top_n=top_n)
            st.plotly_chart(fig, use_container_width=True)
        except ValueError as exc:
            st.warning(f"Top stores chart unavailable: {exc}")


def render_model_performance_page() -> None:
    """Render the Model Performance page."""
    st.title("🤖 Model Performance")

    try:
        metrics = load_evaluation_metrics()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.markdown("### Evaluation Metrics")
    st.plotly_chart(plots.plot_kpi_cards(metrics), use_container_width=True)

    try:
        with st.spinner("Loading validation predictions..."):
            val_df = get_validation_predictions()
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        st.error(f"Could not generate validation predictions: {exc}")
        return

    col_left, col_right = st.columns(2)
    with col_left:
        fig = plots.plot_actual_vs_predicted(val_df["actual"], val_df["predicted"])
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        fig = plots.plot_residual_distribution(val_df["actual"], val_df["predicted"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature Importance")
    try:
        model = get_trained_model()
        feature_names, importances = plots.extract_feature_importance(model)
        fig = plots.plot_feature_importance(feature_names, importances)
        st.plotly_chart(fig, use_container_width=True)
    except ValueError as exc:
        st.info(f"Feature importance unavailable for this model: {exc}")


def render_predictions_page() -> None:
    """Render the Predictions page: upload, predict, view, download."""
    st.title("📈 Predictions")
    st.markdown(
        "Upload a CSV of new, unseen store-day rows to generate sales predictions "
        "using the trained production model."
    )
    st.caption(f"Required columns: {', '.join(REQUIRED_INFERENCE_COLUMNS)}")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        return

    # Streamlit's uploader gives an in-memory buffer, but predict_model's
    # pipeline is path-based (so it can be reused identically from the
    # CLI); writing to a temp file lets us call that exact same function
    # without adding a second, buffer-based code path to maintain.
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / uploaded_file.name
        input_path.write_bytes(uploaded_file.getvalue())
        output_path = Path(tmp_dir) / "predictions.csv"

        try:
            with st.spinner("Generating predictions..."):
                model = get_trained_model()
                predictions_df = run_prediction_pipeline(
                    input_path=input_path, output_path=output_path, model=model
                )
        except FileNotFoundError as exc:
            st.error(f"File error: {exc}")
            return
        except ValueError as exc:
            st.error(f"Could not generate predictions: {exc}")
            return

        st.success(f"Generated {len(predictions_df):,} predictions.")
        st.dataframe(predictions_df, use_container_width=True)

        csv_bytes = predictions_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download predictions.csv",
            data=csv_bytes,
            file_name=Path(PREDICTIONS_OUTPUT_PATH).name,
            mime="text/csv",
        )


def render_business_insights_page() -> None:
    """Render the Business Insights page with narrative interpretation of the EDA/model."""
    st.title("💼 Business Insights")

    try:
        df = get_feature_dataset()
    except FileNotFoundError as exc:
        st.error(f"Could not load the feature dataset: {exc}")
        return

    tab_promo, tab_store, tab_season, tab_reco = st.tabs(
        ["Promotion Effectiveness", "Store Performance", "Seasonality & Competition", "Recommendations"]
    )

    with tab_promo:
        try:
            st.plotly_chart(plots.plot_promotion_impact(df), use_container_width=True)
        except ValueError as exc:
            st.warning(f"Chart unavailable: {exc}")
        st.markdown(
            "- Promotions are associated with a visible upward shift in the sales "
            "distribution, supporting continued (and potentially expanded) use of "
            "targeted promotions during low-traffic periods.\n"
            "- The spread of sales *with* promotion tends to be wider, meaning promo "
            "effectiveness likely varies by store or region and may benefit from "
            "more granular targeting rather than a blanket policy."
        )

    with tab_store:
        try:
            st.plotly_chart(plots.plot_top_performing_stores(df, top_n=10), use_container_width=True)
        except ValueError as exc:
            st.warning(f"Chart unavailable: {exc}")
        st.markdown(
            "- A small number of stores contribute disproportionately to total sales, "
            "typical of retail chains -- these flagship stores are good candidates "
            "for priority inventory allocation.\n"
            "- Store type and assortment breakdowns (see the EDA page) help identify "
            "which store formats consistently under- or over-perform, informing "
            "future store-format decisions."
        )

    with tab_season:
        try:
            st.plotly_chart(plots.plot_monthly_sales_trend(df), use_container_width=True)
        except ValueError as exc:
            st.warning(f"Chart unavailable: {exc}")
        st.markdown(
            "- Monthly aggregation reveals recurring seasonal peaks (e.g. holiday "
            "periods), which should drive seasonal inventory build-up ahead of "
            "predictable demand spikes.\n"
            "- Where competition-related features are available in the engineered "
            "dataset, sustained sales dips can often be traced to nearby competitor "
            "openings and should be monitored alongside store-level trends."
        )

    with tab_reco:
        st.markdown(
            "**Recommendations**\n"
            "1. Prioritize promotion spend on stores/periods showing the most "
            "consistent promo uplift rather than applying promotions uniformly.\n"
            "2. Use the model's predictions to pre-position inventory ahead of "
            "known seasonal peaks identified in the monthly trend.\n"
            "3. Investigate consistently underperforming store types/assortments "
            "as candidates for format changes or targeted local promotions.\n"
            "4. Monitor prediction error (see Model Performance page) over time; "
            "a rising RMSE/MAPE trend is an early signal the model needs "
            "retraining on more recent data."
        )


def render_about_page() -> None:
    """Render the About page: tech stack, folder structure, author, GitHub."""
    st.title("ℹ️ About This Project")

    with st.expander("🛠️ Tech Stack", expanded=True):
        st.markdown(
            "- **Language:** Python\n"
            "- **Data processing:** pandas, NumPy\n"
            "- **Modeling:** scikit-learn (Linear Regression, Random Forest, "
            "HistGradientBoostingRegressor)\n"
            "- **Model selection:** RandomizedSearchCV + TimeSeriesSplit\n"
            "- **Visualization:** Plotly\n"
            "- **Dashboard:** Streamlit\n"
            "- **Persistence:** joblib, JSON"
        )

    with st.expander("📁 Folder Structure"):
        st.code(
            "src/\n"
            "├── config.py\n"
            "├── data/\n"
            "│   ├── load_data.py\n"
            "│   ├── merge_data.py\n"
            "│   └── clean_data.py\n"
            "├── features/\n"
            "│   └── build_features.py\n"
            "├── models/\n"
            "│   ├── train_model.py\n"
            "│   ├── evaluate_model.py\n"
            "│   └── predict_model.py\n"
            "└── visualization/\n"
            "    └── plots.py\n"
            "app.py",
            language="text",
        )

    with st.expander("👤 Author & GitHub"):
        st.markdown(
            "**Author:** _Add your name here_\n\n"
            "**GitHub:** _Add your repository link here_\n\n"
            "Feel free to explore the source code, open issues, or reach out with "
            "questions about the modeling approach."
        )


def main() -> None:
    """Configure the page, render navigation, and dispatch to the selected page."""
    configure_page()
    page = render_sidebar()

    page_renderers = {
        PAGE_HOME: render_home_page,
        PAGE_EDA: render_eda_page,
        PAGE_MODEL_PERFORMANCE: render_model_performance_page,
        PAGE_PREDICTIONS: render_predictions_page,
        PAGE_INSIGHTS: render_business_insights_page,
        PAGE_ABOUT: render_about_page,
    }
    page_renderers[page]()


if __name__ == "__main__":
    main()
