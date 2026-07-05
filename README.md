# Retail Inventory Demand Forecasting & Business Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Status](https://img.shields.io/badge/status-in%20development-yellow)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Framework](https://img.shields.io/badge/dashboard-Streamlit-red)]()

A production-style, end-to-end data science project that forecasts store-level
retail demand and translates those forecasts into actionable inventory and
business insights through an interactive analytics dashboard.

This repository is structured as a real analytics team would build it —
separated data layers, a reusable feature/model pipeline, model versioning,
and a decoupled presentation layer — rather than a single notebook.

---

## 1. Project Overview

Retail chains lose revenue in two directions: **overstocking** (capital tied up,
markdowns, waste) and **understocking** (lost sales, poor customer experience).
This project builds a demand forecasting system on historical store sales data
and surfaces the results through a business-facing dashboard, enabling:

- Store-level daily demand forecasts
- Promotion and holiday impact quantification
- Competition-proximity impact analysis
- Store-level demand volatility and stock-risk flagging
- Full model transparency via feature importance and error analysis

**Dataset:** [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) (Kaggle)
— chosen for its combination of daily sales granularity, promotional events,
holiday calendars, and store metadata (type, assortment, competition distance),
which allows realistic feature engineering and business storytelling.

---

## 2. Tech Stack

| Layer | Tools |
|---|---|
| Data manipulation | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Model persistence | Joblib |
| Visualization | Plotly, Matplotlib |
| Application / dashboard | Streamlit |
| Testing | Pytest |

---

## 3. Repository Structure

```
retail-demand-forecasting/
│
├── data/                   # Data storage layer (never committed except .gitkeep)
│   ├── raw/                # Original, immutable source files
│   ├── interim/            # Merged / partially cleaned data
│   ├── processed/          # Final model-ready datasets
│   └── external/           # Supplementary reference data (e.g. holiday calendars)
│
├── notebooks/              # Exploratory analysis and experimentation only
│
├── src/                    # Production source code (importable package)
│   ├── config.py           # Centralized paths, constants, and settings
│   ├── data/                # Data loading, merging, cleaning
│   ├── features/            # Feature engineering and encoding logic
│   ├── models/               # Training, prediction, evaluation, model registry
│   ├── visualization/        # Reusable chart-building functions
│   └── utils/                 # Shared helpers (logging, timing, etc.)
│
├── pipeline/               # End-to-end orchestration script
│   └── run_pipeline.py      # load → clean → engineer → train → evaluate → serialize
│
├── models/                 # Model artifact storage
│   ├── trained/              # Serialized model files (.joblib)
│   └── metrics/               # Model registry log and evaluation metrics
│
├── app/                    # Streamlit dashboard application
│   ├── Home.py               # App entry point
│   ├── pages/                 # Individual dashboard pages
│   └── components/            # Shared UI components (sidebar, KPI cards, charts)
│
├── tests/                  # Unit tests for data, features, and models
│
├── reports/                # Exported analysis artifacts
│   └── figures/              # Saved charts for documentation/portfolio use
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

---

## 4. Folder Responsibilities

| Folder | Purpose |
|---|---|
| `data/raw/` | Immutable source data exactly as downloaded from Kaggle. Never modified in place. |
| `data/interim/` | Output of merging and cleaning steps — not yet feature-engineered. |
| `data/processed/` | Final, model-ready datasets, including train/validation/test splits. |
| `data/external/` | Any auxiliary reference data brought in beyond the core Kaggle dataset. |
| `notebooks/` | Scratch space for EDA and experimentation. Code that becomes reusable is migrated into `src/`. |
| `src/data/` | Scripted, repeatable data loading, merging, and cleaning logic. |
| `src/features/` | Feature construction (lags, rolling statistics, calendar features) and categorical encoders. |
| `src/models/` | Model training, inference, evaluation, and a lightweight model registry for version tracking. |
| `src/visualization/` | Chart-generation functions shared between notebooks and the Streamlit app. |
| `src/utils/` | Cross-cutting utilities such as logging configuration and timing decorators. |
| `pipeline/` | Single orchestration entry point that reproduces the entire workflow end-to-end. |
| `models/trained/` | Serialized, versioned model artifacts ready for inference. |
| `models/metrics/` | Persisted evaluation metrics and the model registry log. |
| `app/` | The Streamlit multi-page dashboard that consumes processed data and trained models. |
| `tests/` | Automated checks for data integrity, feature correctness, and model behavior. |
| `reports/figures/` | Static exported visuals for documentation and portfolio use. |

---

## 5. Project Status

This repository is under active development. Current stage: **repository
scaffolding and configuration complete** — data ingestion, feature
engineering, modeling, and the dashboard are implemented in subsequent
milestones.

| Milestone | Status |
|---|---|
| Repository scaffolding & configuration | ✅ Complete |
| Data loading & cleaning pipeline | ⏳ Pending |
| Feature engineering | ⏳ Pending |
| Model training & evaluation | ⏳ Pending |
| Streamlit dashboard | ⏳ Pending |
| Testing suite | ⏳ Pending |
| Deployment | ⏳ Pending |

---

## 6. Getting Started

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/retail-demand-forecasting.git
cd retail-demand-forecasting

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Download the [Rossmann Store Sales dataset](https://www.kaggle.com/c/rossmann-store-sales)
from Kaggle and place `train.csv`, `test.csv`, and `store.csv` into `data/raw/`.

---

## 7. Running the Project

> Pipeline and app entry points will become executable once the corresponding
> modules are implemented in later milestones.

```bash
# Run the full data-to-model pipeline (once implemented)
python pipeline/run_pipeline.py

# Launch the dashboard (once implemented)
streamlit run app/Home.py
```

---

## 8. License

This project is licensed under the MIT License.

---

## 9. Author

Developed as a portfolio project demonstrating end-to-end data science
engineering practices for analytics and data science roles.
