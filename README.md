# HBAAC Retail Time-Series Forecasting Project

## 1. Project Overview
Welcome to the **HBAAC Time-Series Forecasting** project. The goal of this project is to build a robust, production-ready machine learning pipeline to forecast daily transaction quantities for a retail system spanning **15,972 SKUs** across two distinct 28-day forecasting horizons:
- **Validation Window**: Daily demand forecasting for days F1 to F28.
- **Evaluation Window**: Daily demand forecasting for days F29 to F56 (represented in the submission structure as days F1 to F28 in the `_evaluation` rows).

This competition involves massive scale demand forecasting (15k+ products over multiple years), high sparsity (many zero-demand days), and negative quantity anomalies representing returned transactions.

---

## 2. Directory Structure
This repository is established with a highly structured layout to facilitate rigorous experiment tracking, modular feature engineering, and robust model validation:

```text
HBAAC/
├── hbaac-round2/             # Raw Competition Datasets (Ignored in Version Control)
│   ├── train.csv             # Historical transaction logs
│   ├── sample_submission.csv # Target prediction template
│   └── baseline_submission.csv # Generated baseline submission
├── EDA/                      # Exploratory Data Analysis Folder
│   ├── eda_visualizations.py # Standalone EDA visualization generator
│   └── eda_plots/            # Saved graphical plots and markdown reports
│       ├── pareto_sku_distribution.png
│       ├── macro_sales_trends.png
│       ├── micro_calendar_seasonality.png
│       └── analysis.md       # Comprehensive EDA analysis report
├── README.md                 # Project overview and pipeline documentation
├── COMPETITION_RULES.md      # Detailed competition constraints and evaluation metrics
├── EXPERIMENT_TRACKING.md    # Log table for all model iterations and leaderboard scores
├── FEATURE_DICTIONARY.md     # Catalog of engineered features and their mathematical definitions
└── preprocess_and_baseline.py # Preprocessing pipeline & Baseline model generation script
```

---

## 3. Data Flow and Architecture
```mermaid
graph TD
    A[hbaac-round2/train.csv] --> B[preprocess_and_baseline.py]
    C[hbaac-round2/sample_submission.csv] --> B
    B -->|Ingest, Clean, Parse| D[Continuous Time-Series Matrix]
    D -->|Group & Sum by Date/ItemCode| E[Daily Aggregated Sales]
    E -->|Apply .clip(lower=0)| F[Enforced Positive Quantity]
    F -->|Pivot & Reindex 2020-11-17 to 2025-09-05| G[Structured Matrix]
    G -->|Mean of Last 28 Days| H[Baseline Predictor]
    H -->|Map Validation & Set Eval to 0| I[hbaac-round2/baseline_submission.csv]
```

---

## 4. Getting Started & Execution Instructions

### Prerequisites
Ensure you have the virtual environment activated and the required dependencies installed:
```bash
pip install -r requirements.txt
```

### Running the Preprocessing & Baseline Pipeline
To run the preprocessing pipeline and generate the baseline forecast:
```bash
python preprocess_and_baseline.py
```

### Running the Exploratory Data Analysis (EDA) Script
To run the EDA pipeline and regenerate the analytical plots:
```bash
python EDA/eda_visualizations.py
```

### Script Execution Logic
1. **Type Casting & Robust Parsing**: Column dates are converted to datetime format, and string-formatted monetary columns (`UnitPrice`, `SalesAmount`, `Unit Cost`, `Cost Amount`) containing localized decimals and thousand separators are parsed into standardized floats.
2. **Returns Handling**: Transaction records are aggregated daily. Daily returns exceeding purchases (resulting in negative sum quantities) are clipped to `0` to adhere to the non-negative prediction constraint.
3. **Time-Series Matrix Construction**: Daily aggregated data is pivoted into a broad time-series matrix where rows are `ItemCode` and columns represent a continuous chronological date range.
4. **Baseline Modeling**: The script computes the mean quantity per item over the final 28 training days and populates the validation window of the submission file accordingly.
