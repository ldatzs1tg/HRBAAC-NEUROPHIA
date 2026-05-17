"""
HBAAC Retail Time-Series Exploratory Data Analysis (EDA) Pipeline.

Author: Visualization Expert & Retail Analytics Specialist
Description: PEP-8 compliant, production-grade exploratory data analysis script.
             Generates three core business-critical visualizations using
             Matplotlib and Seaborn:
             1. Pareto Distribution (Long-tail SKU analysis)
             2. Macro Seasonality Trends (Weekly aggregated sales timeline)
             3. Micro Demand Structure (Day-of-week and Month-of-year boxplots)
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants & Configurations
# Make paths robust so the script runs successfully from both the project root and the EDA/ folder
if os.path.exists(os.path.join("hbaac-round2", "train.csv")):
    TRAIN_PATH = os.path.join("hbaac-round2", "train.csv")
    OUTPUT_DIR = os.path.join("EDA", "eda_plots")
else:
    TRAIN_PATH = os.path.join("..", "hbaac-round2", "train.csv")
    OUTPUT_DIR = "eda_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set premium styling aesthetics globally
sns.set_theme(style="white", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "font.family": "sans-serif"
})


def parse_monetary_column(series: pd.Series) -> pd.Series:
    """
    Parses dirty monetary columns containing string-formatted prices.
    Uses a highly efficient unique-value mapping strategy.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float).fillna(0.0)
    
    cleaned_series = series.astype(str).str.strip().str.replace('"', '').str.replace("'", "")
    cleaned_series = cleaned_series.replace({'nan': '0', '': '0'})
    
    unique_vals = cleaned_series.unique()
    parsed_map = {}
    
    for val in unique_vals:
        if not val or val == '0':
            parsed_map[val] = 0.0
            continue
            
        if ',' in val:
            parts = val.split(',')
            if len(parts) == 2:
                if len(parts[1]) == 3:
                    parsed_map[val] = float("".join(parts))
                else:
                    parsed_map[val] = float(f"{parts[0]}.{parts[1]}")
            else:
                if len(parts[-1]) == 3:
                    parsed_map[val] = float("".join(parts))
                else:
                    parsed_map[val] = float("".join(parts[:-1]) + "." + parts[-1])
        else:
            try:
                parsed_map[val] = float(val)
            except ValueError:
                parsed_map[val] = 0.0
                
    return cleaned_series.map(parsed_map).fillna(0.0)


def load_and_preprocess_data(train_path: str) -> pd.DataFrame:
    """
    Loads raw transaction log, parses Date, parses monetary columns,
    aggregates Quantity daily, and clips negative quantities from returns.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing training dataset file: {train_path}")
        
    logger.info("Loading transaction logs for EDA from %s...", train_path)
    df = pd.read_csv(train_path)
    
    logger.info("Casting date column to datetime...")
    df['Date'] = pd.to_datetime(df['Date'])
    
    logger.info("Parsing and casting monetary columns...")
    monetary_cols = ['UnitPrice', 'SalesAmount', 'Unit Cost', 'Cost Amount']
    for col in monetary_cols:
        if col in df.columns:
            df[col] = parse_monetary_column(df[col])
            
    # Aggregate to daily levels by ItemCode
    logger.info("Aggregating sales quantities daily by ItemCode...")
    df_agg = df.groupby(['Date', 'ItemCode'], as_index=False).agg({
        'Quantity': 'sum',
        'SalesAmount': 'sum'
    })
    
    # Non-negative constraint enforcement (handling return anomalies)
    df_agg['Quantity'] = df_agg['Quantity'].clip(lower=0)
    df_agg['SalesAmount'] = df_agg['SalesAmount'].clip(lower=0)
    
    return df_agg


def plot_pareto_distribution(df_agg: pd.DataFrame, output_dir: str) -> None:
    """
    1. Long-Tail Distribution Analysis (Pareto Principle)
    
    Business Rationale:
      In retail demand forecasting, the Pareto Principle (the 80/20 rule) is highly
      prominent. A tiny fraction of SKUs (ItemCode) drives the vast majority of
      total sales volume. Visualizing this cumulative volume ranked by SKU importance
      proves this long-tail structure, which justifies prioritizing modeling efforts,
      hyperparameter tuning, and safety stock optimizations for top-performing SKUs
      that carry the heaviest WRMSSE evaluation weight.
      
    Chart Selection:
      A dual-axis visualization is implemented:
      - Primary Axis (Bar Chart): Bar sales volume for individual ranked ItemCodes.
      - Secondary Axis (Line Chart): The continuous cumulative percentage curve.
    """
    logger.info("Generating Pareto Long-Tail Distribution plot...")
    
    # Calculate cumulative Quantity by ItemCode
    sku_sales = df_agg.groupby('ItemCode')['Quantity'].sum().reset_index()
    sku_sales = sku_sales.sort_values(by='Quantity', ascending=False).reset_index(drop=True)
    sku_sales['Cumulative_Quantity'] = sku_sales['Quantity'].cumsum()
    total_quantity = sku_sales['Quantity'].sum()
    sku_sales['Cumulative_Percentage'] = (sku_sales['Cumulative_Quantity'] / total_quantity) * 100
    
    # Find the SKU index where cumulative percentage crosses 80%
    cross_80_idx = np.where(sku_sales['Cumulative_Percentage'] >= 80)[0][0]
    sku_pct_at_80 = ((cross_80_idx + 1) / len(sku_sales)) * 100
    
    # Plotting setup
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Primary Axis - Bar Chart (Top 100 SKUs for readability)
    top_n = min(100, len(sku_sales))
    bars = ax1.bar(
        range(top_n), 
        sku_sales['Quantity'].head(top_n), 
        color="#1f77b4", 
        alpha=0.85, 
        edgecolor='none', 
        width=0.8,
        label="Individual SKU Demand"
    )
    ax1.set_ylabel("Total Demand Quantity Sold", color="#1f77b4")
    ax1.set_xlabel("SKUs (ItemCode) Sorted by Sales Volume (Top 100 displayed)")
    ax1.tick_params(axis='y', labelcolor="#1f77b4")
    ax1.set_title("Pareto Long-Tail Demand Distribution across SKUs", pad=15)
    
    # Secondary Axis - Cumulative Percentage Line Chart
    ax2 = ax1.twinx()
    line = ax2.plot(
        range(top_n), 
        sku_sales['Cumulative_Percentage'].head(top_n), 
        color="#e377c2", 
        linewidth=2.5, 
        label="Cumulative % of Total Demand"
    )
    ax2.set_ylabel("Cumulative Percentage of Total Demand (%)", color="#e377c2")
    ax2.tick_params(axis='y', labelcolor="#e377c2")
    
    # Annotate the 80% line
    ax2.axhline(80, color="#d62728", linestyle="--", alpha=0.7, linewidth=1.5)
    ax2.text(
        x=top_n * 0.4, 
        y=82, 
        s=f"Top {sku_pct_at_80:.1f}% of SKUs account for 80% of total demand volume", 
        color="#d62728", 
        weight='semibold'
    )
    
    # Aesthetics polish: Remove top spines
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pareto_sku_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved Pareto plot to %s.", plot_path)


def plot_macro_trends(df_agg: pd.DataFrame, output_dir: str) -> None:
    """
    2. Macro Seasonality and Trend Analysis (Weekly Sales Timeline)
    
    Business Rationale:
      A retail store's demand operates inside global macroeconomic cycles, yearly
      seasonal peaks, and business disruptions. Visualizing the sum demand across
      all SKUs over the historical 5-year range highlights long-term trends and
      repetitive yearly peaks (like Tet holiday spikes or year-end sales). This is
      essential for detecting structural breaks or shifting baselines that models
      must account for.
      
    Chart Selection:
      A weekly resampled line chart is utilized. Weekly aggregation smooths out the
      extreme daily variance and zero-sales sparsity while preserving the clear
      underlying seasonal contours, trends, and macro-events.
    """
    logger.info("Generating Macro Seasonality Trends timeline...")
    
    # Group by Date to get total global daily quantity
    daily_sales = df_agg.groupby('Date')['Quantity'].sum().reset_index()
    daily_sales = daily_sales.set_index('Date')
    
    # Resample to Weekly ('W') frequency to smooth high-frequency daily noise
    weekly_sales = daily_sales['Quantity'].resample('W').sum().reset_index()
    
    # Plotting setup
    plt.figure(figsize=(14, 5.5))
    plt.plot(
        weekly_sales['Date'], 
        weekly_sales['Quantity'], 
        color="#2ca02c", 
        linewidth=2.0, 
        alpha=0.9,
        label="Weekly Demand Sum"
    )
    
    # Apply a 12-week rolling average line to represent the long-term trend
    weekly_sales['Trend_12W'] = weekly_sales['Quantity'].rolling(window=12, center=True).mean()
    plt.plot(
        weekly_sales['Date'], 
        weekly_sales['Trend_12W'], 
        color="#ff7f0e", 
        linestyle="-", 
        linewidth=2.5,
        label="12-Week Central Trend"
    )
    
    # Aesthetics Polish
    plt.title("Global Retail Demand Timeline: Macro Trends and Peak Seasonality", pad=15)
    plt.xlabel("Timeline (Years)")
    plt.ylabel("Total Quantity Sold (Weekly)")
    plt.grid(axis='y', linestyle=":", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    
    # Remove outer spines
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "macro_sales_trends.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved Macro trends plot to %s.", plot_path)


def plot_micro_seasonality(df_agg: pd.DataFrame, output_dir: str) -> None:
    """
    3. Micro Demand Structure (Calendar Seasonality)
    
    Business Rationale:
      Retail demand is fundamentally driven by human calendar habits. Customers buy
      differently depending on the day of the week (weekend spikes vs. weekday lulls)
      and month of the year (summer dips vs. holiday shopping surges). Extracting
      this temporal behavior directly informs feature engineering (allowing models
      to use day-of-week lags and monthly target encodings).
      
    Chart Selection:
      Boxplots are selected because they represent the entire statistical distribution
      (median, quartiles, and extremes) rather than simple means. This captures the
      inherent variance, skewness, and outlying purchase spikes on promotional days
      for each calendar group.
    """
    logger.info("Generating Micro Demand calendar boxplots...")
    
    # Aggregate sales globally by Date
    daily_sales = df_agg.groupby('Date')['Quantity'].sum().reset_index()
    
    # Extract Calendar Features
    daily_sales['DayOfWeek'] = daily_sales['Date'].dt.day_name()
    daily_sales['Month'] = daily_sales['Date'].dt.strftime('%b')
    
    # Order categories logically
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    daily_sales['DayOfWeek'] = pd.Categorical(daily_sales['DayOfWeek'], categories=day_order, ordered=True)
    daily_sales['Month'] = pd.Categorical(daily_sales['Month'], categories=month_order, ordered=True)
    
    # Plotting setup: 2 subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    
    # Plot 1: Day of Week Boxplot
    sns.boxplot(
        x='DayOfWeek', 
        y='Quantity', 
        data=daily_sales, 
        ax=axes[0], 
        palette="Blues",
        showfliers=False,  # Exclude extreme outliers for clear distribution display
        linewidth=1.2
    )
    axes[0].set_title("Demand Distribution by Day of Week", pad=12)
    axes[0].set_xlabel("Day of the Week")
    axes[0].set_ylabel("Daily Quantity Sold (Global Sum)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30)
    
    # Plot 2: Month of Year Boxplot
    sns.boxplot(
        x='Month', 
        y='Quantity', 
        data=daily_sales, 
        ax=axes[1], 
        palette="Purples",
        showfliers=False,
        linewidth=1.2
    )
    axes[1].set_title("Demand Distribution by Month of Year", pad=12)
    axes[1].set_xlabel("Month of the Year")
    axes[1].set_ylabel("")  # Clear redundant label
    
    # Aesthetics Polish: Clean spines and add light grid
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.grid(axis='y', linestyle=":", alpha=0.4)
        
    plt.suptitle("Micro Demand Patterns: Calendar-Driven Cyclic Seasonality", y=0.98)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "micro_calendar_seasonality.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved Micro seasonality boxplot to %s.", plot_path)


def main():
    """
    Orchestration main execution flow.
    """
    logger.info("=== Starting Exploratory Data Analysis (EDA) Visualization Pipeline ===")
    start_time = time.time()
    
    try:
        # Step 1: Load and clean data (reusing clean transactional logic)
        df_agg = load_and_preprocess_data(TRAIN_PATH)
        
        # Step 2: Plot Pareto Long-tail Distribution
        plot_pareto_distribution(df_agg, OUTPUT_DIR)
        
        # Step 3: Plot Macro Trends Timeline
        plot_macro_trends(df_agg, OUTPUT_DIR)
        
        # Step 4: Plot Micro Seasonality Patterns
        plot_micro_seasonality(df_agg, OUTPUT_DIR)
        
        logger.info("=== EDA Pipeline Completed Successfully in %.2f seconds! ===", time.time() - start_time)
        logger.info("All plots are saved in directory: %s/", OUTPUT_DIR)
        
    except Exception as e:
        logger.exception("EDA pipeline failed due to an error: %s", str(e))


if __name__ == "__main__":
    main()
