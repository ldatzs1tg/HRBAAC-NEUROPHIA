"""
HBAAC Weekend Demand Audit and Anomaly Detection.

Author: Data Scientist & Visualization Expert
Description: PEP-8 compliant data audit script to inspect Saturday vs Sunday sales.
             Produces two premium subplots:
             1. Overlaid Time-Series Scatter/Line Plot for weekend sales comparison.
             2. Outlier Detection Boxplot comparing Saturday vs Sunday distribution.
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

# Constants and Paths
if os.path.exists(os.path.join("hbaac-round2", "train.csv")):
    TRAIN_PATH = os.path.join("hbaac-round2", "train.csv")
    OUTPUT_DIR = os.path.join("EDA", "eda_plots")
else:
    TRAIN_PATH = os.path.join("..", "hbaac-round2", "train.csv")
    OUTPUT_DIR = "eda_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PLOT_PATH = os.path.join(OUTPUT_DIR, "weekend_anomalies.png")

# Set Premium Style Aesthetics
sns.set_theme(style="ticks")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "font.family": "sans-serif"
})


def load_and_preprocess_data(train_path: str) -> pd.DataFrame:
    """
    Loads raw transaction logs, aggregates quantity daily, clips negative values,
    and filters for Saturday and Sunday transactions.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing training dataset file: {train_path}")
        
    logger.info("Loading transaction logs from %s...", train_path)
    start_time = time.time()
    df = pd.read_csv(train_path, low_memory=False)
    logger.info("Loaded %d rows in %.2f seconds.", len(df), time.time() - start_time)
    
    # Cast Date column to datetime
    logger.info("Casting 'Date' column to datetime...")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Parse Quantity to numerical values safely
    logger.info("Parsing 'Quantity' column...")
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0.0)
    
    # Net daily sales by SKU
    logger.info("Aggregating sales quantities daily by ItemCode...")
    df_sku_daily = df.groupby(['Date', 'ItemCode'], as_index=False)['Quantity'].sum()
    
    # Enforce non-negative constraint
    df_sku_daily['Quantity'] = df_sku_daily['Quantity'].clip(lower=0.0)
    
    # Group by Date to get total system quantity
    logger.info("Aggregating daily quantity system-wide...")
    df_daily = df_sku_daily.groupby('Date', as_index=False)['Quantity'].sum()
    
    # Extract dayofweek
    df_daily['dayofweek'] = df_daily['Date'].dt.dayofweek
    
    # Filter for Saturday (5) and Sunday (6)
    logger.info("Filtering Saturday (dayofweek==5) and Sunday (dayofweek==6) data...")
    df_weekend = df_daily[df_daily['dayofweek'].isin([5, 6])].copy()
    
    # Map DayName
    df_weekend['DayName'] = df_weekend['dayofweek'].map({5: 'Saturday', 6: 'Sunday'})
    
    return df_weekend


def main():
    logger.info("=== Starting Weekend Demand Audit ===")
    
    try:
        # Load and preprocess data
        df_weekend = load_and_preprocess_data(TRAIN_PATH)
        
        # Prepare figure
        fig, axes = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [1.2, 1]})
        
        # Plot 1: Overlaid Time-Series Line/Scatter Plot
        logger.info("Generating Plot 1: Overlaid Time-Series...")
        ax1 = axes[0]
        
        # Filter datasets
        df_sat = df_weekend[df_weekend['dayofweek'] == 5]
        df_sun = df_weekend[df_weekend['dayofweek'] == 6]
        
        # Saturday - deep elegant blue
        ax1.scatter(df_sat['Date'], df_sat['Quantity'], color='#2E3B84', alpha=0.75, 
                    s=25, label='Saturday (Thứ 7)', marker='o')
        ax1.plot(df_sat['Date'], df_sat['Quantity'], color='#2E3B84', alpha=0.3, 
                 linewidth=1, linestyle='--')
        
        # Sunday - vibrant red/coral (makes spikes pop out instantly)
        ax1.scatter(df_sun['Date'], df_sun['Quantity'], color='#E53935', alpha=0.85, 
                    s=35, label='Sunday (Chủ Nhật)', marker='X')
        ax1.plot(df_sun['Date'], df_sun['Quantity'], color='#E53935', alpha=0.3, 
                 linewidth=1, linestyle='-')
        
        ax1.set_title("System-Wide Total Quantity: Saturday vs Sunday Timeline Audit", 
                      fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel("Date", fontsize=11, labelpad=8)
        ax1.set_ylabel("Total Quantity", fontsize=11, labelpad=8)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(frameon=True, facecolor='white', edgecolor='none', shadow=False)
        
        # Format X-axis dates beautifully
        import matplotlib.dates as mdates
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
        
        # Plot 2: Boxplot (Anomaly/Outlier Detection)
        logger.info("Generating Plot 2: Boxplot comparison...")
        ax2 = axes[1]
        
        # Premium custom boxplot palette
        custom_palette = {'Saturday': '#3F51B5', 'Sunday': '#E53935'}
        
        sns.boxplot(
            data=df_weekend, 
            x='DayName', 
            y='Quantity', 
            ax=ax2, 
            order=['Saturday', 'Sunday'],
            palette=custom_palette,
            width=0.4,
            fliersize=6,
            flierprops={'marker': 'D', 'markerfacecolor': 'black', 'markeredgecolor': 'none', 'alpha': 0.6}
        )
        
        # Overlay stripplot with jitter for premium visualization of individual data points density
        sns.stripplot(
            data=df_weekend,
            x='DayName',
            y='Quantity',
            ax=ax2,
            order=['Saturday', 'Sunday'],
            color='black',
            alpha=0.15,
            size=3,
            jitter=0.15,
            dodge=True
        )
        
        ax2.set_title("Sales Distribution & Outlier Detection: Saturday vs Sunday", 
                      fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel("Day of Week", fontsize=11, labelpad=8)
        ax2.set_ylabel("Total Quantity", fontsize=11, labelpad=8)
        ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
        
        # Global optimizations
        plt.suptitle("Weekend Transaction Audit & Anomaly Detection Profile", 
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        logger.info("Saving generated audit plots to %s...", OUTPUT_PLOT_PATH)
        plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches='tight')
        logger.info("=== Audit Visualization successfully saved! ===")
        
    except Exception as e:
        logger.exception("An error occurred during audit plot generation: %s", str(e))


if __name__ == "__main__":
    main()
