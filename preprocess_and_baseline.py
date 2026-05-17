"""
HBAAC Time-Series Forecasting Pipeline & Baseline Generator.

Author: Senior Data Scientist & Retail Analytics Specialist
Description: PEP-8 compliant pipeline to preprocess historical retail transaction
             logs, build a continuous time-series matrix, and generate a baseline
             forecast submission.
"""

import os
import re
import time
import logging
import pandas as pd
import numpy as np

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants and Configuration
TRAIN_PATH = os.path.join("hbaac-round2", "train.csv")
SAMPLE_SUB_PATH = os.path.join("hbaac-round2", "sample_submission.csv")
OUTPUT_PATH = os.path.join("hbaac-round2", "baseline_submission.csv")

START_DATE = "2020-11-17"
END_DATE = "2025-09-05"
BASELINE_WINDOW_DAYS = 28


def parse_monetary_column(series: pd.Series) -> pd.Series:
    """
    Parses dirty monetary columns containing string-formatted prices.
    
    Handles:
      - Trailing/leading quotes or whitespaces.
      - Decimal commas (e.g., "123,1" -> 123.1).
      - Multiple comma thousand separators (e.g., "1,318,181" -> 1318181.0).
      - Multi-comma formatted values (e.g., "1,318,181,818" -> 1318181818.0).
      
    This unique-value mapping strategy dramatically speeds up pandas parsing.
    """
    if pd.api.types.is_numeric_dtype(series):
        # Already numerical, fill NaNs and return
        return series.astype(float).fillna(0.0)
    
    # Strip quotes, whitespaces, and convert to string
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
            # Single comma case
            if len(parts) == 2:
                # Heuristic: If followed by exactly 3 digits, it's a thousand separator (e.g., "1,318")
                # Otherwise, it represents a decimal comma (e.g., "123,1" or "71545,45")
                if len(parts[1]) == 3:
                    parsed_map[val] = float("".join(parts))
                else:
                    parsed_map[val] = float(f"{parts[0]}.{parts[1]}")
            # Multiple commas case (e.g., "1,318,181" or "1,318,181,818")
            else:
                # If the final group has exactly 3 digits, treat all commas as thousand separators
                if len(parts[-1]) == 3:
                    parsed_map[val] = float("".join(parts))
                # Otherwise, the last comma is a decimal and previous ones are thousand separators
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
    Loads raw transaction log, parses types, and handles returns.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing training dataset file: {train_path}")
        
    logger.info("Loading transaction logs from %s...", train_path)
    start_time = time.time()
    
    # Ingest CSV
    df = pd.read_csv(train_path)
    logger.info("Loaded %d rows in %.2f seconds.", len(df), time.time() - start_time)
    
    # Convert Date column
    logger.info("Casting date column to datetime...")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Process monetary columns
    monetary_cols = ['UnitPrice', 'SalesAmount', 'Unit Cost', 'Cost Amount']
    logger.info("Parsing and type casting monetary columns: %s...", monetary_cols)
    for col in monetary_cols:
        if col in df.columns:
            df[col] = parse_monetary_column(df[col])
            
    # Daily aggregation & Return transaction handling
    logger.info("Aggregating sales quantities daily by ItemCode...")
    # Grouping by Date and ItemCode and summing Quantity to capture net daily sales
    df_agg = df.groupby(['Date', 'ItemCode'], as_index=False)['Quantity'].sum()
    
    # Non-negative constraint enforcement (clip negative quantities from excessive returns)
    logger.info("Enforcing non-negative daily quantity constraint...")
    df_agg['Quantity'] = df_agg['Quantity'].clip(lower=0)
    
    return df_agg


def build_timeseries_matrix(df_agg: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pivots daily aggregated quantities and reindexes to a continuous date range.
    """
    logger.info("Pivoting DataFrame into time-series matrix (rows=ItemCode, cols=Date)...")
    df_pivot = df_agg.pivot(index='ItemCode', columns='Date', values='Quantity')
    
    logger.info("Reindexing matrix to continuous range: %s to %s...", start_date, end_date)
    continuous_dates = pd.date_range(start=start_date, end=end_date)
    df_matrix = df_pivot.reindex(columns=continuous_dates, fill_value=0)
    
    logger.info("Time-series matrix successfully created. Shape: %s", df_matrix.shape)
    return df_matrix


def generate_baseline_submission(
    df_matrix: pd.DataFrame, 
    sample_sub_path: str, 
    output_path: str
) -> None:
    """
    Generates baseline forecast by taking historical demand mean and mapping to template.
    """
    if not os.path.exists(sample_sub_path):
        raise FileNotFoundError(f"Missing submission template file: {sample_sub_path}")
        
    logger.info("Calculating mean quantity over the last %d days of training data...", BASELINE_WINDOW_DAYS)
    # Target columns representing the last 28 columns (dates) in our matrix
    last_28_days = df_matrix.iloc[:, -BASELINE_WINDOW_DAYS:]
    mean_last_28 = last_28_days.mean(axis=1)
    
    logger.info("Loading submission template: %s...", sample_sub_path)
    submission = pd.read_csv(sample_sub_path, index_col=0)
    
    logger.info("Mapping baseline predictions using vectorized index transformations...")
    # Extract ItemCodes from submission indexes (e.g. 'SKU-00001_validation' -> 'SKU-00001')
    submission_idx = submission.index.to_series()
    parts = submission_idx.str.rsplit('_', n=1)
    item_codes = parts.str[0]
    window_type = parts.str[1]
    
    # Map calculated historical means (defaults to 0.0 for items not present in training)
    mapped_means = item_codes.map(mean_last_28).fillna(0.0)
    
    # Enforce evaluation rows to remain 0.0 (only validation rows get baseline means)
    final_predictions = np.where(window_type == 'validation', mapped_means, 0.0)
    
    # Assign prediction values across all forecast horizons F1 to F28
    for col in submission.columns:
        submission[col] = final_predictions
        
    logger.info("Saving baseline submission file to %s...", output_path)
    submission.to_csv(output_path)
    logger.info("Submission generated successfully! Total rows: %d.", len(submission))


def main():
    """
    Orchestration main execution flow.
    """
    start_pipeline = time.time()
    logger.info("=== Starting Preprocessing and Baseline Pipeline ===")
    
    try:
        # Step 1: Preprocess transactional data and handle returns
        df_agg = load_and_preprocess_data(TRAIN_PATH)
        
        # Step 2: Reshape to broad continuous time-series matrix
        df_matrix = build_timeseries_matrix(df_agg, START_DATE, END_DATE)
        
        # Step 3: Compute final 28-day demand averages and write submission
        generate_baseline_submission(df_matrix, SAMPLE_SUB_PATH, OUTPUT_PATH)
        
        logger.info("=== Pipeline Completed Successfully in %.2f seconds! ===", time.time() - start_pipeline)
        
    except Exception as e:
        logger.exception("Pipeline failed due to an error: %s", str(e))


if __name__ == "__main__":
    main()
