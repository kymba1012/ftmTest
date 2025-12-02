"""
Data Audit Script
-----------------
Run this script to inspect the quality of your dataset (CSV or XLSX).
It will report:
1. Missing values in key grouping columns (Division, Manager).
2. Correlation between the 'error' column and the 'status' column.
3. Distribution of errors.

Usage:
    python data_analysis.py
"""

import pandas as pd
from pathlib import Path

# Try to find the dataset
CSV_PATH = Path("20251128_FtData.csv")
XLSX_PATH = Path("20251128_FtData.xlsx")

def load_data():
    if CSV_PATH.exists():
        print(f"Loading {CSV_PATH}...")
        return pd.read_csv(CSV_PATH, dtype=str)
    elif XLSX_PATH.exists():
        print(f"Loading {XLSX_PATH}...")
        return pd.read_excel(XLSX_PATH, dtype=str)
    else:
        print("❌ No data file found (checked .csv and .xlsx)")
        return None

def analyze():
    df = load_data()
    if df is None:
        return

    total_rows = len(df)
    print(f"\n--- General Stats ---")
    print(f"Total Rows: {total_rows}")
    print(f"Columns: {list(df.columns)}")

    # 1. Missing Value Analysis
    print(f"\n--- Missing Data Analysis ---")
    print("Columns often used for grouping:")
    group_cols = ['division', 'manager', 'customer', 'job', 'status']
    for col in group_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"  {col:<15}: {missing} missing values ({missing/total_rows:.1%})")
        else:
            print(f"  {col:<15}: [Column not found]")

    # 2. Error vs Status Insight
    if 'status' in df.columns and 'error' in df.columns:
        print(f"\n--- Technical Error vs Human Status ---")
        # Convert error to numeric for counting
        df['error_num'] = pd.to_numeric(df['error'], errors='coerce').fillna(0).astype(int)
        
        print("Breakdown of 'error' flag by 'status':")
        summary = df.groupby(['status', 'error_num']).size().unstack(fill_value=0)
        summary.columns = ['Clean (Error=0)', 'Error (Error=1)']
        print(summary)
        
        print("\nInsight:")
        print("If you see numbers in the 'Error=1' column for 'APPROVED' rows,")
        print("it means tickets with data errors are being approved manually.")

if __name__ == "__main__":
    analyze()
