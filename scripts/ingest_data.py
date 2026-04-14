import pandas as pd
from pathlib import Path

# Setup paths - Now targeting the Excel file
RAW_DATA = Path("data/raw/supermarket_transactions.xlsx")
PROCESSED_DATA = Path("data/processed/cleaned_transactions.csv")

def ingest_and_clean():
    print("--- Starting Data Ingestion from Excel ---")
    
    # 1. Load Excel data
    if not RAW_DATA.exists():
        print(f"❌ Error: Could not find {RAW_DATA}")
        return

    # Use openpyxl engine to read the .xlsx file
    df = pd.read_excel(RAW_DATA, engine='openpyxl')
    print(f" Loaded {len(df)} transactions from Excel.")

    # 2. Basic Cleaning
    required_cols = ["customer_id", "timestamp", "total_amount"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found columns: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
    df = df.dropna(subset=["customer_id", "timestamp", "total_amount"])
    df = df[df["total_amount"] > 0]

    # 3. Save as CSV for ML efficiency
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA, index=False)
    
    print(f" Cleaned data saved to {PROCESSED_DATA}")

if __name__ == "__main__":
    ingest_and_clean()