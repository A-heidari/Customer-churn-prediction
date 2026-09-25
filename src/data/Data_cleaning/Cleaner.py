"""
src/data/Data_cleaning/Cleaner.py

Cleaning pipeline for the Telco Customer Churn dataset.
"""

import pandas as pd
from src.data.ingestion.load_data import load_data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline and return a cleaned copy."""
    df = df.copy()

    # Strip whitespace from string columns
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda col: col.str.strip())

    # Drop non-predictive identifier
    df = df.drop(columns=["customerID"], errors="ignore")

    # Fix TotalCharges: stored as string, blank for tenure == 0 customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Normalize "No xxx service" values into a clean "No"
    df = df.replace({"No internet service": "No", "No phone service": "No"})

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    return df


if __name__ == "__main__":
    df = load_data("Telco.csv")
    df = clean_data(df)

    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("\nHead:\n", df.head())