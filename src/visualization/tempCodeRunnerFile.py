"""
EDA script: initial exploration of Telco churn dataset.

Responsibilities:
- Load dataset
- Inspect structure
- Check missing values
- Check duplicates
- Analyze target distribution
- Generate basic visualizations
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.ingestion.load_data import load_data


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

DEFAULT_FILENAME = "Telco.csv"
TARGET_COL = "Churn"

FIGURE_DIR = Path("reports/figures")


# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------

def load_dataset(
    filename: str = DEFAULT_FILENAME
) -> pd.DataFrame:
    """
    Load dataset using data loader.
    """

    try:
        df = load_data(filename)

    except FileNotFoundError:
        logger.error(
            "File '%s' not found.",
            filename
        )
        sys.exit(1)
        

    except Exception as e:
        logger.exception(
            "Failed loading '%s': %s",
            filename,
            e
        )
        sys.exit(1)


    if df.empty:
        logger.warning(
            "Dataset '%s' is empty.",
            filename
        )


    return df



# ---------------------------------------------------------
# Basic Inspection
# ---------------------------------------------------------

def quick_look(
    df: pd.DataFrame
) -> None:
    """
    Show basic dataset information.
    """

    logger.info(
        "Dataset shape: %s",
        df.shape
    )


    print("\n--- FIRST 5 ROWS ---")
    print(df.head())


    print("\n--- INFO ---")
    df.info()


    print("\n--- MISSING VALUES ---")

    missing = df.isnull().sum()

    if missing.any():
        print(
            missing[missing > 0]
        )
    else:
        print(
            "No missing values."
        )


    print("\n--- DUPLICATES ---")

    duplicates = df.duplicated().sum()

    print(
        f"{duplicates} duplicate rows found."
    )



# ---------------------------------------------------------
# Numeric Summary
# ---------------------------------------------------------

def numeric_summary(
    df: pd.DataFrame
) -> None:
    """
    Display statistics for numerical columns.
    """

    numeric_df = df.select_dtypes(
        include="number"
    )


    if numeric_df.empty:
        logger.info(
            "No numeric columns found."
        )
        return


    print("\n--- NUMERIC SUMMARY ---")

    print(
        numeric_df.describe().T
    )



# ---------------------------------------------------------
# Target Handling
# ---------------------------------------------------------

def resolve_target_column(
    df: pd.DataFrame,
    target_col: str
) -> str | None:
    """
    Find target column ignoring case.
    """

    if target_col in df.columns:
        return target_col


    matches = [
        col
        for col in df.columns
        if col.lower() == target_col.lower()
    ]


    if matches:

        logger.warning(
            "Using '%s' instead of '%s'",
            matches[0],
            target_col
        )

        return matches[0]


    logger.error(
        "Target column '%s' not found.",
        target_col
    )

    return None



# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

def churn_summary(
    df: pd.DataFrame,
    target_col: str = TARGET_COL
) -> None:
    """
    Plot target distribution.
    """

    target = resolve_target_column(
        df,
        target_col
    )


    if target is None:
        return


    print(
        f"\n--- {target.upper()} RATE ---"
    )

    print(
        df[target].value_counts(normalize=True)
    )


    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    fig, ax = plt.subplots(
        figsize=(6,4)
    )


    sns.countplot(
        x=target,
        data=df,
        ax=ax
    )


    ax.set_title(
        "Churn Distribution"
    )

    ax.set_xlabel(
        target
    )

    ax.set_ylabel(
        "Count"
    )


    total = len(df)


    for bar in ax.patches:

        percentage = (
            bar.get_height()
            / total
            * 100
        )


        ax.annotate(
            f"{percentage:.1f}%",
            (
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height()
            ),
            ha="center",
            va="bottom"
        )


    plt.tight_layout()


    output = (
        FIGURE_DIR
        /
        "churn_distribution.png"
    )


    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )


    logger.info(
        "Saved figure: %s",
        output
    )


    plt.show()



# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------

def main(
    filename: str = DEFAULT_FILENAME
) -> None:

    df = load_dataset(filename)

    quick_look(df)

    numeric_summary(df)

    churn_summary(df)



if __name__ == "__main__":

    file_arg = (
        sys.argv[1]
        if len(sys.argv) > 1
        else DEFAULT_FILENAME
    )

    main(file_arg)