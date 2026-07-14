"""
EDA script: initial exploration of the Telco churn dataset.

Responsibilities:
- Load the dataset (via the data loader)
- Build a structured EDA report (shape, missing values, duplicates,
  outliers, target distribution)
- Write the report to a Markdown file
- Generate basic visualizations (target distribution, correlation heatmap)
- Provide a CLI entry point that ties everything together
"""


from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.ingestion.load_data import load_data

__all__ = [
    "DataLoadError",
    "EDAConfig",
    "resolve_target_column",
    "outlier_summary",
    "build_eda_report",
    "write_report_markdown",
    "load_dataset",
    "plot_target_distribution",
    "correlation_heatmap",
    "parse_args",
    "run_eda",
    "main",
]


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
# No basicConfig() at import time -- this is a library-shaped module.
# Logging is configured only in main(), where this file genuinely acts
# as "the application".

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

DEFAULT_FILENAME = "Telco.csv"
DEFAULT_TARGET_COL = "Churn"
DEFAULT_FIGURE_DIR = Path("reports/figures")
DEFAULT_REPORT_PATH = Path("reports/eda_report.md")


# ---------------------------------------------------------
# Exceptions
# ---------------------------------------------------------

class DataLoadError(RuntimeError):
    """Raised when the dataset cannot be loaded for EDA purposes."""


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

@dataclass
class EDAConfig:
    """Configuration for a single EDA run."""

    filename: str = DEFAULT_FILENAME
    target_col: str = DEFAULT_TARGET_COL
    figure_dir: Path = DEFAULT_FIGURE_DIR
    report_path: Path = DEFAULT_REPORT_PATH
    show: bool = True


# ---------------------------------------------------------
# Target handling
# ---------------------------------------------------------

def resolve_target_column(df: pd.DataFrame, target_col: str) -> str | None:
    """
    Find the target column, ignoring case.

    Returns the actual column name as it appears in `df`, or None if
    no case-insensitive match exists.
    """
    if target_col in df.columns:
        return target_col

    matches = [col for col in df.columns if col.lower() == target_col.lower()]

    if matches:
        logger.warning("Using '%s' instead of '%s'.", matches[0], target_col)
        return matches[0]

    logger.error("Target column '%s' not found.", target_col)
    return None


# ---------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------

def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count IQR-based outliers for every numeric column.

    Returns a DataFrame indexed by column name with a single
    'outlier_count' column. Empty if `df` has no numeric columns.
    """
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return pd.DataFrame(columns=["outlier_count"])

    counts: dict[str, int] = {}

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()

        if series.empty:
            counts[col] = 0
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        counts[col] = int(((series < lower) | (series > upper)).sum())

    return pd.DataFrame.from_dict(counts, orient="index", columns=["outlier_count"])


# ---------------------------------------------------------
# Report building
# ---------------------------------------------------------

def build_eda_report(df: pd.DataFrame, target_col: str) -> dict:
    """
    Build a structured EDA report as a dict.

    Keys: shape, missing, duplicates, outliers, target_col,
    target_distribution.
    """
    resolved_target = resolve_target_column(df, target_col)

    missing = df.isnull().sum().to_dict()
    duplicates = int(df.duplicated().sum())
    outliers = outlier_summary(df)

    if resolved_target is not None:
        target_distribution = (
            df[resolved_target].value_counts(normalize=True).to_dict()
        )
    else:
        target_distribution = None

    return {
        "shape": df.shape,
        "missing": missing,
        "duplicates": duplicates,
        "outliers": outliers,
        "target_col": resolved_target,
        "target_distribution": target_distribution,
    }


def write_report_markdown(report: dict, path: str | Path) -> Path:
    """Write an EDA report dict to a Markdown file, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows, cols = report["shape"]
    lines: list[str] = [
        "# EDA Report",
        "",
        f"**Shape:** {rows} rows x {cols} columns",
        "",
        "## Missing values",
    ]

    missing = {k: v for k, v in report["missing"].items() if v}
    if missing:
        lines.extend(f"- {col}: {count}" for col, count in missing.items())
    else:
        lines.append("No missing values.")

    lines += ["", "## Duplicates", f"{report['duplicates']} duplicate rows found."]

    outliers = report.get("outliers")
    if outliers is not None and not outliers.empty:
        lines += ["", "## Outliers"]
        lines.extend(
            f"- {col}: {int(row['outlier_count'])}"
            for col, row in outliers.iterrows()
        )

    target_col = report.get("target_col")
    lines += ["", f"## {target_col} distribution" if target_col else "## Target distribution"]

    if target_col and report.get("target_distribution"):
        lines.extend(
            f"- {value}: {pct:.2%}"
            for value, pct in report["target_distribution"].items()
        )
    else:
        lines.append("Target column not found.")

    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def load_dataset(filename: str) -> pd.DataFrame:
    """
    Load the dataset via the shared data loader, wrapping failures in
    a single, predictable DataLoadError.
    """
    try:
        df = load_data(filename)
    except FileNotFoundError as exc:
        logger.error("File '%s' not found.", filename)
        raise DataLoadError(f"File '{filename}' not found.") from exc
    except Exception as exc:  # noqa: BLE001 - deliberately broad, re-raised typed
        logger.exception("Failed loading '%s'.", filename)
        raise DataLoadError(f"Failed to load '{filename}': {exc}") from exc

    if df.empty:
        logger.warning("Dataset '%s' is empty.", filename)

    return df


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
    show: bool = True,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
) -> Path | None:
    """
    Plot and save the target column's distribution.

    Returns the output path, or None if the target column can't be
    resolved.
    """
    resolved_target = resolve_target_column(df, target_col)
    if resolved_target is None:
        return None

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=resolved_target, data=df, ax=ax)
    ax.set_title(f"{resolved_target} Distribution")
    ax.set_xlabel(resolved_target)
    ax.set_ylabel("Count")

    total = len(df)
    for bar in ax.patches:
        height = bar.get_height()
        pct = (height / total * 100) if total else 0
        ax.annotate(
            f"{pct:.1f}%",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output = figure_dir / f"{resolved_target.lower()}_distribution.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    logger.info("Saved figure: %s", output)
    return output


def correlation_heatmap(
    df: pd.DataFrame,
    show: bool = True,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
) -> Path | None:
    """
    Plot and save a correlation heatmap of numeric columns.

    Returns None (and skips plotting) if fewer than two numeric
    columns are present.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        logger.info("Not enough numeric columns for a correlation heatmap.")
        return None

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()

    output = figure_dir / "correlation_heatmap.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    logger.info("Saved figure: %s", output)
    return output


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the EDA script."""
    parser = argparse.ArgumentParser(description="Run EDA on the Telco churn dataset.")
    parser.add_argument(
        "filename",
        nargs="?",
        default=EDAConfig().filename,
        help="CSV filename to load (relative to data/raw/).",
    )
    parser.add_argument(
        "--target-col",
        dest="target_col",
        default=EDAConfig().target_col,
        help="Name of the target column.",
    )
    parser.add_argument(
        "--no-show",
        dest="no_show",
        action="store_true",
        help="Do not display plots interactively.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------

def run_eda(config: EDAConfig) -> dict:
    """Run the full EDA pipeline for a given config and return the report."""
    df = load_dataset(config.filename)
    report = build_eda_report(df, target_col=config.target_col)

    write_report_markdown(report, config.report_path)
    plot_target_distribution(
        df, target_col=config.target_col, show=config.show, figure_dir=config.figure_dir
    )
    correlation_heatmap(df, show=config.show, figure_dir=config.figure_dir)

    return report


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args(argv)
    config = EDAConfig(
        filename=args.filename,
        target_col=args.target_col,
        show=not args.no_show,
    )

    try:
        run_eda(config)
    except DataLoadError as exc:
        logger.error("EDA aborted: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()