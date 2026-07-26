"""
Production-grade CSV data loading utility.

Responsibilities:
- Resolve the project root safely, lazily (no work done at import time)
- Validate dataset paths against traversal / absolute-path injection,
  including symlink-based escapes (defense in depth via resolve())
- Load CSV files, with optional multi-encoding fallback
- Validate schema (required columns)
- Provide precise, chainable exceptions for every failure mode
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from pathlib import Path
from typing import Any
import logging
import os

import pandas as pd

__all__ = [
    "load_data",
    "get_base_dir",
    "PathSecurityError",
    "ProjectRootNotFoundError",
]


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
# Library code should never call logging.basicConfig() (or otherwise
# configure the root logger) as an import side effect -- that decision
# belongs to whichever application imports this module. We attach a
# NullHandler only, so nothing is emitted -- and nothing warns about
# missing handlers -- until the consumer configures logging itself.

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------
# Exceptions
# ---------------------------------------------------------

class DataLoaderError(Exception):
    """Base error for data loading."""


class ProjectRootNotFoundError(RuntimeError, DataLoaderError):
    """Raised when the project root cannot be auto-detected."""
    pass


class PathSecurityError(ValueError, DataLoaderError):
    """Raised when a supplied path attempts to escape the data directory."""
    pass


class DatasetLoadError(RuntimeError, DataLoaderError):
    """Raised when dataset unreadable."""
    pass


class DatasetSchemaError(ValueError, DataLoaderError):
    """Raised when missing required columns."""
    pass

# ---------------------------------------------------------
# Project root detection
# ---------------------------------------------------------


@lru_cache(maxsize=1)
def get_base_dir() -> Path:
    """
    Return the project root directory.

    Priority:
        1. PROJECT_ROOT environment variable.
        2. Auto-detected by walking up from this file looking for a
           pyproject.toml or .git marker.

    The result is cached for the life of the process -- this is a
    filesystem walk whose answer never changes mid-run. Tests that
    mutate PROJECT_ROOT between cases should call
    ``get_base_dir.cache_clear()`` first, or simply pass ``base_dir``
    explicitly to ``load_data`` (preferred -- see its docstring).

    Raises:
        ProjectRootNotFoundError: if no marker is found and
            PROJECT_ROOT is unset.
    """
    env = os.getenv("PROJECT_ROOT")
    if env:
        return Path(env)

    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    raise ProjectRootNotFoundError(
        "Could not auto-detect project root. "
        "Set the PROJECT_ROOT environment variable."
    )


def __getattr__(name: str) -> Path:
    """PEP 562 lazy module attribute.

    Preserves the old ``module.BASE_DIR`` access pattern for backward
    compatibility, without paying the resolution cost -- or crash risk
    -- at import time. Only evaluated if/when someone actually asks
    for it.
    """
    if name == "BASE_DIR":
        return get_base_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------
# Path validation
# ---------------------------------------------------------

def _validate_relative_path(raw_path: str, label: str) -> None:
    """Reject absolute paths and parent-directory traversal segments."""
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise PathSecurityError(
            f"Invalid {label}: {raw_path!r} must be a relative path.")
    if ".." in candidate.parts:
        raise PathSecurityError(
            f"Invalid {label}: {raw_path!r} must not contain '..'.")


# ---------------------------------------------------------
# CSV reading with encoding fallback
# ---------------------------------------------------------

def _read_csv(
    file_path: Path,
    encoding: str | Sequence[str],
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV, retrying alternate encodings only on decode failure.

    Parser/structural errors are never retried across encodings --
    a malformed CSV is malformed regardless of encoding, so retrying
    would just waste time and muddy the eventual error message.
    """
    encodings = [encoding] if isinstance(encoding, str) else list(encoding)
    if not encodings:
        raise ValueError("`encoding` must contain at least one candidate.")

    last_unicode_error: UnicodeDecodeError | None = None

    for i, enc in enumerate(encodings):
        try:
            df = pd.read_csv(file_path, encoding=enc, **read_csv_kwargs)
        except UnicodeDecodeError as e:
            last_unicode_error = e
            logger.debug(
                "Encoding '%s' failed for '%s'; trying next candidate.",
                enc, file_path.name,
            )
            continue
        except pd.errors.EmptyDataError as e:
            logger.exception("CSV file has no columns to parse.")
            raise RuntimeError(
                f"'{file_path.name}' is empty or has no parseable columns."
            ) from e
        except pd.errors.ParserError as e:
            logger.exception("Malformed CSV file.")
            raise RuntimeError(
                f"'{file_path.name}' appears to be malformed.") from e
        except Exception as e:
            logger.exception("Unexpected error while loading CSV.")
            raise RuntimeError(f"Unable to load '{file_path.name}'.") from e
        else:
            if i > 0:
                logger.warning(
                    "Loaded '%s' using fallback encoding '%s' (default '%s' failed).",
                    file_path.name, enc, encodings[0],
                )
            return df

    logger.exception("Encoding error while reading CSV.")
    raise RuntimeError(
        f"Unable to decode '{file_path.name}' using any of {encodings}. "
        "Try another encoding."
    ) from last_unicode_error


# ---------------------------------------------------------
# Public loader
# ---------------------------------------------------------

def load_data(
    filename: str,
    folder: str = "raw",
    encoding: str | Sequence[str] = "utf-8",
    required_columns: Sequence[str] | None = None,
    base_dir: Path | None = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """
    Load a CSV file from the data directory.

    Args:
        filename: Name of the CSV file, relative to `data/<folder>/`.
        folder: Folder inside data/. Defaults to "raw".
        encoding: A single encoding, or an ordered list of encodings
            to try as fallbacks (only on UnicodeDecodeError).
        required_columns: Columns that must exist in the dataset.
        base_dir: Optional custom project root. If omitted, it is
            auto-detected via `get_base_dir()`. Tests should inject
            this explicitly rather than relying on auto-detection.
        **read_csv_kwargs: Passed through to `pandas.read_csv`
            (e.g. dtype, parse_dates, usecols, na_values).

    Returns:
        pandas DataFrame.

    Raises:
        PathSecurityError: If `folder` or `filename` attempts an
            absolute path, parent-directory traversal, or -- via a
            symlink -- resolves outside the data directory.
        FileNotFoundError: If the dataset does not exist.
        RuntimeError: If the file cannot be decoded or parsed.
        ValueError: If `filename` is empty or required columns
            are missing.
    """
    if not filename:
        raise ValueError("filename must not be empty.")

    base_dir = base_dir or get_base_dir()

    # -- Security validation --------------------------------------------
    _validate_relative_path(folder, "folder")
    _validate_relative_path(filename, "filename")

    # -- Build the path, then confirm it still resolves inside data/ ----
    # (String-level checks above stop plain ".." traversal; resolving
    # and comparing real paths also catches symlink-based escapes.)
    data_root = (base_dir / "data").resolve()
    file_path = base_dir / "data" / folder / filename
    resolved_path = file_path.resolve()

    if not resolved_path.is_relative_to(data_root):
        raise PathSecurityError(
            f"Resolved path escapes the data directory: {resolved_path}"
        )

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    # -- Load -------------------------------------------------------
    df = _read_csv(file_path, encoding=encoding, **read_csv_kwargs)

    # -- Validate schema --------------------------------------------
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}")

    logger.info(
        "Loaded '%s' successfully (%d rows x %d columns).",
        filename, len(df), len(df.columns),
    )

    return df


# ---------------------------------------------------------
# Manual execution
# ---------------------------------------------------------

if __name__ == "__main__":
    # basicConfig belongs here, not at module scope -- this is the one
    # place this file is genuinely "the application" rather than a
    # library some other module imports.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        df = load_data(
            filename="Telco.csv",
            required_columns=["customerID", "gender", "Churn"],
        )
    except (PathSecurityError, FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Failed to load dataset: %s", exc)
        raise SystemExit(1) from exc

    print(df.head())
    