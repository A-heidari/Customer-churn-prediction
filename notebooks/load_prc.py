from __future__ import annotations
from functools import lru_cache

from pathlib import Path
import logging 
import os 
import pandas as pd 
from collections.abc import Sequence
from typing import Any



__all__ = [
    "load_data",
    "get_base_dir"
    "ProjectRootNotFoundError",
    "PathtSecurityError"
]


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ProjectRootNotFoundError(RuntimeError):
    """Raised ProjectRootNotFoundError by auto-detected."""

class PathSecurityError(ValueError):
    """Raised"""