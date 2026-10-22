"""Shared utilities."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


def configure_matplotlib() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "movielens-matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def shorten_title(title: str, maximum: int = 34) -> str:
    return title if len(title) <= maximum else title[: maximum - 1].rstrip() + "…"


def logarithmic_bins(values: np.ndarray, count: int = 26) -> np.ndarray:
    minimum = max(1, int(np.min(values)))
    maximum = int(np.max(values))
    if minimum == maximum:
        return np.array([minimum - 0.5, maximum + 0.5])
    bins = np.unique(np.floor(np.geomspace(minimum, maximum + 1, count)).astype(int))
    bins = bins[bins >= 1]
    return np.append(bins, maximum + 1) if bins[-1] <= maximum else bins

