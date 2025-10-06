"""Storage helpers for Telegram parser artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


PathLike = Union[str, Path]


def save_parquet(df: pd.DataFrame, path: PathLike) -> None:
    """Persist dataframe to Parquet ensuring directories exist."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)


def load_parquet(path: PathLike) -> pd.DataFrame:
    """Load a parquet file into a DataFrame."""

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Parquet file not found: {target}")
    return pd.read_parquet(target)


__all__ = ["save_parquet", "load_parquet"]

