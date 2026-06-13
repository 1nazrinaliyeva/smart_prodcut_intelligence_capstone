from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_product_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "train": load_parquet(processed_dir / "product_train.parquet"),
        "val": load_parquet(processed_dir / "product_val.parquet"),
        "test": load_parquet(processed_dir / "product_test.parquet"),
    }


def load_review_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "train": load_parquet(processed_dir / "train.parquet"),
        "val": load_parquet(processed_dir / "val.parquet"),
        "test": load_parquet(processed_dir / "test.parquet"),
    }
