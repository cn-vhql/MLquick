from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError(f"不支持的文件格式: {path.suffix}")
