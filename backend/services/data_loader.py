"""Load and cache Excel / handbook data."""
from __future__ import annotations

import pathlib
from functools import lru_cache

import pandas as pd

from backend.utils import ensure_combined_text
from config import Config


def _resolve_path(path: str, root: pathlib.Path | None = None) -> str:
    if str(path).startswith(('http://', 'https://')):
        return path
    p = pathlib.Path(path)
    if p.is_absolute():
        return str(p)
    base = root or pathlib.Path(__file__).resolve().parent.parent.parent
    return str((base / path).resolve())


@lru_cache(maxsize=1)
def load_instruments() -> pd.DataFrame:
    path = _resolve_path(Config.EXCEL_FILE_PATH)
    df = pd.read_excel(path, sheet_name=Config.EXCEL_SHEET_NAME).fillna('')
    return ensure_combined_text(df)


@lru_cache(maxsize=1)
def load_project_usage() -> pd.DataFrame:
    path = _resolve_path(Config.PROJECT_USAGE_PATH)
    return pd.read_excel(path).fillna('')


@lru_cache(maxsize=1)
def load_handbook() -> str:
    path = _resolve_path(Config.HANDBOOK_PATH)
    with open(path, encoding='utf-8') as f:
        return f.read()


def clear_caches() -> None:
    load_instruments.cache_clear()
    load_project_usage.cache_clear()
    load_handbook.cache_clear()
