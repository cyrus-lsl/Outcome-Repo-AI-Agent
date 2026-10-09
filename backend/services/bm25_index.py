"""BM25 search with two-pass refinement."""
from __future__ import annotations

import pandas as pd

from backend.utils import ensure_combined_text, keyword_search_scored
from config import Config


def two_pass_search(
    df: pd.DataFrame,
    query: str,
    pass1_top_n: int | None = None,
    pass2_top_n: int | None = None,
) -> pd.DataFrame:
    """Pass 1: BM25 over full corpus. Pass 2: BM25 over pass-1 subset."""
    pass1_top_n = pass1_top_n or Config.BM25_PASS1_TOP_N
    pass2_top_n = pass2_top_n or Config.BM25_PASS2_TOP_N

    if not query or not query.strip():
        return ensure_combined_text(df).head(pass2_top_n)

    pass1 = keyword_search_scored(df, query, top_n=pass1_top_n)
    if not pass1:
        return ensure_combined_text(df).head(pass2_top_n)

    df_pass1 = pd.DataFrame([r['instrument'] for r in pass1])
    pass2 = keyword_search_scored(df_pass1, query, top_n=pass2_top_n)
    if not pass2:
        return df_pass1.head(pass2_top_n)

    return pd.DataFrame([r['instrument'] for r in pass2])
