"""Shared helpers for instrument search."""
import re

import pandas as pd
from rank_bm25 import BM25Okapi

COMBINE_COLS = (
    'Measurement Instrument', 'Acronym', 'Purpose',
    'Target Group(s)', 'Outcome Domain',
)


def ensure_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    if 'combined_text' in df.columns:
        return df
    parts = [df[c].astype(str) for c in COMBINE_COLS if c in df.columns]
    if parts:
        df = df.copy()
        df['combined_text'] = pd.concat(parts, axis=1).apply(
            lambda row: ' '.join(v for v in row if v and v != 'nan'), axis=1,
        )
    return df


def keyword_search_scored(df: pd.DataFrame, query: str, top_n: int = 10) -> list:
    """Return [{'instrument': Series, 'score': float}, ...] sorted by relevance."""
    df = ensure_combined_text(df).reset_index(drop=True)

    texts = df['combined_text'].fillna('').astype(str).str.lower().tolist()
    corpus = [text.split() for text in texts]

    query_tokens = [t for t in str(query).lower().split() if len(t) > 2]
    if not query_tokens:
        return []

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)

    top_idx = scores.argsort()[::-1][:top_n]

    return [
        {'instrument': df.iloc[i], 'score': float(scores[i])}
        for i in top_idx
        if scores[i] > 0
    ]


def keyword_prefilter(df: pd.DataFrame, query: str, top_n: int = 25) -> pd.DataFrame:
    """Fast local pre-filter. Returns top_n rows by keyword overlap."""
    results = keyword_search_scored(df, query, top_n)
    if not results:
        return ensure_combined_text(df)
    return pd.DataFrame([r['instrument'] for r in results])


def is_programme_level_yes(x: object) -> bool:
    return str(x or '').strip().lower() in ('yes', 'y', 'true', '1')


def validated_in_hk(x: object) -> bool:
    s = str(x or '').lower().strip()
    if not s or s in ('-', 'na', 'n/a'):
        return False
    if 'not validated' in s or 'not in hong' in s or re.search(r'not .*hong|\bno\b', s):
        return False
    if s.startswith('yes'):
        return True
    if ('hong' in s or 'hk' in s) and ('valid' in s or 'develop' in s or 'refer' in s):
        return True
    return 'validated' in s and 'hong' in s


def sanitize_input(text: str, max_length: int = 500) -> str:
    if not isinstance(text, str):
        return ''
    text = text.strip()[:max_length]
    return ''.join(c for c in text if ord(c) >= 32 or c in '\n\t')


def field_str(row, col: str) -> str:
    val = row.get(col, '')
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ''
    return str(val).strip()


def row_to_result(row, confidence=None) -> dict:
    name = field_str(row, 'Measurement Instrument')
    return {
        'name': name,
        'acronym': field_str(row, 'Acronym'),
        'purpose': field_str(row, 'Purpose'),
        'target': field_str(row, 'Target Group(s)'),
        'domain': field_str(row, 'Outcome Domain'),
        'no_of_items': field_str(row, 'No. of Questions / Statements'),
        'sample_q1': field_str(row, 'Sample Question / Statement - 1'),
        'sample_q2': field_str(row, 'Sample Question / Statement - 2'),
        'sample_q3': field_str(row, 'Sample Question / Statement - 3'),
        'scale': field_str(row, 'Scale'),
        'scoring': field_str(row, 'Scoring'),
        'validated': field_str(row, 'Validated in Hong Kong'),
        'programme_level': field_str(row, 'Programme-level metric?'),
        'download_eng': field_str(row, 'Download (Eng)'),
        'download_chi': field_str(row, 'Download (Chi)'),
        'citation': field_str(row, 'Citation'),
        'confidence_score': confidence,
    }
