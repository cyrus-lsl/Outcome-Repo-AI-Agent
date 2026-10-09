"""Compare instruments agent — full row lookup then LLM comparison."""
from __future__ import annotations

import json
import logging
import re

import pandas as pd

from backend.services.data_loader import load_instruments
from backend.services.llm_client import LLMClient
from backend.utils import row_to_result, sanitize_input
from config import Config

logger = logging.getLogger(__name__)


class CompareAgent:
    def __init__(self, llm: LLMClient | None = None, config: type[Config] = Config):
        self.llm = llm or LLMClient(config)
        self.config = config

    def run(self, query: str, entities: dict | None = None, session_context: dict | None = None) -> dict:
        entities = entities or {}
        session_context = session_context or {}
        query = sanitize_input(query)

        names = list(entities.get('compare_targets') or entities.get('instruments') or [])
        if len(names) < 2:
            names.extend(self._extract_names_from_query(query, load_instruments()))
        if len(names) < 2 and session_context.get('last_matched'):
            names = [m.get('name') for m in session_context['last_matched'][:2] if m.get('name')]

        names = self._resolve_names(names, load_instruments())
        if len(names) < 2:
            return {
                'intent': 'compare',
                'text': 'Please name at least **two instruments** to compare (e.g. "Compare PHQ-9 and GAD-7").',
                'matched': [],
            }

        rows = []
        for name in names[:4]:
            row = self._lookup_row(name, load_instruments())
            if row is not None:
                rows.append(row_to_result(row))

        if len(rows) < 2:
            return {
                'intent': 'compare',
                'text': f'Could only find {len(rows)} instrument(s) in the database. Check the names and try again.',
                'matched': rows,
            }

        comparison = self._llm_compare(query, rows)
        return {
            'intent': 'compare',
            'text': comparison,
            'matched': rows,
        }

    def _lookup_row(self, name: str, df: pd.DataFrame) -> pd.Series | None:
        n = name.strip().lower()
        hit = df[df['Measurement Instrument'].astype(str).str.lower() == n]
        if not hit.empty:
            return hit.iloc[0]
        if 'Acronym' in df.columns:
            hit = df[df['Acronym'].astype(str).str.lower() == n]
            if not hit.empty:
                return hit.iloc[0]
        hit = df[df['Measurement Instrument'].astype(str).str.lower().str.contains(n, regex=False)]
        return hit.iloc[0] if not hit.empty else None

    def _resolve_names(self, names: list, df: pd.DataFrame) -> list[str]:
        resolved = []
        seen = set()
        for name in names:
            row = self._lookup_row(name, df)
            if row is not None:
                canonical = str(row.get('Measurement Instrument', '')).strip()
                if canonical.lower() not in seen:
                    seen.add(canonical.lower())
                    resolved.append(canonical)
        return resolved

    def _extract_names_from_query(self, query: str, df: pd.DataFrame) -> list[str]:
        found = []
        query_l = query.lower()
        for _, row in df.iterrows():
            name = str(row.get('Measurement Instrument', '')).strip()
            acronym = str(row.get('Acronym', '')).strip()
            if name and name.lower() in query_l:
                found.append(name)
            elif acronym and len(acronym) >= 3 and re.search(rf'\b{re.escape(acronym.lower())}\b', query_l):
                found.append(name)
        return found

    def _llm_compare(self, query: str, instruments: list[dict]) -> str:
        payload = json.dumps(instruments, ensure_ascii=False, indent=2)
        system = (
            'You compare measurement instruments for programme officers and researchers.\n'
            'Given full instrument records as JSON, write a clear structured comparison in markdown.\n'
            'Cover: purpose, target group, outcome domain, number of items, HK validation, '
            'programme-level suitability, and practical pros/cons.\n'
            'End with a brief recommendation on when to prefer each instrument.'
        )
        user = f"User question: {query}\n\nInstruments:\n{payload}"
        try:
            return self.llm.chat(system, user, max_tokens=1000)
        except Exception as e:
            logger.error('Compare LLM failed: %s', e, exc_info=True)
            return self._fallback_compare(instruments)

    def _fallback_compare(self, instruments: list[dict]) -> str:
        parts = ['**Instrument Comparison**\n']
        for ins in instruments:
            parts.append(
                f"### {ins['name']} ({ins.get('acronym', '')})\n"
                f"- **Domain:** {ins.get('domain', '—')}\n"
                f"- **Target:** {ins.get('target', '—')}\n"
                f"- **Items:** {ins.get('no_of_items', '—')}\n"
                f"- **HK Validated:** {ins.get('validated', '—')}\n"
                f"- **Programme-level:** {ins.get('programme_level', '—')}\n"
            )
        return '\n'.join(parts)
