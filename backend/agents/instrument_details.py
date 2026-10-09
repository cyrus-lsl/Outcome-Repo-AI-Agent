"""Instrument details agent — lookup project usage from Excel."""
from __future__ import annotations

import re

import pandas as pd

from backend.services.data_loader import load_instruments, load_project_usage
from backend.utils import sanitize_input

PROJECT_NO_PATTERN = re.compile(r'\bP\d{4}-\d{3}\b', re.IGNORECASE)


def _normalize(s: str) -> str:
    return str(s or '').strip().lower()


class InstrumentDetailsAgent:
    def run(self, query: str, entities: dict | None = None) -> dict:
        entities = entities or {}
        query = sanitize_input(query)
        usage_df = load_project_usage()
        instruments_df = load_instruments()

        project_no = entities.get('project_no')
        if not project_no:
            m = PROJECT_NO_PATTERN.search(query)
            project_no = m.group(0).upper() if m else None

        instrument_names = entities.get('instruments') or []
        rows = pd.DataFrame()

        if project_no and 'Project No' in usage_df.columns:
            rows = usage_df[usage_df['Project No'].astype(str).str.upper() == project_no.upper()]
            if rows.empty:
                return {
                    'intent': 'instrument_details',
                    'text': f'No usage records found for project **{project_no}**.',
                    'matched': [],
                    'details': [],
                }
            text = self._format_by_project(rows, project_no)
            details = rows.to_dict('records')
        elif instrument_names:
            rows = self._filter_by_instruments(usage_df, instrument_names)
            if rows.empty:
                names = ', '.join(instrument_names)
                return {
                    'intent': 'instrument_details',
                    'text': f'No project usage records found for: **{names}**.',
                    'matched': [],
                    'details': [],
                }
            text = self._format_by_instrument(rows, instrument_names[0])
            details = rows.to_dict('records')
        else:
            matched_name = self._match_instrument_from_query(query, instruments_df)
            if matched_name:
                rows = usage_df[
                    usage_df['Measurement Instrument'].astype(str).str.lower() == matched_name.lower()
                ]
                if not rows.empty:
                    text = self._format_by_instrument(rows, matched_name)
                    details = rows.to_dict('records')
                    return {
                        'intent': 'instrument_details',
                        'text': text,
                        'matched': [],
                        'details': details,
                    }

            if 'Actual Use in Project' in usage_df.columns:
                rows = usage_df.copy()
                text = self._format_simple_usage(rows)
                details = rows.to_dict('records')
                return {
                    'intent': 'instrument_details',
                    'text': text,
                    'matched': [],
                    'details': details,
                }

            return {
                'intent': 'instrument_details',
                'text': (
                    'Please provide a **project number** (e.g. P2024-001) or an **instrument name** '
                    'so I can look up usage records.'
                ),
                'matched': [],
                'details': [],
            }

        return {
            'intent': 'instrument_details',
            'text': text,
            'matched': [],
            'details': details,
        }

    def _filter_by_instruments(self, usage_df: pd.DataFrame, names: list) -> pd.DataFrame:
        mask = pd.Series(False, index=usage_df.index)
        for name in names:
            n = _normalize(name)
            if 'Measurement Instrument' in usage_df.columns:
                mask |= usage_df['Measurement Instrument'].astype(str).str.lower().str.contains(n, regex=False)
            if 'Acronym' in usage_df.columns:
                mask |= usage_df['Acronym'].astype(str).str.lower() == n
        return usage_df[mask]

    def _format_simple_usage(self, rows: pd.DataFrame) -> str:
        if rows.empty:
            return 'No project usage data available.'
        lines = ['**Project usage records**\n']
        for i, record in enumerate(rows.head(10).to_dict('records'), 1):
            instrument = str(record.get('Measurement Instrument', '')).strip()
            acronym = str(record.get('Acronym', '')).strip()
            outcome_domain = str(record.get('Outcome Domain', '')).strip()
            actual_use = str(record.get('Actual Use in Project', '')).strip()
            description_parts = []
            if outcome_domain:
                description_parts.append(f'Outcome domain: {outcome_domain}')
            if actual_use:
                description_parts.append(f'Actual use: {actual_use}')
            description = '; '.join(description_parts)
            if acronym:
                lines.append(f'{i}. **{instrument}** ({acronym}) — {description}')
            else:
                lines.append(f'{i}. **{instrument}** — {description}')
        return '\n'.join(lines)

    def _match_instrument_from_query(self, query: str, instruments_df: pd.DataFrame) -> str | None:
        query_l = query.lower()
        for _, row in instruments_df.iterrows():
            name = str(row.get('Measurement Instrument', '')).strip()
            acronym = str(row.get('Acronym', '')).strip()
            if name and name.lower() in query_l:
                return name
            if acronym and len(acronym) >= 3 and acronym.lower() in query_l:
                return name
        return None

    def _format_by_project(self, rows: pd.DataFrame, project_no: str) -> str:
        project_name = str(rows.iloc[0].get('Project Name', '')).strip()
        header = f"**Project {project_no}**"
        if project_name:
            header += f" — {project_name}"
        header += f"\n\nFound **{len(rows)}** instrument(s):\n\n"
        parts = [header]
        for _, row in rows.iterrows():
            parts.append(self._row_summary(row))
        return '\n\n'.join(parts)

    def _format_by_instrument(self, rows: pd.DataFrame, instrument_name: str) -> str:
        if 'Actual Use in Project' in rows.columns:
            header = f"**Usage records for {instrument_name}**\n\n"
            parts = [header]
            for _, row in rows.iterrows():
                actual_use = str(row.get('Actual Use in Project', '')).strip()
                acronym = str(row.get('Acronym', '')).strip()
                outcome_domain = str(row.get('Outcome Domain', '')).strip()
                description_parts = []
                if outcome_domain:
                    description_parts.append(f'Outcome domain: {outcome_domain}')
                if actual_use:
                    description_parts.append(f'Actual use: {actual_use}')
                description = '; '.join(description_parts)
                if acronym:
                    parts.append(f"**{instrument_name}** ({acronym}) — {description}")
                else:
                    parts.append(f"**{instrument_name}** — {description}")
            return '\n\n'.join(parts)

        header = f"**Usage records for {instrument_name}**\n\nFound in **{len(rows)}** project(s):\n\n"
        parts = [header]
        for _, row in rows.iterrows():
            parts.append(self._row_summary(row, show_project=True))
        return '\n\n'.join(parts)

    def _row_summary(self, row, show_project: bool = False) -> str:
        lines = []
        if show_project:
            lines.append(f"**{row.get('Project No', '')}** — {row.get('Project Name', '')}")
        lines.append(f"**Instrument:** {row.get('Measurement Instrument', '')} ({row.get('Acronym', '')})")
        if row.get('Outcome Domain'):
            lines.append(f"**Outcome Domain:** {row.get('Outcome Domain')}")
        if row.get('Usage Context'):
            lines.append(f"**Usage:** {row.get('Usage Context')}")
        if row.get('Target Group'):
            lines.append(f"**Target Group:** {row.get('Target Group')}")
        if row.get('Collection Frequency'):
            lines.append(f"**Frequency:** {row.get('Collection Frequency')}")
        if row.get('Notes'):
            lines.append(f"**Notes:** {row.get('Notes')}")
        return '\n'.join(lines)
