"""Previous usage agent — answer whether an instrument has been used before."""
from __future__ import annotations

import json

import pandas as pd

try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:  # Keep prompt generation compatible before optional dependencies are installed.
    ChatPromptTemplate = None

from backend.services.data_loader import load_project_usage
from backend.services.llm_client import LLMClient
from backend.utils import sanitize_input
from config import Config


class PreviousUsageAgent:
    """Check if a measurement instrument appears in prior project records."""

    def __init__(self, llm: LLMClient | None = None, config: type[Config] = Config):
        self.llm = llm or LLMClient(config)
        self.config = config

    def run(self, query: str, entities: dict | None = None) -> dict:
        query = sanitize_input(query)
        usage_df = load_project_usage().fillna('')
        entities = entities or {}

        instrument_names = entities.get('instruments') or []
        if not instrument_names:
            instrument_names = self._extract_instruments(query)

        if not instrument_names:
            return {
                'intent': 'instrument_details',
                'text': 'I can check whether an instrument was used before. Please name the instrument or acronym, for example “PHQ-9” or “Lawton”.',
                'matched': [],
                'details': [],
            }

        filtered = self._filter_usage(usage_df, instrument_names)
        if filtered.empty:
            names = ', '.join(instrument_names)
            return {
                'intent': 'instrument_details',
                'text': f'No prior project records were found for **{names}**.',
                'matched': [],
                'details': [],
            }

        text = self._format_response(instrument_names[0], filtered)
        return {
            'intent': 'instrument_details',
            'text': text,
            'matched': filtered.to_dict('records'),
            'details': filtered.to_dict('records'),
        }

    def _extract_instruments(self, query: str) -> list[str]:
        lower = query.lower()
        df = load_project_usage().fillna('')
        matches: list[str] = []
        for _, row in df.iterrows():
            name = str(row.get('Measurement Instrument', '')).strip()
            acronym = str(row.get('Acronym', '')).strip()
            if name and name.lower() in lower:
                matches.append(name)
                continue
            if acronym and acronym.lower() in lower:
                matches.append(name or acronym)
        return matches

    def _filter_usage(self, usage_df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
        if usage_df.empty:
            return usage_df

        mask = pd.Series(False, index=usage_df.index)
        for name in names:
            name_l = str(name).strip().lower()
            if 'Measurement Instrument' in usage_df.columns:
                mask |= usage_df['Measurement Instrument'].astype(str).str.lower().str.contains(name_l, regex=False)
            if 'Acronym' in usage_df.columns:
                mask |= usage_df['Acronym'].astype(str).str.lower().str.contains(name_l, regex=False)

        return usage_df[mask]

    def _format_response(self, instrument_name: str, rows: pd.DataFrame) -> str:
        instrument_name = str(instrument_name).strip() or 'This instrument'
        records = self._compact_records(rows)

        try:
            system_prompt = (
                'You are a careful research assistant. Rewrite the project usage records into a clear, polished answer. '
                'Keep the facts from the records and do not invent anything. Answer in fluent English. '
                'Group identical uses together and do not repeat duplicate records. '
                'State the outcome domain once, then list distinct project uses as concise bullets. '
                'Do not include markdown tables.'
            )
            records_json = json.dumps(records, ensure_ascii=False)
            if ChatPromptTemplate is not None:
                prompt = ChatPromptTemplate.from_messages([
                    ('system', system_prompt),
                    ('human', 'Instrument name: {instrument_name}\n\nProject usage records: {records}'),
                ])
                messages = prompt.format_messages(
                    instrument_name=instrument_name, records=records_json,
                )
                system, user = messages[0].content, messages[1].content
            else:
                system = system_prompt
                user = f'Instrument name: {instrument_name}\n\nProject usage records: {records_json}'
            text = self.llm.chat(
                system,
                user,
                max_tokens=500,
                temperature=0.2,
            )
            if text and text.strip():
                return text.strip()
        except Exception:
            pass

        lines = [f'**Yes — {instrument_name} has been used in previous projects.**\n']
        for i, record in enumerate(records, 1):
            instrument = str(record.get('Measurement Instrument', instrument_name)).strip() or instrument_name
            acronym = str(record.get('Acronym', '')).strip()
            outcome_domain = str(record.get('Outcome Domain', '')).strip()
            actual_use = str(record.get('Actual Use in Project', '')).strip()
            description_parts = []
            if outcome_domain:
                description_parts.append(f'Outcome domain: {outcome_domain}')
            if actual_use:
                description_parts.append(f'Actual use: {actual_use}')
            description = '; '.join(description_parts)
            if acronym and instrument:
                lines.append(f'{i}. **{instrument}** ({acronym}) — {description}')
            elif description:
                lines.append(f'{i}. **{instrument}** — {description}')
            else:
                lines.append(f'{i}. **{instrument}**')
        return '\n'.join(lines)

    def _compact_records(self, rows: pd.DataFrame) -> list[dict]:
        """Remove duplicate domain/use pairs before sending records to the LLM."""
        compacted = []
        seen = set()
        for record in rows.to_dict('records'):
            key = (
                ''.join(str(record.get('Outcome Domain', '')).strip().lower().split()).replace('-', ''),
                str(record.get('Actual Use in Project', '')).strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            compacted.append(record)
        return compacted
