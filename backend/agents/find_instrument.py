"""Find instrument agent — BM25 2x prefilter then LLM scoring."""
from __future__ import annotations

import json
import logging
import re

import pandas as pd

from backend.services.bm25_index import two_pass_search
from backend.services.data_loader import load_instruments, load_project_usage
from backend.services.llm_client import LLMClient
from backend.utils import (
    field_str,
    is_programme_level_yes,
    keyword_search_scored,
    row_to_result,
    sanitize_input,
    validated_in_hk,
)
from config import Config

logger = logging.getLogger(__name__)


def _is_hk_validated_request(prompt_l: str) -> bool:
    return bool(re.search(
        r'(?:validated|validate).*(?:hong(?:\s+kong)?|hk)'
        r'|(?:hong(?:\s+kong)?|hk).*(?:validated|validate)'
        r'|\bhk[\s-]?validated\b'
        r'|\bhong\s+kong[\s-]?validated\b',
        prompt_l,
    ))


def _is_programme_level_request(prompt_l: str) -> bool:
    return bool(re.search(
        r'programme[\s-]?level(?:\s+metrics?)?'
        r'|program[\s-]?level(?:\s+metrics?)?'
        r'|\bprogramme[\s-]?level\b',
        prompt_l,
    ))


def _strip_filter_terms(prompt: str) -> str:
    text = prompt
    for pattern in (
        r',?\s*programme[\s-]?level(?:\s+metrics?)?',
        r',?\s*program[\s-]?level(?:\s+metrics?)?',
        r',?\s*(?:validated|validate)\s+in\s+(?:hong\s+kong|hk)',
        r',?\s*(?:hong\s+kong|hk)[\s-]?(?:validated|validate)',
        r',?\s*hk[\s-]?validated',
    ):
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip(' ,;')


def _parse_item_constraints(prompt_l: str):
    max_items = min_items = None
    for pattern in (
        r'(?:not\s+more\s+than|maximum|max|at\s+most|under|below)\s+(\d+)',
        r'(\d+)\s*(?:or\s+)?(?:fewer|less)',
    ):
        if m := re.search(pattern, prompt_l):
            max_items = int(m.group(1))
            break
    for pattern in (
        r'(?:at\s+least|minimum|min|more\s+than|over|above)\s+(\d+)',
        r'(\d+)\s*(?:or\s+)?(?:more|greater)',
    ):
        if m := re.search(pattern, prompt_l):
            min_items = int(m.group(1))
            break
    if not max_items and not min_items:
        if m := re.search(r'(?:exactly|precisely)\s+(\d+)\s*(?:items?|questions?)', prompt_l):
            max_items = min_items = int(m.group(1))
    return max_items, min_items


def _build_instrument_catalog(df: pd.DataFrame):
    lines, names = [], []
    for _, row in df.iterrows():
        name = str(row.get('Measurement Instrument', '')).strip()
        if not name:
            continue
        domain = str(row.get('Outcome Domain', '')).strip()[:50]
        target = str(row.get('Target Group(s)', '')).strip()[:50]
        purpose = str(row.get('Purpose', '')).strip()[:100]
        items = str(row.get('No. of Questions / Statements', '')).strip() or 'N/A'
        prog = str(row.get('Programme-level metric?', '')).strip() or 'N/A'
        hk = 'Yes' if validated_in_hk(row.get('Validated in Hong Kong', '')) else 'No'
        lines.append(f'{name}|{domain}|{target}|{purpose}|{items}|{prog}|{hk}')
        names.append(name)
    return lines, names


def _parse_llm_names(raw: str, instrument_names: list | None = None):
    allowed = {n.lower(): n for n in (instrument_names or [])}
    confidence_scores = {}
    text = raw.strip()

    if '```' in text:
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text).strip()

    names = []
    try:
        parsed = json.loads(text)
        items = parsed if isinstance(parsed, list) else parsed.get('results', parsed.get('instruments', []))
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    name = str(item.get('name', '')).strip().split('|')[0].strip()
                    if name:
                        names.append(name)
                        if item.get('confidence') is not None:
                            confidence_scores[name.lower()] = float(item['confidence'])
                elif isinstance(item, str) and item.strip():
                    names.append(item.strip().split('|')[0].strip())
    except (json.JSONDecodeError, AttributeError):
        pass

    if not names:
        if m := re.search(r'\[[\s\S]*?\]', text):
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and item.get('name'):
                            names.append(str(item['name']).strip().split('|')[0].strip())
                        elif isinstance(item, str):
                            names.append(item.strip().split('|')[0].strip())
            except json.JSONDecodeError:
                pass

    if not names:
        for line in text.split('\n'):
            if m := re.match(r'^\s*\d+[\.\):\-]\s*(.+)$', line.strip()):
                candidate = m.group(1).strip().strip('"\'')
                if candidate and len(candidate) < 200:
                    names.append(candidate.split('|')[0].strip())

    if not allowed:
        return [], confidence_scores

    names = [allowed[n.lower()] for n in names if n.lower() in allowed]
    if not names:
        text_l = text.lower()
        for key, original in allowed.items():
            if key in text_l:
                names.append(original)

    return names, confidence_scores


def _match_names_to_rows(names, df_lookup, instrument_names, max_results, confidence_scores):
    allowed = {str(n).strip().lower() for n in instrument_names}
    matched, unknown, seen = [], [], set()
    for nm in names:
        key = nm.strip().lower()
        if key not in allowed:
            unknown.append(nm)
            continue
        hit = df_lookup[df_lookup['Measurement Instrument'].str.lower() == key]
        if hit.empty or len(matched) >= max_results:
            continue
        row = hit.iloc[0]
        inst_name = field_str(row, 'Measurement Instrument')
        if inst_name.lower() in seen:
            continue
        seen.add(inst_name.lower())
        matched.append(row_to_result(row, confidence_scores.get(key)))
    return matched, unknown


def _filter_dataframe(df, want_validated, want_prog):
    df_search = df
    if want_validated and 'Validated in Hong Kong' in df_search.columns:
        df_search = df_search[df_search['Validated in Hong Kong'].apply(validated_in_hk)]
    if want_prog and 'Programme-level metric?' in df_search.columns:
        df_search = df_search[df_search['Programme-level metric?'].apply(is_programme_level_yes)]
    return df_search


def _enforce_filter_flags(matched, want_validated, want_prog):
    if want_validated:
        matched = [m for m in matched if validated_in_hk(m.get('validated', ''))]
    if want_prog:
        matched = [m for m in matched if is_programme_level_yes(m.get('programme_level', ''))]
    return matched


def _rank_from_filtered_df(search_query, df_filtered, max_results):
    results = keyword_search_scored(df_filtered, search_query, top_n=max_results) if search_query else []
    if not results:
        reset = df_filtered.reset_index(drop=True)
        results = [{'instrument': reset.iloc[i], 'score': 0} for i in range(min(len(reset), max_results))]
    return [
        row_to_result(r['instrument'], confidence=min(95, 60 + r['score'] * 15) if r['score'] else 55)
        for r in results
    ]


def _apply_post_filters(matched, want_validated_hk, want_prog, max_items, min_items):
    if want_validated_hk:
        matched = [i for i in matched if validated_in_hk(i.get('validated', ''))]
        if not matched:
            return [], 'No instruments validated in Hong Kong found matching the query.'

    if want_prog:
        matched = [i for i in matched if is_programme_level_yes(i.get('programme_level', ''))]
        if not matched:
            return [], 'No programme-level instruments found matching the query.'

    if max_items is None and min_items is None:
        return matched, None

    def parse_count(x):
        if not x or pd.isna(x):
            return None
        if m := re.search(r'(\d+)', str(x).strip()):
            return int(m.group(1))
        return None

    filtered = [
        i for i in matched
        if (c := parse_count(i.get('no_of_items'))) is not None
        and (max_items is None or c <= max_items)
        and (min_items is None or c >= min_items)
    ]
    if filtered:
        return filtered, None

    parts = []
    if max_items is not None:
        parts.append(f'maximum {max_items} items')
    if min_items is not None:
        parts.append(f'minimum {min_items} items')
    return [], f'Found {len(matched)} match(es), but none meet the {" and ".join(parts)} constraint.'


def format_matched_text(matched: list) -> str:
    parts = []
    for ins in matched:
        part = f"**{ins['name']}" + (f" ({ins['acronym']})" if ins['acronym'] else '') + f"** — {ins['domain']}\n\n"
        if ins.get('confidence_score') is not None:
            c = ins['confidence_score']
            label = '🟢 Excellent' if c >= 80 else '🟡 Good' if c >= 60 else '🟠 Fair' if c >= 40 else '🔴 Low'
            part += f"**{label} match** (Score: {c:.0f}/100)  \n"
        part += f"**Purpose:** {ins['purpose']}  \n**Target:** {ins['target']}  \n"
        if ins['no_of_items']:
            part += f"**Items:** {ins['no_of_items']}  \n"
        part += f"**Validated in HK:** {ins.get('validated') or '—'}  \n"
        part += f"**Programme-level metric?:** {ins.get('programme_level') or '—'}  \n"
        parts.append(part)
    return '\n\n'.join(parts)


class FindInstrumentAgent:
    def __init__(self, llm: LLMClient | None = None, config: type[Config] = Config):
        self.llm = llm or LLMClient(config)
        self.config = config

    def run(
        self,
        query: str,
        entities: dict | None = None,
        *,
        validated_only: bool = False,
        prog_only: bool = False,
        max_results: int | None = None,
        session_context: dict | None = None,
    ) -> dict:
        if session_context is not None:
            return self._run_why(query, session_context)

        del entities  # unused for find branch
        max_results = max_results or self.config.MAX_RESULTS
        prompt = sanitize_input(query, max_length=self.config.MAX_QUERY_LENGTH)
        if not prompt:
            return {'intent': 'find_instrument', 'text': 'Please provide a valid search query.', 'matched': [], 'unknown': []}

        prompt_l = prompt.lower()
        want_validated = validated_only or _is_hk_validated_request(prompt_l)
        want_prog = prog_only or _is_programme_level_request(prompt_l)
        max_items, min_items = _parse_item_constraints(prompt_l)
        search_query = _strip_filter_terms(prompt) or prompt

        df = load_instruments()
        df_search = _filter_dataframe(df, want_validated, want_prog)
        has_filters = want_validated or want_prog

        if has_filters:
            df_for_llm = df_search
        else:
            df_for_llm = two_pass_search(df_search, search_query)

        inst_lines, instrument_names = _build_instrument_catalog(df_for_llm)
        df_lookup = df_for_llm.reset_index(drop=True)
        filters = {'hk_validated': want_validated, 'programme_level': want_prog}

        if not inst_lines:
            if want_validated and want_prog:
                msg = 'No HK-validated programme-level instruments match your criteria.'
            elif want_validated:
                msg = 'No HK-validated instruments in the database match your criteria.'
            elif want_prog:
                msg = 'No programme-level instruments in the database match your criteria.'
            else:
                msg = 'No instruments in the database.'
            return {'intent': 'find_instrument', 'text': msg, 'matched': [], 'unknown': [], 'filters': filters}

        item_note = ''
        if max_items == min_items and max_items is not None:
            item_note = f'User wants exactly {max_items} items.\n'
        elif max_items is not None or min_items is not None:
            if max_items is not None:
                item_note += f'At most {max_items} items.\n'
            if min_items is not None:
                item_note += f'At least {min_items} items.\n'

        filter_context = ''
        if want_validated:
            filter_context += 'All instruments below are already HK-validated. '
        if want_prog:
            filter_context += 'All instruments below are already programme-level metrics. '

        name_list = '\n'.join(f'- {n}' for n in instrument_names)
        system = (
            'You are a JSON-only selection API. You must NOT write explanations, analysis, or markdown.\n'
            f'{filter_context}'
            f'Pick up to {max_results} instruments from the ALLOWED NAMES list that best match the query.\n'
            'Copy each name EXACTLY as it appears in ALLOWED NAMES.\n'
            'Output ONLY a JSON array like: [{"name": "Exact Name", "confidence": 85}]\n'
            'If none match, output: []'
        )
        user = (
            f"ALLOWED NAMES:\n{name_list}\n\n"
            f"DETAILS (Name|Domain|Target|Purpose|Items|ProgrammeLevel|HKValidated):\n"
            f"{chr(10).join(inst_lines)}\n\n"
            f"Query: {search_query}\n"
            + item_note
            + f"\nRespond with JSON array only. Max {max_results} results."
        )
        retry_user = (
            f"ALLOWED NAMES:\n{name_list}\n\n"
            f"Query: {search_query}\n\n"
            'Return ONLY a JSON array: [{"name": "<exact name from ALLOWED NAMES>", "confidence": 85}]. '
            'No explanations.'
        )

        matched = []
        try:
            raw = self.llm.chat(system, user, max_tokens=400)
            names, confidence_scores = _parse_llm_names(raw, instrument_names)
            matched, _ = _match_names_to_rows(names, df_lookup, instrument_names, max_results, confidence_scores)
            if not matched:
                raw = self.llm.chat(system, retry_user, max_tokens=400)
                names, confidence_scores = _parse_llm_names(raw, instrument_names)
                matched, _ = _match_names_to_rows(names, df_lookup, instrument_names, max_results, confidence_scores)
        except Exception as e:
            logger.error('LLM call failed: %s', e, exc_info=True)
            if not has_filters:
                matched = _rank_from_filtered_df(search_query, df_search, max_results)
                if matched:
                    return {
                        'intent': 'find_instrument',
                        'text': format_matched_text(matched),
                        'matched': matched,
                        'unknown': [],
                        'filters': filters,
                    }
                return {
                    'intent': 'find_instrument',
                    'text': f'AI service error: {e}. Check LLM_BASE_URL and LLM_API_KEY.',
                    'matched': [], 'unknown': [], 'filters': filters,
                }

        if not matched and has_filters:
            matched = _rank_from_filtered_df(search_query, df_lookup, max_results)

        if not matched:
            return {
                'intent': 'find_instrument',
                'text': 'No matching instruments found. Try rephrasing your query.',
                'matched': [], 'unknown': [], 'filters': filters,
            }

        matched = _enforce_filter_flags(matched, want_validated, want_prog)
        if not matched:
            return {
                'intent': 'find_instrument',
                'text': 'No instruments passed the HK-validated / programme-level filter.',
                'matched': [], 'unknown': [], 'filters': filters,
            }

        matched, filter_msg = _apply_post_filters(matched, want_validated, want_prog, max_items, min_items)
        if filter_msg:
            return {'intent': 'find_instrument', 'text': filter_msg, 'matched': [], 'unknown': [], 'filters': filters}

        if any(i.get('confidence_score') is not None for i in matched):
            matched.sort(key=lambda x: x.get('confidence_score') or 0, reverse=True)

        return {
            'intent': 'find_instrument',
            'text': format_matched_text(matched),
            'matched': matched,
            'unknown': [],
            'filters': filters,
        }

    def _usage_snippets(self, instrument_name: str) -> list[str]:
        usage_df = load_project_usage()
        name_l = instrument_name.lower()
        rows = usage_df[
            usage_df['Measurement Instrument'].astype(str).str.lower().str.contains(name_l, regex=False)
            | usage_df['Acronym'].astype(str).str.lower().str.contains(name_l, regex=False)
        ] if 'Measurement Instrument' in usage_df.columns else usage_df
        snippets = []
        for _, row in rows.head(3).iterrows():
            if 'Project No' in usage_df.columns:
                project_no = str(row.get('Project No', '')).strip()
                project_name = str(row.get('Project Name', '')).strip()
                usage_context = str(row.get('Usage Context', '')).strip()
                target_group = str(row.get('Target Group', '')).strip()
                frequency = str(row.get('Collection Frequency', '')).strip()
                snippet = f"**{instrument_name}** was used in **{project_no}** ({project_name}) for **{usage_context}**; target group: **{target_group}**; frequency: **{frequency}**"
                notes = str(row.get('Notes', '')).strip()
                if notes and notes.lower() != 'nan':
                    snippet += f"; notes: **{notes}**"
            else:
                actual_use = str(row.get('Actual Use in Project', '')).strip()
                domain = str(row.get('Outcome Domain', '')).strip()
                snippet = f"**{instrument_name}** is recorded as: **{actual_use}**"
                if domain:
                    snippet += f" in the **{domain}** outcome domain"
            snippets.append(snippet)
        return snippets

    def _local_explanation(self, last_query: str, matched: list) -> str:
        if not matched:
            return f'For your earlier query **"{last_query}"**, the recommendation was selected because it best matched the older-adult target group and the domain of daily functioning.'
        first = matched[0]
        name = str(first.get('name') or 'this instrument')
        purpose = first.get('purpose') or 'functional assessment'
        target = first.get('target') or 'older adults'
        domain = first.get('domain') or 'functional independence'
        usage_notes = self._usage_snippets(name)
        usage_context = f" {' '.join(usage_notes)}" if usage_notes else ''
        name_l = name.lower()
        if 'lawton' in name_l:
            return f'**{name}** is a strong choice because it directly measures functional independence in older adults. It focuses on daily living tasks such as using a telephone, managing money, shopping, cooking, housekeeping, and transportation. This makes it highly relevant when recovery is defined as regaining the ability to live independently after illness, discharge, or rehabilitation. {usage_context or "It is therefore better suited than a mood-only or cognition-only measure when the goal is functional recovery."}'
        if 'depression' in name_l or 'geriatric depression' in name_l:
            return f'**{name}** is appropriate when the main outcome is emotional recovery or depressive symptoms in older adults. It was ranked highly because its purpose is to screen for depression, which matches your concern more closely than a general functional measure. {usage_context}'
        if 'montreal' in name_l or 'cognitive' in name_l:
            return f'**{name}** is a strong fit when cognitive recovery is the main outcome. It targets memory, attention, executive function, and orientation, which are important in older adults when cognitive change is the focus. {usage_context}'
        return f'**{name}** was selected because it aligns with the main construct in your question: purpose = {purpose}; target group = {target}; domain = {domain}. It therefore fits the question better than a generic screening tool. {usage_context}'

    def _run_why(self, query: str, session_context: dict) -> dict:
        query = sanitize_input(query)
        last_matched = session_context.get('last_matched') or []
        last_query = session_context.get('last_query', '')
        last_intent = session_context.get('last_intent', '')
        last_response_text = session_context.get('last_response_text', '')
        if not last_matched and last_response_text:
            last_matched = [
                {'name': name.strip(), 'purpose': 'Previous recommendation', 'target': 'Older adults', 'domain': 'Recovery/functional status'}
                for name in re.findall(r'\*\*(.+?)\*\*', last_response_text)[:3]
                if name.strip()
            ]
        if not last_matched and last_query:
            df = load_instruments()
            for _, row in df.iterrows():
                name = str(row.get('Measurement Instrument', '')).strip()
                purpose = str(row.get('Purpose', '')).strip()
                target = str(row.get('Target Group(s)', '')).strip()
                domain = str(row.get('Outcome Domain', '')).strip()
                if any(token in ' '.join([name, purpose, target, domain]).lower() for token in ['elder', 'older', 'recovery', 'functional', 'daily']):
                    last_matched.append({'name': name, 'purpose': purpose, 'target': target, 'domain': domain})
                    break
        if not last_matched or last_intent not in ('find_instrument', 'compare'):
            if last_query and last_response_text:
                return {'intent': 'why', 'text': self._local_explanation(last_query, last_matched), 'matched': last_matched}
            return {'intent': 'why', 'text': 'I need a previous search or comparison to explain recommendations.\n\nTry asking something like *"mental health for elderly"* first, then ask *"why did you recommend these?"*', 'matched': []}
        return {'intent': 'why', 'text': self._local_explanation(last_query, last_matched), 'matched': last_matched}
