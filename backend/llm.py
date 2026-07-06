"""Web LLM search — single entry point for AI-powered instrument matching."""
import json
import logging
import re

import pandas as pd
from openai import OpenAI

from backend.utils import (
    is_programme_level_yes,
    keyword_prefilter,
    keyword_search_scored,
    sanitize_input,
    validated_in_hk,
)

logger = logging.getLogger(__name__)

META_PATTERNS = (
    r'\bwhat can you\b', r'\bhow (?:do|can) you\b', r'\bwhat do you do\b',
    r'\bhelp me understand\b', r'\bwho are you\b',
)


def _get_client(config):
    return OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY), config.LLM_MODEL


def _is_meta_question(prompt: str) -> bool:
    p = prompt.lower()
    return any(re.search(pat, p) for pat in META_PATTERNS)


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
    """Remove filter phrases so search focuses on the topic (e.g. 'life')."""
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
    """Parse LLM output into instrument names. Falls back to finding catalog names in text."""
    allowed = {n.lower(): n for n in (instrument_names or [])}
    confidence_scores = {}
    text = raw.strip()

    # Strip markdown fences
    if '```' in text:
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text).strip()

    names = []

    # Try full JSON parse
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

    # Try extracting a JSON array embedded in prose
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

    # Numbered list: "1. Instrument Name"
    if not names:
        for line in text.split('\n'):
            if m := re.match(r'^\s*\d+[\.\):\-]\s*(.+)$', line.strip()):
                candidate = m.group(1).strip().strip('"\'')
                if candidate and len(candidate) < 200:
                    names.append(candidate.split('|')[0].strip())

    # Only keep names that exist in the allowed catalog
    if not allowed:
        return [], confidence_scores

    names = [allowed[n.lower()] for n in names if n.lower() in allowed]
    if not names:
        text_l = text.lower()
        for key, original in allowed.items():
            if key in text_l:
                names.append(original)

    return names, confidence_scores


def _llm_pick(client, model, system, user, max_tokens=400):
    completion = client.chat.completions.create(
        model=model,
        messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return completion.choices[0].message.content


def _enforce_filter_flags(matched, want_validated, want_prog):
    """Drop any result that violates active hard filters (safety net)."""
    if want_validated:
        matched = [m for m in matched if validated_in_hk(m.get('validated', ''))]
    if want_prog:
        matched = [m for m in matched if is_programme_level_yes(m.get('programme_level', ''))]
    return matched


def _rank_from_filtered_df(search_query, df_filtered, max_results):
    """Rank instruments within an already-filtered dataframe (LLM fallback)."""
    results = keyword_search_scored(df_filtered, search_query, top_n=max_results) if search_query else []
    if not results:
        reset = df_filtered.reset_index(drop=True)
        results = [{'instrument': reset.iloc[i], 'score': 0} for i in range(min(len(reset), max_results))]
    return [
        _row_to_result(r['instrument'], confidence=min(95, 60 + r['score'] * 15) if r['score'] else 55)
        for r in results
    ]


def _llm_select_from_catalog(client, model, system, user, retry_user, instrument_names, df_lookup, max_results):
    raw = _llm_pick(client, model, system, user)
    names, confidence_scores = _parse_llm_names(raw, instrument_names)
    matched, _ = _match_names_to_rows(names, df_lookup, instrument_names, max_results, confidence_scores)
    if matched:
        return matched
    raw = _llm_pick(client, model, system, retry_user)
    names, confidence_scores = _parse_llm_names(raw, instrument_names)
    return _match_names_to_rows(names, df_lookup, instrument_names, max_results, confidence_scores)[0]


def _field_str(row, col: str) -> str:
    val = row.get(col, '')
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ''
    return str(val).strip()


def _row_to_result(row, confidence=None):
    name = _field_str(row, 'Measurement Instrument')
    return {
        'name': name,
        'acronym': _field_str(row, 'Acronym'),
        'purpose': _field_str(row, 'Purpose'),
        'target': _field_str(row, 'Target Group(s)'),
        'domain': _field_str(row, 'Outcome Domain'),
        'no_of_items': _field_str(row, 'No. of Questions / Statements'),
        'sample_q1': _field_str(row, 'Sample Question / Statement - 1'),
        'sample_q2': _field_str(row, 'Sample Question / Statement - 2'),
        'sample_q3': _field_str(row, 'Sample Question / Statement - 3'),
        'scale': _field_str(row, 'Scale'),
        'scoring': _field_str(row, 'Scoring'),
        'validated': _field_str(row, 'Validated in Hong Kong'),
        'programme_level': _field_str(row, 'Programme-level metric?'),
        'download_eng': _field_str(row, 'Download (Eng)'),
        'download_chi': _field_str(row, 'Download (Chi)'),
        'citation': _field_str(row, 'Citation'),
        'confidence_score': confidence,
    }


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
        inst_name = _field_str(row, 'Measurement Instrument')
        if inst_name.lower() in seen:
            continue
        seen.add(inst_name.lower())
        matched.append(_row_to_result(row, confidence_scores.get(key)))
    return matched, unknown


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


def _filter_dataframe(df, want_validated, want_prog):
    df_search = df
    if want_validated and 'Validated in Hong Kong' in df_search.columns:
        df_search = df_search[df_search['Validated in Hong Kong'].apply(validated_in_hk)]
    if want_prog and 'Programme-level metric?' in df_search.columns:
        df_search = df_search[df_search['Programme-level metric?'].apply(is_programme_level_yes)]
    return df_search


def _format_text(matched):
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


def search_by_chat(prompt, df, config, validated_only=False, prog_only=False, max_results=8):
    """Call the configured web LLM to find instruments in the dataset."""
    prompt = sanitize_input(prompt, max_length=config.MAX_QUERY_LENGTH)
    if not prompt:
        return {'text': 'Please provide a valid search query.', 'matched': [], 'unknown': []}

    prompt_l = prompt.lower()
    if _is_meta_question(prompt_l):
        return {
            'text': (
                f"I'm an AI assistant that searches your measurement instrument database "
                f"(model: **{config.LLM_MODEL}**).\n\n"
                "**Try asking:**\n"
                "- mental health assessment for elderly\n"
                "- physical activity questionnaire for youth\n"
                "- quality of life scale validated in Hong Kong"
            ),
            'matched': [],
            'unknown': [],
        }

    want_validated = validated_only or _is_hk_validated_request(prompt_l)
    want_prog = prog_only or _is_programme_level_request(prompt_l)
    max_items, min_items = _parse_item_constraints(prompt_l)
    search_query = _strip_filter_terms(prompt) or prompt

    df_search = _filter_dataframe(df, want_validated, want_prog)
    has_filters = want_validated or want_prog

    # Step 1: hard-filter HK validated / programme level (done above as df_search)
    # Step 2: LLM picks from candidates — when filtered, send ALL of df_search (no keyword prefilter)
    if has_filters:
        df_for_llm = df_search
    else:
        prefilter_n = getattr(config, 'PREFILTER_TOP_N', 25)
        df_for_llm = keyword_prefilter(df_search, search_query, top_n=prefilter_n) if prefilter_n else df_search

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
        return {'text': msg, 'matched': [], 'unknown': [], 'filters': filters}

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
        client, model = _get_client(config)
        matched = _llm_select_from_catalog(
            client, model, system, user, retry_user, instrument_names, df_lookup, max_results,
        )
    except Exception as e:
        logger.error('LLM call failed: %s', e, exc_info=True)
        if not has_filters:
            return {'text': f'AI service error: {e}. Check LLM_BASE_URL and LLM_API_KEY.',
                    'matched': [], 'unknown': [], 'filters': filters}

    if not matched and has_filters:
        logger.info('LLM returned no valid picks; ranking within filtered set (%d rows)', len(df_lookup))
        matched = _rank_from_filtered_df(search_query, df_lookup, max_results)

    if not matched:
        return {
            'text': 'No matching instruments found. Try rephrasing, or tick the filter checkboxes above.',
            'matched': [],
            'unknown': [],
            'filters': filters,
        }

    matched = _enforce_filter_flags(matched, want_validated, want_prog)
    if not matched:
        return {
            'text': 'No instruments passed the HK-validated / programme-level filter.',
            'matched': [],
            'unknown': [],
            'filters': filters,
        }

    matched, filter_msg = _apply_post_filters(matched, want_validated, want_prog, max_items, min_items)
    if filter_msg:
        return {'text': filter_msg, 'matched': [], 'unknown': [], 'filters': filters}

    if any(i.get('confidence_score') is not None for i in matched):
        matched.sort(key=lambda x: x.get('confidence_score') or 0, reverse=True)

    return {'text': _format_text(matched), 'matched': matched, 'unknown': [],
            'filters': filters}
