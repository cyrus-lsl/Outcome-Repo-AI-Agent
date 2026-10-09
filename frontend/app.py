import logging
import os
import pathlib
import sys
import uuid

import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.orchestrator import get_orchestrator
from backend.services.data_loader import clear_caches, load_instruments
from config import Config, logger


def load_environment():
    try:
        from dotenv import load_dotenv, find_dotenv

        candidates = []
        root_dir = ROOT
        repo_env = root_dir / '.env'
        if repo_env.is_file():
            candidates.append(repo_env)
        nested_env = repo_env / '.env'
        if nested_env.is_file():
            candidates.append(nested_env)

        for env_path in candidates:
            load_dotenv(env_path, override=False)

        if not candidates:
            env_path = find_dotenv(usecwd=True)
            if env_path:
                load_dotenv(env_path, override=False)
    except ImportError:
        pass


def resolve_path(path: str) -> str:
    if isinstance(path, str) and path.lower().startswith(('http://', 'https://')):
        return path
    if os.path.isabs(path):
        return path
    return str((ROOT / path).resolve())


@st.cache_resource
def initialize_agent():
    try:
        excel_path = resolve_path(Config.EXCEL_FILE_PATH)
        from backend.agent_core import MeasurementInstrumentAgent
        return MeasurementInstrumentAgent(excel_path, sheet_name=Config.EXCEL_SHEET_NAME)
    except Exception as e:
        logger.error('Failed to initialize agent: %s', e, exc_info=True)
        return None


@st.cache_data(ttl=3600)
def load_dataframe(excel_path: str, sheet_name: str):
    return pd.read_excel(excel_path, sheet_name=sheet_name).fillna('')


def save_uploaded_excel(uploaded_file, dest_path: str):
    try:
        with open(dest_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True, None
    except Exception as e:
        logger.error('Failed to save uploaded Excel: %s', e, exc_info=True)
        return False, str(e)


INTENT_LABELS = {
    'find_instrument': '🔍 Find Instrument',
    'instrument_details': '📋 Project Usage',
    'compare': '⚖️ Compare',
    'why': '💡 Explanation',
    'handbook': '📖 Handbook',
}


def _ensure_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


def _display_instrument_cards(response):
    matched = response.get('matched', [])
    if not matched:
        if text := response.get('text', ''):
            st.info(text)
        return

    filters = response.get('filters', {})
    if filters.get('programme_level') or filters.get('hk_validated'):
        tags = []
        if filters.get('hk_validated'):
            tags.append('HK-validated only')
        if filters.get('programme_level'):
            tags.append('Programme-level only')
        st.caption('🔎 Active filters: ' + ', '.join(tags))

    for idx, ins in enumerate(matched, 1):
        title = f"{idx}. {ins.get('name', 'Unknown')}"
        if ins.get('acronym'):
            title += f" ({ins['acronym']})"

        with st.expander(title, expanded=(idx == 1)):
            if ins.get('domain'):
                st.markdown(
                    f'<span class="badge badge-domain">📁 {ins["domain"]}</span>',
                    unsafe_allow_html=True,
                )

            badges = ''
            val = str(ins.get('validated', '')).lower()
            if val and val.startswith('yes') and 'not' not in val:
                badges += '<span class="badge badge-validated">✓ HK Validated</span>'
            if str(ins.get('programme_level', '')).strip().lower() in ('yes', 'y', 'true'):
                badges += '<span class="badge badge-programme">📊 Programme-level</span>'
            if badges:
                st.markdown(badges, unsafe_allow_html=True)

            if ins.get('confidence_score') is not None:
                c = ins['confidence_score']
                color = '#22c55e' if c >= 80 else '#eab308' if c >= 60 else '#f97316' if c >= 40 else '#ef4444'
                label = '🟢 Excellent' if c >= 80 else '🟡 Good' if c >= 60 else '🟠 Fair' if c >= 40 else '🔴 Low'
                st.markdown(
                    f'<div style="background:{color}20;padding:0.5rem;border-radius:0.5rem;'
                    f'border-left:4px solid {color};margin:0.5rem 0;">'
                    f'<strong>{label} match</strong> (Score: {c:.0f}/100)</div>',
                    unsafe_allow_html=True,
                )

            st.markdown('---')
            if ins.get('purpose'):
                st.markdown(f"**📝 Purpose:**  \n{ins['purpose']}")
            if ins.get('target'):
                st.markdown(f"**👥 Target Group:** {ins['target']}")
            st.markdown(f"**✓ Validated in Hong Kong:** {ins.get('validated') or '—'}")
            st.markdown(f"**📊 Programme-level metric?:** {ins.get('programme_level') or '—'}")

            col1, col2 = st.columns(2)
            with col1:
                if ins.get('no_of_items'):
                    st.metric('Items', ins['no_of_items'])
                if ins.get('scale'):
                    st.markdown(f"**Scale:** {ins['scale']}")
            with col2:
                if ins.get('scoring'):
                    st.markdown(f"**Scoring:** {ins['scoring']}")

            samples = [ins[k] for k in ('sample_q1', 'sample_q2', 'sample_q3') if ins.get(k)]
            if samples:
                with st.expander('📋 Sample Questions'):
                    for q in samples:
                        st.markdown(f'• {q}')

            links = []
            if ins.get('download_eng'):
                links.append(f"[📥 Download (English)]({ins['download_eng']})")
            if ins.get('download_chi'):
                links.append(f"[📥 Download (中文)]({ins['download_chi']})")
            if links:
                st.markdown('**Downloads:** ' + ' | '.join(links))
            if ins.get('citation'):
                with st.expander('📚 Citation'):
                    st.markdown(ins['citation'])


def _display_response(response):
    if not isinstance(response, dict):
        st.markdown(response if isinstance(response, str) else str(response))
        return

    intent = response.get('intent', 'find_instrument')
    label = INTENT_LABELS.get(intent, intent)
    tool_summary = response.get('tool_summary') or {}
    tool_name = tool_summary.get('called') or 'unknown_tool'
    route_label = f"Route: {label} • Tool: {tool_name}"
    st.caption(route_label)

    if response.get('status') and response.get('status') != 'ok':
        st.warning(f"Status: {response['status']}")

    if 'text' in response:
        st.markdown(response['text'])

    if response.get('plan') or response.get('tool_calls'):
        with st.expander('🔎 Evidence and tool trace', expanded=False):
            if response.get('plan'):
                st.json(response['plan'])
            if response.get('tool_calls'):
                st.json(response['tool_calls'])

    if intent in ('find_instrument', 'compare'):
        if response.get('matched'):
            st.divider()
            if intent == 'compare':
                st.markdown('**Compared instruments:**')
            else:
                st.markdown('**Top matches:**')
            _display_instrument_cards({'matched': response.get('matched', []), 'filters': {}})
    elif intent == 'instrument_details':
        if response.get('details'):
            st.divider()
            st.markdown('**Project usage evidence:**')
            _display_usage_evidence(response['details'])


def _display_usage_evidence(records):
    grouped = {}
    for record in records:
        domain = str(record.get('Outcome Domain') or 'Outcome domain not specified').strip()
        domain_key = ''.join(domain.lower().split()).replace('-', '')
        actual_use = str(record.get('Actual Use in Project') or 'Use not specified').strip()
        grouped.setdefault(domain_key, {'label': domain, 'uses': set()})['uses'].add(actual_use)

    for group in grouped.values():
        st.markdown(f"**Outcome domain: {group['label']}**")
        for actual_use in sorted(group['uses']):
            st.markdown(f'- {actual_use}')

    with st.expander(f'Show {len(records)} source records'):
        st.json(records)


CHAT_CSS = """
<style>
.badge { display:inline-block; padding:0.2rem 0.6rem; border-radius:12px;
         font-size:0.75rem; font-weight:600; margin:0.2rem 0.2rem 0.2rem 0; }
.badge-domain { background:rgba(28,131,225,0.2); color:#1c83e1; }
.badge-validated { background:rgba(34,197,94,0.2); color:#22c55e; }
.badge-programme { background:rgba(168,85,247,0.2); color:#a855f7; }
.main-title { font-size:2.5rem; font-weight:700; color:#1c83e1; }
.subtitle { color:#6b7280; font-size:1rem; }
</style>
"""


def render_chat_page():
    st.markdown(CHAT_CSS, unsafe_allow_html=True)
    st.subheader('💬 Chat with AI Assistant')
    st.markdown(
        "Ask about instruments, project usage, comparisons, or follow up with *why* questions. "
        "The assistant uses structured tool calls to ground its answer in the dataset."
    )

    _ensure_session_id()

    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message['role']):
            if message['role'] == 'user':
                st.markdown(message['content'])
            else:
                _display_response(message['content'])

    if prompt := st.chat_input('Ask about measurement instruments...'):
        st.session_state.chat_messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        with st.chat_message('assistant'):
            with st.spinner('🤔 Thinking...'):
                orchestrator = get_orchestrator()
                response = orchestrator.handle_message(st.session_state.session_id, prompt)
            _display_response(response)
            st.session_state.chat_messages.append({'role': 'assistant', 'content': response})


def render_manual_search_page(agent):
    st.subheader('🧭 Manual Search')
    beneficiaries_input = st.text_input('👥 Target beneficiaries', placeholder='e.g. youth, elderly')
    beneficiaries = [b.strip() for b in beneficiaries_input.split(',')] if beneficiaries_input.strip() else None
    measure = st.text_input('📊 What are you trying to measure?', placeholder='e.g. mental health')

    col1, col2 = st.columns(2)
    with col1:
        hk_only = st.checkbox('✅ Require HK-validated only')
    with col2:
        prog_only = st.checkbox('📊 Programme-level only')

    if st.button('🔍 Search', type='primary', use_container_width=True):
        with st.spinner('Searching...'):
            results = agent.manual_search(
                beneficiaries=beneficiaries,
                measure=measure,
                validated='yes' if hk_only else 'both',
                prog_level='yes' if prog_only else 'both',
            )
        recs = results.get('recommendations', [])
        if not recs:
            st.info('No matching instruments found. Try adjusting your filters.')
            return

        st.success(f'Found {len(recs)} instrument{"s" if len(recs) != 1 else ""}')
        for idx, ins in enumerate(recs, 1):
            title = f"{idx}. {ins.get('name', 'Unknown')}"
            if ins.get('acronym'):
                title += f" ({ins['acronym']})"
            with st.expander(title, expanded=(idx == 1)):
                if ins.get('purpose'):
                    st.markdown(f"**📝 Purpose:**  \n{ins['purpose']}")
                if ins.get('target_group'):
                    st.markdown(f"**👥 Target Group:** {ins['target_group']}")
                if ins.get('domain'):
                    st.markdown(f"**📁 Domain:** {ins['domain']}")


def render_data_management_page(df, excel_path):
    st.title('📂 Data Management')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric('Total Instruments', len(df))
    with c2:
        domains = df['Outcome Domain'].dropna().unique() if 'Outcome Domain' in df.columns else []
        st.metric('Domains', len(domains))
    with c3:
        try:
            usage = load_instruments()
            st.metric('Cached', 'Ready')
        except Exception:
            st.metric('Cached', 'Error')

    st.code(excel_path, language=None)
    st.divider()

    uploaded = st.file_uploader('Upload Excel (.xlsx) to replace dataset', type=['xlsx'])
    if uploaded is not None:
        ok, err = save_uploaded_excel(uploaded, excel_path)
        if ok:
            load_dataframe.clear()
            clear_caches()
            st.success('✅ File saved. Reloading...')
            st.rerun()
        else:
            st.error(f'❌ Save failed: {err}')

    if st.button('🔄 Refresh Data Now', type='primary', use_container_width=True):
        load_dataframe.clear()
        clear_caches()
        st.rerun()


def main():
    st.set_page_config(page_title='Measurement Instrument Assistant', page_icon='📊', layout='wide')
    st.markdown(
        '<h1 class="main-title">📊 Measurement Instrument Assistant</h1>'
        '<p class="subtitle">Multi-branch AI agent for research measurement instruments</p>',
        unsafe_allow_html=True,
    )

    load_environment()
    Config.EXCEL_FILE_PATH = resolve_path(Config.EXCEL_FILE_PATH)
    Config.PROJECT_USAGE_PATH = resolve_path(Config.PROJECT_USAGE_PATH)
    Config.HANDBOOK_PATH = resolve_path(Config.HANDBOOK_PATH)

    is_valid, errors = Config.validate()
    if not is_valid:
        st.error('⚠️ Configuration Error')
        for error in errors:
            st.error(f'• {error}')
        return

    agent = initialize_agent()
    if not agent:
        st.error('❌ Failed to initialize. Check logs.')
        return

    excel_path = Config.EXCEL_FILE_PATH
    try:
        df = load_dataframe(excel_path, Config.EXCEL_SHEET_NAME)
    except FileNotFoundError:
        st.error(f'❌ Data file not found: {excel_path}')
        return
    except Exception as e:
        st.error(f'❌ Error loading data: {e}')
        return

    page = st.sidebar.radio(
        'Navigation',
        ['💬 Chat', '🔍 Manual Search', '📂 Data Management'],
        label_visibility='collapsed',
    )
    if page == '💬 Chat':
        render_chat_page()
    elif page == '🔍 Manual Search':
        render_manual_search_page(agent)
    else:
        render_data_management_page(df, excel_path)


if __name__ == '__main__':
    main()
