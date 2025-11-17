import streamlit as st
import pandas as pd
import os
import sys
import pathlib
import re
from openai import OpenAI

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    

def load_environment():
    """Load environment variables"""
    try:
        from dotenv import load_dotenv, find_dotenv
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
    except ImportError:
        pass


def initialize_agent():
    """Initialize the measurement instrument agent"""
    excel_path = "measurement_instruments.xlsx"
    sheet_name = "Measurement Instruments"
    
    from backend.agent_core import MeasurementInstrumentAgent
    return MeasurementInstrumentAgent(excel_path, sheet_name=sheet_name)


def build_dataset_context(df):
    """Build context string from the dataset"""
    context_parts = [f"Total instruments in database: {len(df)}"]
    
    if 'Outcome Domain' in df.columns:
        domains = df['Outcome Domain'].dropna().unique()[:10]
        context_parts.append(f"Available domains: {', '.join(domains)}")
    
    context_parts.append("\nSample instruments:")
    for i, row in df.head(8).iterrows():
        info = f"{i+1}. {row.get('Measurement Instrument', 'Unknown')}"
        
        if row.get('Acronym'):
            info += f" ({row['Acronym']})"
        if row.get('Outcome Domain'):
            info += f" - {row['Outcome Domain']}"
        if row.get('Purpose'):
            purpose = str(row['Purpose'])[:100] + "..." if len(str(row['Purpose'])) > 100 else str(row['Purpose'])
            info += f" - {purpose}"
        
        context_parts.append(info)
    
    return "\n".join(context_parts)


def call_huggingface_chat(prompt, df):
    """Call Hugging Face chat model for responses.

    The function sends a strict prompt that lists instruments with short
    metadata. The LLM must return a JSON array of instrument names from that
    list. After parsing we do a light post-filter to ensure returned
    instruments contain query tokens in their name/domain/purpose/target.
    """
    # Read HF token robustly (allow quoted tokens in .env)
    hf = os.environ.get("HF_TOKEN")
    if isinstance(hf, str):
        hf = hf.strip().strip('"\'')
    if not hf:
        return "❌ HF_TOKEN not set in environment"

    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf)

    # Build instrument lines and name list
    inst_lines = []
    instrument_names = []
    for _, row in df.iterrows():
        name = str(row.get('Measurement Instrument', '')).strip()
        if not name:
            continue
        domain = str(row.get('Outcome Domain', '')).strip()
        target = str(row.get('Target Group(s)', '')).strip()
        prog_flag = str(row.get('Programme-level metric?', '')).strip()
        inst_lines.append(f"{name} | {domain} | {target} | Programme-level: {prog_flag}")
        instrument_names.append(name)

    system_message = (
        "You are an assistant that selects only instrument NAMES from the provided DATABASE. "
        "You will be given a list of instruments with short metadata (Domain, Target groups, Programme-level flag). Do NOT invent any names. "
        "Return ONLY a valid JSON array (e.g. [\"Name A\", \"Name B\"]) containing up to 6 names that appear in the provided list."
    )

    user_message = (
        f"DATABASE OF MEASUREMENT INSTRUMENTS:\nEach line: Name | Domain | Target(s) | Programme-level\n{chr(10).join(inst_lines)}\n\n"
        f"USER QUERY: {prompt}\n\n"
        + "Choose up to 6 instruments from the provided list that best match the user query, considering Domain, Purpose and Target groups. Return a JSON array with instrument names only."
    )

    try:
        completion = client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905",
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
            max_tokens=400,
            temperature=0.0,
        )
    except Exception as e:
        return f"❌ Error calling Hugging Face AI: {e}"

    llm_text = completion.choices[0].message.content

    # Parse JSON array or fallback to line-splitting
    import json
    names = []
    try:
        parsed = json.loads(llm_text)
        if isinstance(parsed, list):
            names = [str(x).strip() for x in parsed if x]
    except Exception:
        names = [line.strip(' -') for line in llm_text.split('\n') if line.strip()]

    # Validate that returned names actually exist
    matched = []
    unknown = []
    available_set = {n.lower() for n in instrument_names}
    for nm in names:
        if nm.strip().lower() not in available_set:
            unknown.append(nm)
            continue
        try:
            match = df[df['Measurement Instrument'].str.contains(nm, case=False, na=False, regex=False)]
        except TypeError:
            match = df[df['Measurement Instrument'].str.contains(re.escape(nm), case=False, na=False)]
        if not match.empty:
            r = match.iloc[0]
            matched.append({
                'name': r.get('Measurement Instrument', ''),
                'acronym': r.get('Acronym', ''),
                'purpose': r.get('Purpose', ''),
                'target': r.get('Target Group(s)', ''),
                'domain': r.get('Outcome Domain', ''),
                'no_of_items': r.get('No. of Questions / Statements', ''),
                'sample_q1': r.get('Sample Question / Statement - 1', ''),
                'sample_q2': r.get('Sample Question / Statement - 2', ''),
                'sample_q3': r.get('Sample Question / Statement - 3', ''),
                'scale': r.get('Scale', ''),
                'scoring': r.get('Scoring', ''),
                'validated': r.get('Validated in Hong Kong', ''),
                'programme_level': r.get('Programme-level metric?', ''),
                'download_eng': r.get('Download (Eng)', ''),
                'download_chi': r.get('Download (Chi)', ''),
                'citation': r.get('Citation', ''),
            })

    if unknown and os.getenv('DEBUG_HF') == '1':
        print('[DEBUG_HF] Model returned names not in spreadsheet and were ignored:', unknown)

    if not matched:
        return {'text': 'No matching instruments found in the database.', 'matched': [], 'unknown': unknown}

    # Post-filter by simple token overlap to remove clearly irrelevant picks
    def _lower(x):
        try:
            return str(x).lower()
        except Exception:
            return ''

    qtokens = [t for t in re.findall(r"\w+", _lower(prompt)) if len(t) > 2]
    filtered = []
    for ins in matched:
        hay = ' '.join([_lower(ins.get('name', '')), _lower(ins.get('domain', '')), _lower(ins.get('purpose', '')), _lower(ins.get('target', ''))])
        if any(tok in hay for tok in qtokens):
            filtered.append(ins)

    if not filtered:
        filtered = matched
        filter_note = ' (note: heuristics did not find stricter matches; showing best matches)'
    else:
        filter_note = ''

    # Build markdown summary
    out_parts = []
    for ins in filtered:
        part = f"**{ins['name']}" + (f" ({ins['acronym']})" if ins['acronym'] else '') + f"** — {ins['domain']}\n\n"
        part += f"**Purpose:** {ins['purpose']}  \n"
        part += f"**Target:** {ins['target']}  \n"
        if ins['no_of_items']:
            part += f"**Items:** {ins['no_of_items']}  \n"
        if ins['scale']:
            part += f"**Scale:** {ins['scale']}  \n"
        if ins['validated']:
            part += f"**Validated in HK:** {ins['validated']}  \n"
        if ins['programme_level']:
            part += f"**Programme-level metric?:** {ins.get('programme_level')}  \n"
        if ins['download_eng']:
            part += f"[Download (Eng)]({ins['download_eng']})  \n"
        if ins['citation']:
            part += f"**Citation:** {ins['citation']}  \n"
        out_parts.append(part)

    md = "\n\n".join(out_parts) + filter_note
    return {'text': md, 'matched': filtered, 'unknown': unknown}


def render_chat_page(agent, df):
    """Render the chat interface"""
    st.subheader("💬 Chat with AI Assistant")
    st.markdown("### What I can do")
    st.markdown(
        "- Search the instrument database using natural-language queries (e.g. 'mental health of elderly')\n"
        "- Return matched instruments with purpose, target group and domain\n"
        "- Manual search with filters on the Manual Search page"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about measurement instruments..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = call_huggingface_chat(prompt, df)
                if isinstance(response, dict) and 'matched' in response:
                    if not response['matched']:
                        st.markdown(response.get('text', 'No matching instruments found in the database.'))
                        display_text = response.get('text', '')
                    else:
                        display_text = ''
                        for ins in response['matched']:
                            with st.expander(ins.get('name', '') + (f" ({ins.get('acronym')})" if ins.get('acronym') else '')):
                                st.markdown(f"**Domain:** {ins.get('domain','')}  ")
                                st.markdown(f"**Purpose:** {ins.get('purpose','')}  ")
                                st.markdown(f"**Target:** {ins.get('target','')}  ")
                                if ins.get('no_of_items'):
                                    st.markdown(f"**Items:** {ins.get('no_of_items')}  ")
                                if ins.get('scale'):
                                    st.markdown(f"**Scale:** {ins.get('scale')}  ")
                                if ins.get('scoring'):
                                    st.markdown(f"**Scoring:** {ins.get('scoring')}  ")
                                if ins.get('validated'):
                                    st.markdown(f"**Validated in HK:** {ins.get('validated')}  ")
                                if ins.get('programme_level'):
                                    st.markdown(f"**Programme-level metric?:** {ins.get('programme_level')}  ")
                                if ins.get('download_eng'):
                                    st.markdown(f"[Download (Eng)]({ins.get('download_eng')})")
                                if ins.get('sample_q1'):
                                    st.markdown(f"**Sample item:** {ins.get('sample_q1')}  ")
                                display_text += f"{ins.get('name','')}\n"
                else:
                    st.markdown(response)
                    display_text = response if isinstance(response, str) else str(response)

        st.session_state.messages.append({"role": "assistant", "content": display_text})

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()


def render_manual_search_page(agent, df):
    """Render the manual search interface"""
    st.subheader("🧭 Manual Search")

    selected_filters = st.multiselect(
        "Choose filters to apply",
        ["Beneficiaries", "Measure", "Validated in HK", "Program-level metrics"]
    )

    beneficiaries = None
    measure = None
    validated = "both"
    prog_level = "both"

    if "Beneficiaries" in selected_filters:
        beneficiaries_input = st.text_input(
            "Target beneficiaries (comma-separated)",
            placeholder="e.g. youth, elderly, teachers"
        )
        beneficiaries = [b.strip() for b in beneficiaries_input.split(',')] if beneficiaries_input.strip() else None

    if "Measure" in selected_filters:
        measure = st.text_input("What are you trying to measure?")

    if "Validated in HK" in selected_filters:
        validated = st.selectbox("Validated in HK", ["both", "yes", "no"])

    if "Program-level metrics" in selected_filters:
        prog_level = st.selectbox("Program-level metrics", ["both", "yes", "no"])

    if st.button("Search"):
        try:
            results = agent.manual_search(
                beneficiaries=beneficiaries,
                measure=measure,
                validated=validated,
                prog_level=prog_level
            )
            recs = results.get('recommendations', []) if isinstance(results, dict) else []
            if not recs:
                st.markdown('No matching instruments found.')
            else:
                for ins in recs:
                    with st.expander(ins.get('name', '') + (f" ({ins.get('acronym')})" if ins.get('acronym') else '')):
                        st.markdown(f"**Domain:** {ins.get('domain','')}  ")
                        st.markdown(f"**Purpose:** {ins.get('purpose','')}  ")
                        st.markdown(f"**Target:** {ins.get('target_group','')}  ")
                        if ins.get('programme_level'):
                            st.markdown(f"**Programme-level metric?:** {ins.get('programme_level')}  ")
        except Exception as e:
            st.error(f"Search error: {e}")


def main():
    # Page configuration
    st.set_page_config(
        page_title="Measurement Instrument Assistant",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Measurement Instrument Assistant")
    st.markdown("**Powered by Hugging Face AI**")
    
    # Load environment and initialize agent
    load_environment()
    agent = initialize_agent()
    
    if not agent:
        return
    
    # Load data for context
    try:
        df = pd.read_excel("measurement_instruments.xlsx", sheet_name="Measurement Instruments")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Page navigation
    page = st.sidebar.radio("Navigation", ["Chat", "Manual Search"])
    
    if page == "Chat":
        render_chat_page(agent, df)
    else:
        render_manual_search_page(agent, df)


if __name__ == "__main__":
    main()