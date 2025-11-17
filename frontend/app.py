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
    """Call Hugging Face chat model for responses"""
    try:
        # Read HF token robustly (allow quoted tokens in .env)
        hf = os.environ.get("HF_TOKEN")
        if isinstance(hf, str):
            hf = hf.strip().strip('\"\'')
        if not hf:
            raise RuntimeError('HF_TOKEN not set in environment')

        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf,
        )
        context = build_dataset_context(df)
        # Provide full instrument list to the model and require selection only
        # from that list to avoid invented names.
        instrument_names = [str(n) for n in df['Measurement Instrument'].dropna().unique().tolist()]

        system_message = (
            "You are an assistant that selects only instrument NAMES from the provided DATABASE. "
            "You will be given the full list of available instrument names. Do NOT invent any names. "
            "Return ONLY a valid JSON array (e.g. [\"Name A\", \"Name B\"]) containing up to 6 names that appear in the provided list."
        )

        user_message = (
            f"DATABASE OF MEASUREMENT INSTRUMENTS:\nAvailable instruments:\n{chr(10).join(['- ' + n for n in instrument_names])}\n\n"
            f"USER QUERY: {prompt}\n\n"
            "Return a JSON array of up to 6 instrument names from the provided list that best match the user query."
        )

        completion = client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=400,
            temperature=0.0
        )

        llm_text = completion.choices[0].message.content

        import json
        names = []
        try:
            parsed = json.loads(llm_text)
            if isinstance(parsed, list):
                names = [str(x).strip() for x in parsed if x]
        except Exception:
            names = [line.strip(' -') for line in llm_text.split('\n') if line.strip()]

        # Validate that returned names actually exist in the spreadsheet (case-insensitive)
        matched = []
        unknown = []
        available_set = {n.lower() for n in instrument_names}
        import re
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
                # Build a structured dict for the instrument
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

        # Build a readable markdown summary as well as structured results
        out_parts = []
        for ins in matched:
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

        md = "\n\n".join(out_parts)
        return {'text': md, 'matched': matched, 'unknown': unknown}
        
    except Exception as e:
        return f"❌ Error calling Hugging Face AI: {str(e)}"


def render_chat_page(agent, df):
    """Render the chat interface"""
    st.subheader("💬 Chat with AI Assistant")
    # Brief capabilities summary shown on the chat page per user request.
    st.markdown("### What I can do")
    st.markdown(
        "- Search the instrument database using natural-language queries (e.g. 'mental health of elderly')\n"
        "- Return matched instruments with purpose, target group and domain\n"
        "- Manual search with filters on the Manual Search page"
    )
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant", 
            "content": "Hello! I'm your AI assistant for measurement instruments. How can I help you find the right tools for your research?"
        }]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about measurement instruments..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = call_huggingface_chat(prompt, df)
                # If structured response returned, render expanders; otherwise print text
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
                                # collect names for session display
                                display_text += f"{ins.get('name','')}\n"
                else:
                    st.markdown(response)
                    display_text = response if isinstance(response, str) else str(response)

        st.session_state.messages.append({"role": "assistant", "content": display_text})

    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = [{
            "role": "assistant", 
            "content": "Chat cleared. How can I help you with measurement instruments?"
        }]
        st.rerun()


def render_manual_search_page(agent, df):
    """Render the manual search interface"""
    st.subheader("🧭 Manual Search")
    
    # Filter selection
    selected_filters = st.multiselect(
        "Choose filters to apply", 
        ["Beneficiaries", "Measure", "Validated in HK", "Program-level metrics"]
    )

    # Filter inputs
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

    # Search button
    if st.button("Search"):
        if not agent:
            st.error("Agent not available. Using basic filtering.")
            return

            try:
                results = agent.manual_search(
                    beneficiaries=beneficiaries,
                    measure=measure, 
                    validated=validated,
                    prog_level=prog_level
                )
                # results is a dict with 'recommendations'
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