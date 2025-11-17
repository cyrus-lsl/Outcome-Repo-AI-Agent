import streamlit as st
import pandas as pd
import os
import sys
import pathlib
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
                matched.append(match.iloc[0])

        if unknown and os.getenv('DEBUG_HF') == '1':
            print('[DEBUG_HF] Model returned names not in spreadsheet and were ignored:', unknown)

        if not matched:
            return "No matching instruments found in the database."

        out_parts = []
        for row in matched:
            name = row.get('Measurement Instrument', '')
            acronym = row.get('Acronym', '')
            purpose = row.get('Purpose', '')
            target = row.get('Target Group(s)', '')
            domain = row.get('Outcome Domain', '')

            part = f"- {name}"
            if acronym:
                part += f" ({acronym})"
            if domain:
                part += f" — {domain}"
            part += f"\n  Purpose: {purpose}\n  Target: {target}\n"
            out_parts.append(part)

        return "\n\n".join(out_parts)
        
    except Exception as e:
        return f"❌ Error calling Hugging Face AI: {str(e)}"


def render_chat_page(agent, df):
    """Render the chat interface"""
    st.subheader("💬 Chat with AI Assistant")
    
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
                st.markdown(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response})

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
            st.markdown(agent.format_response(results))
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