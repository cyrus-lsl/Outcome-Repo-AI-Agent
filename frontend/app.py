import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import sys
import pathlib

# Ensure project root is on sys.path so `backend` package can be imported when
# Streamlit runs the script from a different working directory.
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agent_core import MeasurementInstrumentAgent

def main():
    st.set_page_config(
        page_title="Measurement Instrument Assistant",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Measurement Instrument Assistant")
    st.markdown("**Powered by Hugging Face AI**")
    # Try to load a .env file so environment variables like HF_TOKEN are available to Streamlit
    try:
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
        else:
            # fallback to default search (cwd, parent, etc.)
            load_dotenv(override=False)
    except Exception:
        # non-fatal: continue without failing the app
        pass

    # (Removed runtime diagnostics per user request)
    
    
    # Define the Excel file path
    excel_file_path = "measurement_instruments.xlsx"
    sheet_name = "Measurement Instruments"
    
    # Check if file exists and load data
    df = None
    if os.path.exists(excel_file_path):
        try:
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
                
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.error("❌ measurement_instruments.xlsx not found")
        st.info("Please make sure the Excel file is in the same folder as this app")
        return

    # Page selector (Chat vs Manual search)
    page = st.sidebar.radio("Select page", ["Chat", "Manual search"])

    # Initialize the agent (will read the Excel file)
    agent = None
    try:
        agent = MeasurementInstrumentAgent(excel_file_path, sheet_name=sheet_name)
    except Exception:
        # If agent init fails, we still allow viewing the dataset or manual form that doesn't depend on TF-IDF
        agent = None

    if page == "Chat":
        st.divider()
        st.subheader("💬 Chat with AI Assistant")
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your AI assistant for measurement instruments. I have access to your database and can help you find the right tools for your research. What would you like to know?"}
            ]

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

        if st.button("Clear Chat History"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat cleared. How can I help you with measurement instruments?"}
            ]
            st.rerun()

    else:  # Manual search page
        st.divider()
        st.subheader("🧭 Manual search form")
        st.markdown("Select which filters you want to apply, then provide values for the selected filters.")

        # available filter options
        filter_options = ["Beneficiaries", "Measure", "Validated in HK", "Program-level metrics"]
        selected_filters = st.multiselect("Choose filters to apply", filter_options)

        # prepare inputs depending on selection
        beneficiaries_arg = None
        measure_arg = None
        validated_choice = 'both'
        prog_choice = 'both'

        if "Beneficiaries" in selected_filters:
            # try to build a choices list from the dataset
            beneficiaries_choices = []
            try:
                all_vals = df['Target Group(s)'].dropna().astype(str).tolist()
                # split comma-separated entries and dedupe
                seen = set()
                for v in all_vals:
                    for part in [p.strip() for p in v.split(',') if p.strip()]:
                        if part.lower() not in seen:
                            beneficiaries_choices.append(part)
                            seen.add(part.lower())
            except Exception:
                beneficiaries_choices = []

            if beneficiaries_choices:
                beneficiaries_arg = st.multiselect("Select beneficiary groups (you can pick multiple)", beneficiaries_choices)
            else:
                beneficiaries_text = st.text_input("Who are your target beneficiaries? (comma-separated)")
                beneficiaries_arg = [b.strip() for b in beneficiaries_text.split(',')] if beneficiaries_text and beneficiaries_text.strip() else None

        if "Measure" in selected_filters:
            measure_arg = st.text_input("What are you trying to measure?")

        if "Validated in HK" in selected_filters:
            validated_choice = st.selectbox("Validated in HK", ["both", "yes", "no"], index=0)

        if "Program-level metrics" in selected_filters:
            prog_choice = st.selectbox("Program-level metrics", ["both", "yes", "no"], index=0)

        if st.button("Search"):
            # normalize beneficiaries_arg
            if isinstance(beneficiaries_arg, str):
                beneficiaries_arg = [b.strip() for b in beneficiaries_arg.split(',') if b.strip()]
            if isinstance(beneficiaries_arg, list) and len(beneficiaries_arg) == 0:
                beneficiaries_arg = None

            if measure_arg and measure_arg.strip() == '':
                measure_arg = None

            if agent is None:
                st.error("Agent could not be initialized (check the Excel file). Manual filtering will still run on the loaded DataFrame.")

            results = None
            try:
                if agent is not None:
                    results = agent.manual_search(beneficiaries=beneficiaries_arg, measure=measure_arg, validated=validated_choice, prog_level=prog_choice)
                else:
                    # fallback: simple pandas filtering
                    df_local = df.copy()
                    if beneficiaries_arg:
                        mask = False
                        for b in beneficiaries_arg:
                            mask = mask | df_local['Target Group(s)'].astype(str).str.contains(str(b), case=False, na=False)
                        df_local = df_local[mask]
                    if measure_arg:
                        mask = df_local['Outcome Domain'].astype(str).str.contains(measure_arg, case=False, na=False) | df_local['Outcome Keywords'].astype(str).str.contains(measure_arg, case=False, na=False) | df_local['Purpose'].astype(str).str.contains(measure_arg, case=False, na=False)
                        df_local = df_local[mask]
                    formatted = {'query': measure_arg or (', '.join(beneficiaries_arg) if beneficiaries_arg else ''), 'recommendations': []}
                    for _, row in df_local.head(10).iterrows():
                        formatted['recommendations'].append({
                            'name': row.get('Measurement Instrument', ''),
                            'acronym': row.get('Acronym', ''),
                            'purpose': row.get('Purpose', ''),
                            'target_group': row.get('Target Group(s)', ''),
                            'domain': row.get('Outcome Domain', ''),
                            'similarity_score': None,
                        })
                    results = formatted
            except Exception as e:
                st.error(f"Error running manual search: {e}")

            if results:
                # display nicely
                if agent is not None:
                    st.markdown(agent.format_response(results))
                else:
                    st.write(results)

def call_huggingface_chat(prompt, df):
    """Call Hugging Face chat model for intelligent responses"""
    
    try:
        # Initialize Hugging Face client
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
        )
        
        # Prepare context from the dataset
        context = build_dataset_context(df)
        
        system_message = """You are an expert research assistant specializing in measurement instruments and psychological assessment tools. 
        Provide detailed, practical advice about selecting and using measurement instruments for research and clinical purposes."""
        
        user_message = f"""DATABASE OF MEASUREMENT INSTRUMENTS:
{context}

USER QUESTION: {prompt}

Please provide a comprehensive, helpful response about measurement instruments. Be specific and practical in your recommendations."""

        completion = client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error calling Hugging Face AI: {str(e)}"

def build_dataset_context(df):
    """Build context string from the dataset"""
    
    context_parts = []
    context_parts.append(f"Total instruments in database: {len(df)}")
    
    # Add domains information
    if 'Outcome Domain' in df.columns:
        domains = df['Outcome Domain'].dropna().unique()[:10]
        context_parts.append(f"Available domains: {', '.join(domains)}")
    
    # Add sample of instruments
    context_parts.append("\nSample instruments:")
    for i, row in df.head(8).iterrows():
        instrument_info = f"{i+1}. {row.get('Measurement Instrument', 'Unknown')}"
        
        if 'Acronym' in df.columns and pd.notna(row.get('Acronym')):
            instrument_info += f" ({row['Acronym']})"
        
        if 'Outcome Domain' in df.columns and pd.notna(row.get('Outcome Domain')):
            instrument_info += f" - {row['Outcome Domain']}"
        
        if 'Purpose' in df.columns and pd.notna(row.get('Purpose')):
            purpose = str(row['Purpose'])
            if len(purpose) > 100:
                purpose = purpose[:100] + "..."
            instrument_info += f" - {purpose}"
        
        context_parts.append(instrument_info)
    
    return "\n".join(context_parts)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Print traceback to stdout (visible in the Streamlit server logs)
        import traceback
        print("Error starting Streamlit app:", e)
        traceback.print_exc()
        # Try to show the error inside Streamlit UI if possible
        try:
            import streamlit as st
            st.set_page_config(page_title="Error")
            st.error("Oh no. Error running app — see details below.")
            st.exception(e)
            st.text(traceback.format_exc())
        except Exception:
            # If Streamlit UI can't be used, just exit after printing
            pass