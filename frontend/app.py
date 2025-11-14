import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

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

    # Lightweight non-sensitive diagnostics to help debug "Oh no. Error running app"
    try:
        env_diag = {
            "cwd": os.getcwd(),
            "env_path_found": bool(env_path) if 'env_path' in locals() else False,
            "hf_token_present": bool(os.environ.get('HF_TOKEN')),
        }
        with st.expander("⚠️ Runtime diagnostics (no secrets shown)"):
            st.write(env_diag)
    except Exception:
        # keep diagnostics best-effort and non-fatal
        pass
    
    
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

    # Chat interface
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
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response using Hugging Face
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = call_huggingface_chat(prompt, df)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared. How can I help you with measurement instruments?"}
        ]
        st.rerun()

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