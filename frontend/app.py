import streamlit as st
import pandas as pd
import os
import sys
import json
import pathlib
import re
import logging
from functools import lru_cache
from openai import OpenAI
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import configuration
try:
    from config import Config, logger
except ImportError:
    # Fallback if config.py doesn't exist
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    Config = None


def load_environment():
    """Load environment variables"""
    try:
        from dotenv import load_dotenv, find_dotenv
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
    except ImportError:
        pass


def resolve_excel_path(excel_path: str) -> str:
    """Resolve Excel file path relative to project root"""
    # Allow direct HTTP/HTTPS URLs for shared Excel files
    if isinstance(excel_path, str) and excel_path.lower().startswith(("http://", "https://")):
        return excel_path
    if os.path.isabs(excel_path):
        # Already an absolute path
        return excel_path
    # Resolve relative to project root
    resolved = ROOT / excel_path
    return str(resolved.resolve())


@st.cache_resource
def initialize_agent():
    """Initialize the measurement instrument agent (cached)"""
    try:
        if Config:
            excel_path = Config.EXCEL_FILE_PATH
            sheet_name = Config.EXCEL_SHEET_NAME
        else:
            excel_path = "measurement_instruments.xlsx"
            sheet_name = "Measurement Instruments"
        
        # Resolve path relative to project root
        excel_path = resolve_excel_path(excel_path)
        
        from backend.agent_core import MeasurementInstrumentAgent
        agent = MeasurementInstrumentAgent(excel_path, sheet_name=sheet_name)
        return agent
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_dataframe(excel_path: str, sheet_name: str):
    """Load and cache the Excel dataframe"""
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        logger.error(f"Failed to load dataframe: {e}", exc_info=True)
        raise


def save_uploaded_excel(uploaded_file, dest_path: str):
    """Persist an uploaded Excel file to the destination path."""
    try:
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True, None
    except Exception as e:
        logger.error(f"Failed to save uploaded Excel to {dest_path}: {e}", exc_info=True)
        return False, str(e)


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


def sanitize_input(text: str, max_length: int = 500) -> str:
    """Sanitize user input to prevent injection attacks"""
    if not isinstance(text, str):
        return ""
    # Remove potentially dangerous characters but keep normal punctuation
    text = text.strip()[:max_length]
    # Remove null bytes and control characters except newlines and tabs
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    return text


def get_llm_config_with_fallback():
    """Get LLM configuration with automatic fallback from Ollama to configured cloud LLM"""
    import requests
    
    # Primary: try local Ollama (using smaller 3B model for better performance on M1)
    ollama_base = "http://localhost:11434/v1"
    ollama_model = "llama3.2:3b"  # Faster than llama3 (8B) on M1 Mac
    
    # Fallback: use configured LLM from environment
    fallback_base = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    fallback_model = os.environ.get("LLM_MODEL", "llama3")
    fallback_token = os.environ.get("LLM_API_KEY") or os.environ.get("HF_TOKEN") or "dummy-token"
    
    # Check if Ollama is reachable
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return ollama_base, ollama_model, "dummy-token"
    except Exception:
        pass
    
    # Fall back to configured LLM
    if fallback_base != ollama_base or fallback_model != ollama_model:
        return fallback_base, fallback_model, fallback_token
    else:
        return ollama_base, ollama_model, "dummy-token"


def call_llm_chat(prompt, df, validated_only=False, prog_only=False, max_results=6):
    """Call the configured LLM for responses.

    Uses LLM to search and match instruments from the database.
    """
    prompt = sanitize_input(prompt, max_length=Config.MAX_QUERY_LENGTH if Config else 500)
    if not prompt:
        return {'text': "Please provide a valid search query.", 'matched': [], 'unknown': []}

    prompt_l = str(prompt or '').lower().strip()
    if not prompt_l:
        return {'text': '', 'matched': [], 'unknown': []}

    try:
        base_url, model_name, api_key = get_llm_config_with_fallback()
        client = OpenAI(base_url=base_url, api_key=api_key)
        
        # Detect meta questions
        classification_prompt = f'Is the following user query asking about what the AI/system can do, or is it searching for a measurement instrument?\n\nUser query: "{prompt}"\n\nRespond with ONLY one word: "META" if asking about system capabilities, or "SEARCH" if searching for instruments.'
        try:
            classification = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": classification_prompt}],
                max_tokens=10, temperature=0.0
            )
            is_meta_question = "META" in classification.choices[0].message.content.strip().upper()
        except Exception:
            is_meta_question = False
    except Exception as e:
        logger.error(f"Error initializing: {e}", exc_info=True)
        return {'text': "An error occurred. Please try again.", 'matched': [], 'unknown': []}
    
    if is_meta_question:
        try:
            _, current_model, _ = get_llm_config_with_fallback()
            model_info = f"using **{current_model}**"
        except:
            model_info = ""
        return {
            'text': f"""I'm an AI assistant designed to help you search for measurement instruments from your research database {model_info}.

**What I can do:**
- Search for measurement instruments by keywords, domains, or target groups
- Filter by validation status (HK-validated)
- Filter by programme-level metrics
- Provide detailed information about each instrument

**To search for instruments, try asking:**
- "mental health assessment for elderly"
- "physical activity questionnaire for youth"
- "quality of life scale validated in Hong Kong"

What measurement instrument would you like to find?""",
            'matched': [],
            'unknown': []
        }
    
    want_validated_hk = validated_only or bool(re.search(r"validated.*hong|validated.*hk|validated in hong|validate.*hong|validate.*hk", prompt_l))

    # Parse item count constraints
    max_items = min_items = None
    try:
        max_patterns = [
            r'(?:not\s+more\s+than|maximum|max|at\s+most|less\s+than|fewer\s+than|under|below)\s+(\d+)\s*(?:items?|questions?|statements?)?',
            r'(\d+)\s*(?:or\s+)?(?:fewer|less)\s*(?:items?|questions?|statements?)?',
        ]
        for pattern in max_patterns:
            if match := re.search(pattern, prompt_l):
                max_items = int(match.group(1))
                break
        
        min_patterns = [
            r'(?:at\s+least|minimum|min|more\s+than|greater\s+than|over|above)\s+(\d+)\s*(?:items?|questions?|statements?)?',
            r'(\d+)\s*(?:or\s+)?(?:more|greater)\s*(?:items?|questions?|statements?)?',
        ]
        for pattern in min_patterns:
            if match := re.search(pattern, prompt_l):
                min_items = int(match.group(1))
                break
        
        if not max_items and not min_items:
            if match := re.search(r'(?:exactly|precisely|exactly\s+)?(\d+)\s*(?:items?|questions?|statements?)(?:\s+only)?', prompt_l):
                max_items = min_items = int(match.group(1))
        
    except Exception as e:
        pass

    def _validated_in_hk_text(x):
        s = str(x or '').lower().strip()
        if not s or s in ('-', 'na', 'n/a') or 'not validated' in s or 'not in hong' in s or re.search(r'not .*hong|\bno\b', s):
            return False
        return s.startswith('yes') or ('validated' in s and 'hong' in s) or (('hong' in s or 'hk' in s) and ('valid' in s or 'develop' in s or 'refer' in s))

    # Send entire dataset to LLM (no pre-filtering)
    inst_lines = []
    instrument_names = []
    for _, row in df.iterrows():
        name = str(row.get('Measurement Instrument', '')).strip()
        if name:
            domain = str(row.get('Outcome Domain', '')).strip()[:50]
            target = str(row.get('Target Group(s)', '')).strip()[:50]
            purpose = str(row.get('Purpose', '')).strip()[:100]  # Add Purpose - crucial for matching!
            items = str(row.get('No. of Questions / Statements', '')).strip() or 'N/A'
            # Format: Name|Domain|Target|Purpose|Items
            inst_lines.append(f"{name}|{domain}|{target}|{purpose}|{items}")
            instrument_names.append(name)

    item_constraint_note = ""
    if max_items == min_items and max_items is not None:
        item_constraint_note = f"IMPORTANT: User requested exactly {max_items} items.\n"
    else:
        if max_items is not None:
            item_constraint_note += f"IMPORTANT: User requested at most {max_items} items.\n"
        if min_items is not None:
            item_constraint_note += f"IMPORTANT: User requested at least {min_items} items.\n"
    
    system_message = f"CRITICAL: You MUST ONLY select instruments from the database provided below. Do NOT use your own knowledge or invent instrument names. The database contains {len(inst_lines)} instruments. Each line format is: Name|Domain|Target|Purpose|Items. Analyze the Purpose field to understand what each instrument measures. Return JSON array of objects with 'name' (must match EXACTLY from database) and 'confidence' (0-100). Return up to {max_results} instruments, or [] if no matches. IMPORTANT: Only return instrument names that appear EXACTLY in the database list below."
    user_message = (
        f"=== DATABASE OF {len(inst_lines)} INSTRUMENTS (YOU MUST ONLY USE THESE) ===\n"
        f"{chr(10).join(inst_lines)}\n"
        f"=== END OF DATABASE ===\n\n"
        f"User Query: {prompt}\n"
        + ("[Filter: HK-validated only]\n" if want_validated_hk else "")
        + ("[Filter: Programme-level only]\n" if prog_only else "")
        + item_constraint_note
        + f"\nCRITICAL INSTRUCTIONS:\n"
        + f"1. ONLY return instrument names that appear EXACTLY in the database above\n"
        + f"2. Do NOT invent, modify, or create new instrument names\n"
        + f"3. Copy the instrument name EXACTLY as it appears in the database\n"
        + f"4. Return JSON array: [{{\"name\": \"Exact Name From Database\", \"confidence\": 85}}, ...]\n"
        + f"5. Maximum {max_results} instruments\n"
        + f"6. If no good matches, return empty array []"
    )

    try:
        completion = client.chat.completions.create(
            model=model_name, messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
            max_tokens=300, temperature=0.0  # Increased for confidence scores
        )
        llm_text = completion.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        logger.error(f"LLM API call failed: {error_type}: {error_msg}", exc_info=True)
        
        # Provide more helpful error message based on error type
        if "APIConnectionError" in error_type or "ConnectError" in error_type or "connection" in error_msg.lower() or "cannot assign requested address" in error_msg.lower():
            return {'text': "Connection error: Unable to reach the LLM service. Please check:\n- Your LLM service is running and accessible\n- Network connectivity\n- LLM_BASE_URL configuration is correct\n\nIf using Ollama locally, make sure it's running: `ollama serve`", 'matched': [], 'unknown': []}
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower() or "Timeout" in error_type:
            return {'text': "The request took too long. The dataset might be too large. Please try a more specific query.", 'matched': [], 'unknown': []}
        elif "rate limit" in error_msg.lower() or "429" in error_msg or "RateLimitError" in error_type:
            return {'text': "Rate limit exceeded. Please wait a moment and try again.", 'matched': [], 'unknown': []}
        elif "401" in error_msg or "unauthorized" in error_msg.lower() or "AuthenticationError" in error_type:
            return {'text': "Authentication error. Please check your LLM API key configuration.", 'matched': [], 'unknown': []}
        elif "APIError" in error_type:
            return {'text': f"API error: {error_msg[:200]}. Please check your LLM configuration and try again.", 'matched': [], 'unknown': []}
        else:
            return {'text': f"I encountered an error ({error_type}): {error_msg[:200]}. Please try again or check the logs for details.", 'matched': [], 'unknown': []}

    cleaned_text = llm_text.strip()
    
    if '```' in cleaned_text:
        first_backticks = cleaned_text.find('```')
        if first_backticks >= 0:
            after_open = cleaned_text.find('\n', first_backticks)
            if after_open > 0:
                cleaned_text = cleaned_text[after_open+1:]
            else:
                cleaned_text = cleaned_text[first_backticks+3:]
        
        last_backticks = cleaned_text.rfind('```')
        if last_backticks > 0:
            cleaned_text = cleaned_text[:last_backticks]
    
    cleaned_text = cleaned_text.strip()
    
    confidence_scores = {}
    
    try:
        parsed = json.loads(cleaned_text)
        names = []
        
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    name = str(item.get('name', '')).strip()
                    confidence = item.get('confidence', None)
                    if name:
                        names.append(name)
                        if confidence is not None:
                            confidence_scores[name.lower()] = float(confidence)
                elif isinstance(item, str):
                    name = item.strip()
                    if name:
                        names.append(name)
        
    except json.JSONDecodeError as e:
        names = []
        confidence_scores = {}
        for line in cleaned_text.split('\n'):
            line = line.strip()
            if not line or line in ['[', ']', ',', '```json', '```']:
                continue
            line = re.sub(r'^[\[\],"\']+', '', line)
            line = re.sub(r'[\[\],"\']+$', '', line)
            line = line.strip().strip('"\'')
            if line and line not in ['[', ']', ',']:
                names.append(line)
    except Exception as e:
        logger.error(f"Unexpected error parsing LLM response: {e}")
        names = []
        confidence_scores = {} 
    
    cleaned_names = []
    cleaned_confidence = {}
    for name in names:
        if '|' in name:
            cleaned_name = name.split('|')[0].strip()
            cleaned_names.append(cleaned_name)
            if cleaned_name.lower() in confidence_scores:
                cleaned_confidence[cleaned_name.lower()] = confidence_scores[cleaned_name.lower()]
        else:
            cleaned_name = name.strip()
            cleaned_names.append(cleaned_name)
            if cleaned_name.lower() in confidence_scores:
                cleaned_confidence[cleaned_name.lower()] = confidence_scores[cleaned_name.lower()]
    
    names = cleaned_names
    confidence_scores = cleaned_confidence

    matched = []
    unknown = []
    available_set = {n.lower() for n in instrument_names}
    matched_names = set()
    
    for nm in names:
        if nm.strip().lower() not in available_set:
            unknown.append(nm)
            continue
        try:
            match = df[df['Measurement Instrument'].str.contains(nm, case=False, na=False, regex=False)]
        except TypeError:
            match = df[df['Measurement Instrument'].str.contains(re.escape(nm), case=False, na=False)]
        if not match.empty and len(matched) < max_results:
            r = match.iloc[0]
            inst_name = r.get('Measurement Instrument', '')
            if inst_name.lower() in matched_names:
                continue
            confidence = confidence_scores.get(inst_name.lower(), None)
            matched.append({
                'name': inst_name,
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
                'confidence_score': confidence
            })
            matched_names.add(inst_name.lower())

    if unknown:
        pass

    
    if not matched:
        if names:
            return {'text': f'LLM found {len(names)} potential matches, but none matched the database. This might indicate a data mismatch. Unknown names: {", ".join(unknown[:5])}.', 'matched': [], 'unknown': unknown}
        return {'text': 'No matching instruments found. Please try a more specific query.', 'matched': [], 'unknown': unknown}

    filtered = matched
    
    if want_validated_hk:
        filtered = [ins for ins in filtered if _validated_in_hk_text(ins.get('validated', ''))]
        if not filtered:
            return {'text': 'No instruments validated in Hong Kong found matching the query.', 'matched': [], 'unknown': unknown}

    if max_items is not None or min_items is not None:
        def parse_count(x):
            if not x or pd.isna(x):
                return None
            if match := re.search(r'(\d+)', str(x).strip()):
                return int(match.group(1))
            return None
        
        before_count = len(filtered)
        item_filtered = [ins for ins in filtered 
                        if (count := parse_count(ins.get('no_of_items', ''))) is not None
                        and (max_items is None or count <= max_items)
                        and (min_items is None or count >= min_items)]
        
        if item_filtered:
            filtered = item_filtered
        else:
            constraints = []
            if max_items is not None:
                constraints.append(f"maximum {max_items} items")
            if min_items is not None:
                constraints.append(f"minimum {min_items} items")
            return {'text': f'Found {before_count} matching instrument(s), but none have {" and ".join(constraints)}. Please try adjusting your search criteria or removing the item count constraint.', 
                   'matched': [], 'unknown': unknown}

    has_confidence = any(ins.get('confidence_score') is not None for ins in filtered)
    if has_confidence:
        filtered.sort(key=lambda x: x.get('confidence_score') or 0, reverse=True)

    out_parts = []
    for ins in filtered:
        part = f"**{ins['name']}" + (f" ({ins['acronym']})" if ins['acronym'] else '') + f"** — {ins['domain']}\n\n"
        if ins.get('confidence_score') is not None:
            confidence = ins['confidence_score']
            if confidence >= 80:
                confidence_label = "🟢 Excellent match"
            elif confidence >= 60:
                confidence_label = "🟡 Good match"
            elif confidence >= 40:
                confidence_label = "🟠 Fair match"
            else:
                confidence_label = "🔴 Low match"
            part += f"**{confidence_label}** (Score: {confidence:.0f}/100)  \n"
        part += f"**Purpose:** {ins['purpose']}  \n**Target:** {ins['target']}  \n"
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

    return {'text': "\n\n".join(out_parts), 'matched': filtered, 'unknown': unknown}


def _display_response(response, matched):
    """Helper function to display assistant response (instruments or text)"""
    if isinstance(response, dict) and 'matched' in response:
        matched = response.get('matched', [])
        if not matched:
            text = response.get('text', '')
            if text:
                st.info(text)
        else:
            pass
            
            for idx, ins in enumerate(matched, 1):
                title_parts = [f"{idx}. {ins.get('name', 'Unknown')}"]
                if ins.get('acronym'):
                    title_parts.append(f"({ins.get('acronym')})")
                
                with st.expander(" ".join(title_parts), expanded=(idx == 1)):
                    if ins.get('domain'):
                        st.markdown(f'<span class="badge badge-domain">📁 {ins.get("domain")}</span>', unsafe_allow_html=True)
                    
                    badge_html = ""
                    validated_val = ins.get('validated', '')
                    if validated_val:
                        s = str(validated_val).lower().strip()
                        is_validated = (s.startswith('yes') or 
                                      (('hong' in s or 'hk' in s) and ('valid' in s or 'develop' in s or 'refer' in s)) or
                                      ('validated' in s and 'hong' in s))
                        if is_validated and 'not' not in s and 'no' not in s:
                            badge_html += '<span class="badge badge-validated">✓ HK Validated</span>'
                    if ins.get('programme_level') and str(ins.get('programme_level', '')).strip().lower() in ['yes', 'y', 'true']:
                        badge_html += '<span class="badge badge-programme">📊 Programme-level</span>'
                    if badge_html:
                        st.markdown(badge_html, unsafe_allow_html=True)
                    
                    if ins.get('confidence_score') is not None:
                        confidence = ins['confidence_score']
                        if confidence >= 80:
                            confidence_label = "🟢 Excellent match"
                            confidence_color = "#22c55e"  # Green
                        elif confidence >= 60:
                            confidence_label = "🟡 Good match"
                            confidence_color = "#eab308"  # Yellow
                        elif confidence >= 40:
                            confidence_label = "🟠 Fair match"
                            confidence_color = "#f97316"  # Orange
                        else:
                            confidence_label = "🔴 Low match"
                            confidence_color = "#ef4444"  # Red
                        
                        st.markdown(
                            f'<div style="background-color: {confidence_color}20; padding: 0.5rem; border-radius: 0.5rem; border-left: 4px solid {confidence_color}; margin: 0.5rem 0;">'
                            f'<strong>{confidence_label}</strong> (Score: {confidence:.0f}/100)'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    st.markdown("---")
                    
                    if ins.get('purpose'):
                        st.markdown(f"**📝 Purpose:**  \n{ins.get('purpose')}")
                    
                    if ins.get('target'):
                        st.markdown(f"**👥 Target Group:** {ins.get('target')}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if ins.get('no_of_items'):
                            st.metric("Items", ins.get('no_of_items'))
                        if ins.get('scale'):
                            st.markdown(f"**Scale:** {ins.get('scale')}")
                    with col2:
                        if ins.get('scoring'):
                            st.markdown(f"**Scoring:** {ins.get('scoring')}")
                    
                    sample_questions = []
                    for q_key in ['sample_q1', 'sample_q2', 'sample_q3']:
                        if ins.get(q_key):
                            sample_questions.append(ins.get(q_key))
                    
                    if sample_questions:
                        with st.expander("📋 Sample Questions", expanded=False):
                            for q in sample_questions:
                                st.markdown(f"• {q}")
                    
                    download_links = []
                    if ins.get('download_eng'):
                        download_links.append(f"[📥 Download (English)]({ins.get('download_eng')})")
                    if ins.get('download_chi'):
                        download_links.append(f"[📥 Download (中文)]({ins.get('download_chi')})")
                    
                    if download_links:
                        st.markdown("**Downloads:** " + " | ".join(download_links))
                    
                    if ins.get('citation'):
                        with st.expander("📚 Citation", expanded=False):
                            st.markdown(ins.get('citation'))
    else:
        st.markdown(response if isinstance(response, str) else str(response))


def render_chat_page(agent, df):
    """Render the chat interface with conversation history"""
    
    st.markdown("""
        <style>
        .main-header {
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: rgba(28, 131, 225, 0.1);
            border-left: 4px solid #1c83e1;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        .result-count {
            color: #1c83e1;
            font-weight: 600;
            margin: 0.5rem 0;
        }
        .instrument-card {
            border: 1px solid rgba(250, 250, 250, 0.2);
            border-radius: 8px;
            padding: 0.5rem;
            margin: 0.5rem 0;
        }
        .badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            margin: 0.2rem 0.2rem 0.2rem 0;
        }
        .badge-domain {
            background-color: rgba(28, 131, 225, 0.2);
            color: #1c83e1;
        }
        .badge-validated {
            background-color: rgba(34, 197, 94, 0.2);
            color: #22c55e;
        }
        .badge-programme {
            background-color: rgba(168, 85, 247, 0.2);
            color: #a855f7;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Auto-scroll script
    st.markdown("""
    <script>
    function scrollToBottom() {
        const chatMessages = document.querySelectorAll('[data-testid="stChatMessage"]');
        if (chatMessages.length > 0) {
            chatMessages[chatMessages.length - 1].scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
    }
    
    function setupChat() {
        setTimeout(scrollToBottom, 500);
    }
    
    // Run on load and after updates
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupChat);
    } else {
        setupChat();
    }
    
    window.addEventListener('load', setupChat);
    
    // Watch for changes and auto-scroll
    const observer = new MutationObserver(function() {
        setTimeout(setupChat, 200);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """, unsafe_allow_html=True)
    
    st.subheader("💬 Chat with AI Assistant")
    
    st.markdown("""
    Hello! I can help you find measurement instruments. Please describe what you're looking for, for example:
    
    - 'mental health assessment for elderly'
    - 'physical activity questionnaire'
    - 'quality of life scale'
    
    What would you like to search for?
    """)
    
    if 'chat_initialized' not in st.session_state:
        st.session_state.chat_messages = []
        st.session_state.chat_initialized = True
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                if isinstance(message["content"], dict):
                    _display_response(message["content"], message.get("matched", []))
                else:
                    st.markdown(message["content"])

    if prompt := st.chat_input("Ask about measurement instruments..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching..."):
                response = call_llm_chat(
                    prompt,
                    df,
                    validated_only=False,
                    prog_only=False,
                    max_results=8,
                )

            _display_response(response, response.get('matched', []) if isinstance(response, dict) else [])
            
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": response,
                "matched": response.get('matched', []) if isinstance(response, dict) else []
            })


def render_data_management_page(df, excel_path):
    """Render the data management page"""
    st.title("📂 Data Management")
    
    st.markdown("### 📊 Current Database")
    overview_col1, overview_col2 = st.columns(2)
    with overview_col1:
        st.metric("Total Instruments", len(df))
    with overview_col2:
        if 'Outcome Domain' in df.columns:
            domains = df['Outcome Domain'].dropna().unique()
            st.metric("Domains", len(domains))
        else:
            st.metric("Domains", "N/A")
    
    with st.container():
        st.markdown("**Current Data Source:**")
        st.code(excel_path, language=None)
    
    st.divider()
    
    st.markdown("### 📤 Upload New Dataset")
    
    with st.container():
        uploaded_file = st.file_uploader(
            "Choose an Excel file (.xlsx) to replace the current dataset",
            type=["xlsx"],
            help="The uploaded file will replace the current dataset and clear the cache automatically",
            key="data_mgmt_uploader"
        )
        
        if uploaded_file is not None:
            with st.spinner("Uploading and saving..."):
                ok, err = save_uploaded_excel(uploaded_file, excel_path)
                if ok:
                    load_dataframe.clear()
                    st.success("✅ File uploaded and saved successfully!")
                    if st.button("🔄 Reload Now", type="primary", use_container_width=True):
                        st.rerun()
                else:
                    st.error(f"❌ Failed to save file: {err}")
    
    st.divider()
    
    st.markdown("### 🔄 Refresh Data")
    st.caption("If you've updated the Excel file externally, click below to reload the data.")
    
    if st.button("🔄 Refresh Data Now", type="primary", use_container_width=True, help="Clear cache and reload from file"):
        try:
            load_dataframe.clear()
            st.success("✅ Cache cleared! Reloading...")
            st.rerun()
        except Exception as e:
            logger.error(f"Failed to refresh data cache: {e}", exc_info=True)
            st.error("❌ Failed to refresh data. Please check the logs.")


def render_manual_search_page(agent, df):
    st.subheader("🧭 Manual Search")
    st.markdown("Use filters to narrow down your search for measurement instruments.")
    
    beneficiaries_input = st.text_input(
        "👥 Target beneficiaries",
        placeholder="e.g. youth, elderly, teachers (leave empty to search all)",
        help="Enter comma-separated list of target groups"
    )
    beneficiaries = [b.strip() for b in beneficiaries_input.split(',')] if beneficiaries_input.strip() else None

    measure = st.text_input(
        "📊 What are you trying to measure?",
        placeholder="e.g. mental health, physical activity, quality of life (leave empty to search all)",
        help="Describe what you want to measure"
    )

    st.markdown("**Additional Filters:**")
    mcol1, mcol2 = st.columns([1, 1])
    with mcol1:
        manual_validated_only = st.checkbox("✅ Require HK-validated only", value=False, help="Show only HK-validated instruments")
    with mcol2:
        manual_prog_only = st.checkbox("📊 Programme-level only", value=False, help="Show only programme-level metrics")

    if st.button("🔍 Search", type="primary", use_container_width=True):
        try:
            validated_param = 'yes' if manual_validated_only else 'both'
            prog_level_param = 'yes' if manual_prog_only else 'both'

            with st.spinner("Searching..."):
                results = agent.manual_search(
                    beneficiaries=beneficiaries,
                    measure=measure,
                    validated=validated_param,
                    prog_level=prog_level_param
                )
            
            recs = results.get('recommendations', []) if isinstance(results, dict) else []
            if not recs:
                st.info("No matching instruments found. Try adjusting your filters or search terms.")
            else:
                st.success(f"Found {len(recs)} instrument{'s' if len(recs) != 1 else ''}")
                st.markdown("---")
                
                for idx, ins in enumerate(recs, 1):
                    title = f"{idx}. {ins.get('name', 'Unknown')}"
                    if ins.get('acronym'):
                        title += f" ({ins.get('acronym')})"
                    
                    with st.expander(title, expanded=(idx == 1)):
                        badge_html = ""
                        if ins.get('domain'):
                            badge_html += f'<span class="badge badge-domain">📁 {ins.get("domain")}</span>'
                        if ins.get('programme_level') and str(ins.get('programme_level', '')).strip().lower() in ['yes', 'y', 'true']:
                            badge_html += '<span class="badge badge-programme">📊 Programme-level</span>'
                        if badge_html:
                            st.markdown(badge_html, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        if ins.get('purpose'):
                            st.markdown(f"**📝 Purpose:**  \n{ins.get('purpose')}")
                        if ins.get('target_group'):
                            st.markdown(f"**👥 Target Group:** {ins.get('target_group')}")
                        if ins.get('domain'):
                            st.markdown(f"**📁 Domain:** {ins.get('domain')}")
                        if ins.get('programme_level'):
                            st.markdown(f"**📊 Programme-level:** {ins.get('programme_level')}")
        except Exception as e:
            logger.error(f"Manual search error: {e}", exc_info=True)
            st.error(f"❌ Search error: {str(e)}")
            st.info("Please try adjusting your search criteria or contact support if the issue persists.")


def main():
    st.set_page_config(
        page_title="Measurement Instrument Assistant",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: #1c83e1;
        }
        .subtitle {
            color: #6b7280;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }
        .sidebar .sidebar-content {
            background-color: #0e1117;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="main-title">📊 Measurement Instrument Assistant</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">AI-powered search for research measurement instruments</p>', unsafe_allow_html=True)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
    
    load_environment()
    
    if Config:
        original_path = Config.EXCEL_FILE_PATH
        Config.EXCEL_FILE_PATH = resolve_excel_path(Config.EXCEL_FILE_PATH)
    
    if Config:
        is_valid, errors = Config.validate()
        if not is_valid:
            st.error("⚠️ Configuration Error")
            for error in errors:
                st.error(f"• {error}")
            st.info("Please check your environment variables and configuration.")
            return
    
    agent = initialize_agent()
    
    if not agent:
        st.error("❌ Failed to initialize the AI agent. Please check the logs for details.")
        logger.error("Agent initialization failed")
        return
    
    if Config:
        excel_path = Config.EXCEL_FILE_PATH
        sheet_name = Config.EXCEL_SHEET_NAME
    else:
        excel_path = "measurement_instruments.xlsx"
        sheet_name = "Measurement Instruments"
    excel_path = resolve_excel_path(excel_path)
    
    try:
        df = load_dataframe(excel_path, sheet_name)
    except FileNotFoundError:
        st.error(f"❌ Data file not found: {excel_path}")
        logger.error(f"Excel file not found: {excel_path}")
        return
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        logger.error(f"Error loading dataframe: {e}", exc_info=True)
        return

    st.sidebar.markdown("### 🧭 Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["💬 Chat", "🔍 Manual Search", "📂 Data Management"],
        key="nav_radio",
        label_visibility="collapsed"
    )
    
    if page == "💬 Chat":
        render_chat_page(agent, df)
    elif page == "🔍 Manual Search":
        render_manual_search_page(agent, df)
    elif page == "📂 Data Management":
        render_data_management_page(df, excel_path)


if __name__ == "__main__":
    main()