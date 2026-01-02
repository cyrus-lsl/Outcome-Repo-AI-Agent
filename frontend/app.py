import streamlit as st
import pandas as pd
import os
import sys
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
            logger.info(f"Loaded environment from {env_path}")
    except ImportError:
        logger.warning("python-dotenv not available, using system environment variables")


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
        
        from backend.agent_core import MeasurementInstrumentAgent
        agent = MeasurementInstrumentAgent(excel_path, sheet_name=sheet_name)
        logger.info("Agent initialized successfully")
        return agent
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_dataframe(excel_path: str, sheet_name: str):
    """Load and cache the Excel dataframe"""
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        logger.info(f"Loaded {len(df)} instruments from {excel_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataframe: {e}", exc_info=True)
        raise


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


def call_huggingface_chat(prompt, df, validated_only=False, prog_only=False, max_results=6, agent=None):
    """Call Hugging Face chat model for responses with semantic search enhancement.

    The function uses semantic search first (if available), then uses LLM to refine results.
    """
    try:
        # Sanitize input
        prompt = sanitize_input(prompt, max_length=Config.MAX_QUERY_LENGTH if Config else 500)
        
        if not prompt:
            return {
                'text': "Please provide a valid search query.",
                'matched': [],
                'unknown': []
            }
        
        # Read HF token robustly (allow quoted tokens in .env)
        if Config and Config.HF_TOKEN:
            hf = Config.HF_TOKEN
        else:
            hf = os.environ.get("HF_TOKEN")
            if isinstance(hf, str):
                hf = hf.strip().strip('"\'')
        
        if not hf:
            logger.error("HF_TOKEN not set in environment")
            return {
                'text': "Configuration error: API token not found. Please contact the administrator.",
                'matched': [],
                'unknown': []
            }

        base_url = Config.HF_BASE_URL if Config else "https://router.huggingface.co/v1"
        model_name = Config.HF_MODEL if Config else "moonshotai/Kimi-K2-Instruct-0905"
        client = OpenAI(base_url=base_url, api_key=hf)
        logger.info(f"Processing query: {prompt[:50]}...")
    except Exception as e:
        logger.error(f"Error in initial setup: {e}", exc_info=True)
        return {
            'text': "An error occurred while initializing the search. Please try again or contact support.",
            'matched': [],
            'unknown': []
        }

    # Detect whether the user explicitly asked for instruments validated in Hong Kong
    prompt_l = str(prompt or '').lower().strip()
    
    # Check for non-substantive queries (greetings, empty, too short)
    greeting_patterns = [r'^\s*(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))\s*[!?.]*\s*$', r'^\s*[!?.]+\s*$']
    is_greeting = any(re.match(pattern, prompt_l) for pattern in greeting_patterns)
    
    if is_greeting or len(prompt_l) < 3:
        return {
            'text': "Hello! I can help you find measurement instruments. Please describe what you're looking for, for example:\n- 'mental health assessment for elderly'\n- 'physical activity questionnaire'\n- 'quality of life scale'\n\nWhat would you like to search for?",
            'matched': [],
            'unknown': []
        }
    want_validated_hk = False
    try:
        if re.search(r"validated.*hong|validated.*hk|validated in hong|validate.*hong|validate.*hk", prompt_l):
            want_validated_hk = True
    except Exception:
        want_validated_hk = False
    # Honor explicit UI toggle override
    if validated_only:
        want_validated_hk = True

    # Parse item count constraints from query (e.g., "not more than 7 items", "less than 10", "maximum 5 items")
    max_items = None
    min_items = None
    try:
        # Patterns for maximum item count
        max_patterns = [
            r'(?:not\s+more\s+than|maximum|max|at\s+most|less\s+than|fewer\s+than|under|below)\s+(\d+)\s*(?:items?|questions?|statements?)?',
            r'(\d+)\s*(?:or\s+)?(?:fewer|less)\s*(?:items?|questions?|statements?)?',
        ]
        for pattern in max_patterns:
            match = re.search(pattern, prompt_l)
            if match:
                max_items = int(match.group(1))
                logger.info(f"Detected maximum item count constraint: {max_items}")
                break
        
        # Patterns for minimum item count
        min_patterns = [
            r'(?:at\s+least|minimum|min|more\s+than|greater\s+than|over|above)\s+(\d+)\s*(?:items?|questions?|statements?)?',
            r'(\d+)\s*(?:or\s+)?(?:more|greater)\s*(?:items?|questions?|statements?)?',
        ]
        for pattern in min_patterns:
            match = re.search(pattern, prompt_l)
            if match:
                min_items = int(match.group(1))
                logger.info(f"Detected minimum item count constraint: {min_items}")
                break
        
        # Exact count (e.g., "exactly 7 items", "7 items")
        exact_pattern = r'(?:exactly|precisely|exactly\s+)?(\d+)\s*(?:items?|questions?|statements?)(?:\s+only)?'
        exact_match = re.search(exact_pattern, prompt_l)
        if exact_match and max_items is None and min_items is None:
            exact_count = int(exact_match.group(1))
            max_items = exact_count
            min_items = exact_count
            logger.info(f"Detected exact item count constraint: {exact_count}")
    except Exception as e:
        logger.warning(f"Error parsing item count constraints: {e}")

    # Helper to interpret free-text 'Validated in Hong Kong' values conservatively
    def _validated_in_hk_text(x):
        s = str(x or '').lower().strip()
        if not s or s in ('-', 'na', 'n/a'):
            return False
        if 'not validated' in s or 'not validated in hong' in s:
            return False
        if 'not in hong' in s or 'not in hong kong' in s or re.search(r'not .*hong', s):
            return False
        if re.search(r"\bno\b", s):
            return False
        if s.startswith('yes'):
            return True
        if ('hong' in s or 'hk' in s or 'hong kong' in s) and ('valid' in s or 'develop' in s or 'refer' in s):
            return True
        if 'validated' in s and 'hong' in s:
            return True
        return False

    # Optionally filter the dataframe to only instruments validated in HK
    df_for_model = df
    if want_validated_hk and 'Validated in Hong Kong' in df.columns:
        try:
            df_for_model = df[df['Validated in Hong Kong'].apply(_validated_in_hk_text)]
        except Exception:
            df_for_model = df
    # Optionally filter to programme-level metrics if requested by UI
    if prog_only and 'Programme-level metric?' in df_for_model.columns:
        try:
            df_for_model = df_for_model[df_for_model['Programme-level metric?'].astype(str).str.strip().str.lower().isin(['yes','y','true'])]
        except Exception:
            pass

    # Filter by item count constraints if specified (apply early for efficiency)
    if max_items is not None or min_items is not None:
        def parse_item_count_from_df(item_str):
            """Parse item count from dataframe cell"""
            if pd.isna(item_str):
                return None
            item_str = str(item_str).strip()
            match = re.search(r'(\d+)', item_str)
            if match:
                return int(match.group(1))
            return None
        
        if 'No. of Questions / Statements' in df_for_model.columns:
            try:
                item_counts = df_for_model['No. of Questions / Statements'].apply(parse_item_count_from_df)
                mask = pd.Series([True] * len(df_for_model), index=df_for_model.index)
                
                if max_items is not None:
                    max_mask = (item_counts <= max_items) | (item_counts.isna())
                    mask = mask & max_mask
                
                if min_items is not None:
                    min_mask = (item_counts >= min_items) | (item_counts.isna())
                    mask = mask & min_mask
                
                df_for_model = df_for_model[mask]
                logger.info(f"Pre-filtered dataframe to {len(df_for_model)} instruments matching item count constraints")
            except Exception as e:
                logger.warning(f"Error filtering by item count: {e}")

    # Try semantic search first if agent is available and has semantic search enabled
    semantic_results = []
    if agent and hasattr(agent, 'use_semantic_search') and agent.use_semantic_search:
        try:
            semantic_search_results = agent.semantic_search(prompt, max_results=max_results * 2, df_override=df_for_model, min_similarity=0.25)
            if semantic_search_results:
                # Convert to our format
                for result in semantic_search_results[:max_results]:
                    row = result['instrument']
                    semantic_results.append({
                        'name': row.get('Measurement Instrument', ''),
                        'acronym': row.get('Acronym', ''),
                        'purpose': row.get('Purpose', ''),
                        'target': row.get('Target Group(s)', ''),
                        'domain': row.get('Outcome Domain', ''),
                        'no_of_items': row.get('No. of Questions / Statements', ''),
                        'sample_q1': row.get('Sample Question / Statement - 1', ''),
                        'sample_q2': row.get('Sample Question / Statement - 2', ''),
                        'sample_q3': row.get('Sample Question / Statement - 3', ''),
                        'scale': row.get('Scale', ''),
                        'scoring': row.get('Scoring', ''),
                        'validated': row.get('Validated in Hong Kong', ''),
                        'programme_level': row.get('Programme-level metric?', ''),
                        'download_eng': row.get('Download (Eng)', ''),
                        'download_chi': row.get('Download (Chi)', ''),
                        'citation': row.get('Citation', ''),
                        'semantic_score': result.get('semantic_score', 0)
                    })
        except Exception as e:
            if os.getenv('DEBUG_HF') == '1':
                print(f'[DEBUG] Semantic search error: {e}')

    # Build instrument lines and name list (use semantic results if available, otherwise all)
    inst_lines = []
    instrument_names = []
    if semantic_results:
        # Use semantic search results to build a focused list for LLM
        for ins in semantic_results:
            name = ins.get('name', '').strip()
            if name:
                instrument_names.append(name)
                items = str(ins.get('no_of_items', '')).strip() or 'N/A'
                inst_lines.append(f"{name} | {ins.get('domain', '')} | {ins.get('target', '')} | Items: {items} | Programme-level: {ins.get('programme_level', '')}")
    else:
        # Fallback to all instruments
        for _, row in df_for_model.iterrows():
            name = str(row.get('Measurement Instrument', '')).strip()
            if not name:
                continue
            domain = str(row.get('Outcome Domain', '')).strip()
            target = str(row.get('Target Group(s)', '')).strip()
            items = str(row.get('No. of Questions / Statements', '')).strip() or 'N/A'
            prog_flag = str(row.get('Programme-level metric?', '')).strip()
            inst_lines.append(f"{name} | {domain} | {target} | Items: {items} | Programme-level: {prog_flag}")
            instrument_names.append(name)

    # If we have good semantic results, use them directly (faster and more accurate)
    if semantic_results and len(semantic_results) >= max_results // 2:
        matched = semantic_results[:max_results]
        unknown = []
    else:
        # Use LLM to refine or get additional results
        item_constraint_note = ""
        if max_items is not None:
            item_constraint_note += f"IMPORTANT: The user requested instruments with at most {max_items} items. Only select instruments where the item count is {max_items} or fewer.\n"
        if min_items is not None:
            item_constraint_note += f"IMPORTANT: The user requested instruments with at least {min_items} items. Only select instruments where the item count is {min_items} or more.\n"
        if max_items is not None and min_items is not None and max_items == min_items:
            item_constraint_note = f"IMPORTANT: The user requested instruments with exactly {max_items} items. Only select instruments with exactly {max_items} items.\n"
        
        system_message = (
            "You are an assistant that selects only instrument NAMES from the provided DATABASE. "
            "You will be given a list of instruments with short metadata (Domain, Target groups, Item count, Programme-level flag). Do NOT invent any names. "
            "IMPORTANT: If the user query is a greeting (like 'hi', 'hello'), a non-substantive query, or too vague to match any instruments, return an EMPTY JSON array: []. "
            "Only return instrument names when the query is a substantive search request about measurement instruments. "
            f"Return ONLY a valid JSON array (e.g. [\"Name A\", \"Name B\"] or [] for non-substantive queries) containing up to {max_results} names that appear in the provided list."
        )

        user_message = (
            f"DATABASE OF MEASUREMENT INSTRUMENTS:\nEach line: Name | Domain | Target(s) | Items: <count> | Programme-level\n{chr(10).join(inst_lines)}\n\n"
            f"USER QUERY: {prompt}\n\n"
            + ("NOTE: The user requested only instruments validated in Hong Kong.\n" if want_validated_hk else "")
            + ("NOTE: The user requested only programme-level instruments.\n" if prog_only else "")
            + (item_constraint_note if item_constraint_note else "")
            + f"If the query is a greeting or not substantive enough to match instruments, return []. Otherwise, choose up to {max_results} instruments from the provided list that best match the user query, considering Domain, Purpose, Target groups, and Item count constraints. Return a JSON array with instrument names only."
        )

        try:
            model_name = Config.HF_MODEL if Config else "moonshotai/Kimi-K2-Instruct-0905"
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
                max_tokens=400,
                temperature=0.0,
            )
        except Exception as e:
            logger.error(f"LLM API call failed: {e}", exc_info=True)
            # If LLM fails but we have semantic results, use those
            if semantic_results:
                matched = semantic_results[:max_results]
                unknown = []
            else:
                return {
                    'text': "I encountered an error while processing your request. Please try again or rephrase your query.",
                    'matched': [],
                    'unknown': []
                }

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
        matched = semantic_results.copy() if semantic_results else []
        unknown = []
        available_set = {n.lower() for n in instrument_names}
        existing_names = {m.get('name', '').lower() for m in matched}
        
        for nm in names:
            if nm.strip().lower() not in available_set:
                unknown.append(nm)
                continue
            if nm.strip().lower() in existing_names:
                continue  # Already in semantic results
            try:
                match = df_for_model[df_for_model['Measurement Instrument'].str.contains(nm, case=False, na=False, regex=False)]
            except TypeError:
                match = df_for_model[df_for_model['Measurement Instrument'].str.contains(re.escape(nm), case=False, na=False)]
            if not match.empty and len(matched) < max_results:
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
                    'semantic_score': None
                })

        if unknown and os.getenv('DEBUG_HF') == '1':
            logger.debug(f'Model returned names not in spreadsheet: {unknown}')

    if not matched:
        # Check if this was a greeting or non-substantive query
        if is_greeting or len(prompt_l) < 3:
            return {
                'text': "Hello! I can help you find measurement instruments. Please describe what you're looking for, for example:\n- 'mental health assessment for elderly'\n- 'physical activity questionnaire'\n- 'quality of life scale'\n\nWhat would you like to search for?",
                'matched': [],
                'unknown': unknown
            }
        return {'text': 'No matching instruments found in the database. Please try a more specific query.', 'matched': [], 'unknown': unknown}

    def _lower(x):
        try:
            return str(x).lower()
        except Exception:
            return ''

    # Sort by semantic score if available (higher is better)
    matched.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)
    
    # Extract meaningful tokens (exclude common words and very short tokens)
    common_words = {'the', 'and', 'or', 'for', 'with', 'from', 'that', 'this', 'what', 'which', 'are', 'can', 'how', 'when', 'where', 'who', 'why', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'have', 'has', 'had', 'was', 'were', 'been', 'being', 'do', 'does', 'did', 'get', 'got', 'give', 'take', 'make', 'see', 'know', 'think', 'want', 'need', 'use', 'find', 'search', 'look', 'show', 'tell', 'ask', 'help', 'about', 'measure', 'assessment', 'scale', 'questionnaire', 'test', 'tool', 'instrument'}
    qtokens = [t for t in re.findall(r"\w+", _lower(prompt)) if len(t) > 2 and t not in common_words]
    
    # If no meaningful tokens after filtering and no semantic scores, return no matches
    has_semantic_scores = any(ins.get('semantic_score') for ins in matched)
    if not qtokens and not has_semantic_scores:
        return {'text': 'Please provide a more specific query. For example, try:\n- "mental health assessment"\n- "physical activity questionnaire"\n- "quality of life scale for elderly"\n\nWhat are you looking for?', 'matched': [], 'unknown': unknown}
    
    # If we have semantic scores, trust them and skip strict token filtering
    if has_semantic_scores:
        filtered = matched
        filter_note = ''
    else:
        # Apply token-based filtering for LLM results
        filtered = []
        for ins in matched:
            hay = ' '.join([_lower(ins.get('name', '')), _lower(ins.get('domain', '')), _lower(ins.get('purpose', '')), _lower(ins.get('target', ''))])
            # Require at least one meaningful token match
            if any(tok in hay for tok in qtokens):
                filtered.append(ins)

        filter_note = ''

        if not filtered:
            filtered = matched
            filter_note = ' (did not find stricter matches; showing best matches)'

    if want_validated_hk:
        validated_filtered = [ins for ins in filtered if _validated_in_hk_text(ins.get('validated', ''))]
        if validated_filtered:
            filtered = validated_filtered
            filter_note = ''
        else:
            return {'text': 'No instruments validated in Hong Kong found matching the query.', 'matched': [], 'unknown': unknown}

    # Filter by item count constraints if specified
    if max_items is not None or min_items is not None:
        def parse_item_count(item_str):
            """Parse item count from string (e.g., '10', '10-15', '10 items')"""
            if not item_str or pd.isna(item_str):
                return None
            item_str = str(item_str).strip()
            # Try to extract first number (for ranges like "10-15", take the first)
            match = re.search(r'(\d+)', item_str)
            if match:
                return int(match.group(1))
            return None
        
        item_filtered = []
        for ins in filtered:
            item_count = parse_item_count(ins.get('no_of_items', ''))
            if item_count is None:
                # If item count is not available, skip this instrument when filtering by items
                # (user explicitly asked for item count, so we should only show instruments with known counts)
                continue
            
            # Check constraints
            if max_items is not None and item_count > max_items:
                continue
            if min_items is not None and item_count < min_items:
                continue
            
            item_filtered.append(ins)
        
        if item_filtered:
            filtered = item_filtered
            filter_note = ''
            logger.info(f"Filtered to {len(filtered)} instruments matching item count constraints")
        else:
            constraint_text = []
            if max_items is not None:
                constraint_text.append(f"maximum {max_items} items")
            if min_items is not None:
                constraint_text.append(f"minimum {min_items} items")
            constraint_str = " and ".join(constraint_text)
            return {
                'text': f'No instruments found matching your query with {constraint_str}. Please try adjusting your search criteria.',
                'matched': [],
                'unknown': unknown
            }

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
    
    # Add custom CSS for better styling
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
    
    st.subheader("💬 Chat with AI Assistant")
    
    # Improved info section with better styling
    with st.container():
        st.markdown("""
        <div class="info-box">
            <strong>💡 How to use:</strong><br>
            Ask questions in natural language, for example:<br>
            • "mental health assessment for elderly"<br>
            • "physical activity questionnaire"<br>
            • "quality of life scale for youth"
        </div>
        """, unsafe_allow_html=True)

    # Initialize checkbox state
    if 'chat_validated_only' not in st.session_state:
        st.session_state.chat_validated_only = False
    if 'chat_prog_only' not in st.session_state:
        st.session_state.chat_prog_only = False

    # Fixed controls at bottom, above chat input
    # These will be placed in Streamlit's bottom container automatically
    filter_col1, filter_col2 = st.columns([1, 1])
    with filter_col1:
        st.session_state.chat_validated_only = st.checkbox("✅ HK-validated only", value=st.session_state.chat_validated_only, help="Show only instruments validated in Hong Kong")
    with filter_col2:
        st.session_state.chat_prog_only = st.checkbox("📊 Programme-level only", value=st.session_state.chat_prog_only, help="Show only programme-level metrics")

    # Chat input - always at bottom (most bottom)
    if prompt := st.chat_input("Ask about measurement instruments..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching..."):
                response = call_huggingface_chat(
                    prompt,
                    df,
                    validated_only=st.session_state.chat_validated_only,
                    prog_only=st.session_state.chat_prog_only,
                    max_results=8,
                    agent=agent,
                )

            if isinstance(response, dict) and 'matched' in response:
                matched = response.get('matched', [])
                if not matched:
                    st.info(response.get('text', 'No matching instruments found in the database.'))
                else:
                    # Show result count
                    st.markdown(f'<div class="result-count">Found {len(matched)} instrument{"s" if len(matched) != 1 else ""}</div>', unsafe_allow_html=True)
                    
                    # Display instruments with improved layout
                    for idx, ins in enumerate(matched, 1):
                        # Build expander title with badges
                        title_parts = [f"{idx}. {ins.get('name', 'Unknown')}"]
                        if ins.get('acronym'):
                            title_parts.append(f"({ins.get('acronym')})")
                        
                        with st.expander(" ".join(title_parts), expanded=(idx == 1)):
                            # Domain badge
                            if ins.get('domain'):
                                st.markdown(f'<span class="badge badge-domain">📁 {ins.get("domain")}</span>', unsafe_allow_html=True)
                            
                            # Validation and programme badges
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
                            
                            st.markdown("---")
                            
                            # Purpose
                            if ins.get('purpose'):
                                st.markdown(f"**📝 Purpose:**  \n{ins.get('purpose')}")
                            
                            # Target group
                            if ins.get('target'):
                                st.markdown(f"**👥 Target Group:** {ins.get('target')}")
                            
                            # Details in columns
                            col1, col2 = st.columns(2)
                            with col1:
                                if ins.get('no_of_items'):
                                    st.metric("Items", ins.get('no_of_items'))
                                if ins.get('scale'):
                                    st.markdown(f"**Scale:** {ins.get('scale')}")
                            with col2:
                                if ins.get('scoring'):
                                    st.markdown(f"**Scoring:** {ins.get('scoring')}")
                            
                            # Sample questions
                            sample_questions = []
                            for q_key in ['sample_q1', 'sample_q2', 'sample_q3']:
                                if ins.get(q_key):
                                    sample_questions.append(ins.get(q_key))
                            
                            if sample_questions:
                                with st.expander("📋 Sample Questions", expanded=False):
                                    for q in sample_questions:
                                        st.markdown(f"• {q}")
                            
                            # Downloads and citation
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


def render_manual_search_page(agent, df):
    st.subheader("🧭 Manual Search")
    st.markdown("Use filters to narrow down your search for measurement instruments.")
    
    # All filters shown directly
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

    # Additional filters
    st.markdown("**Additional Filters:**")
    mcol1, mcol2 = st.columns([1, 1])
    with mcol1:
        manual_validated_only = st.checkbox("✅ Require HK-validated only", value=False, help="Show only HK-validated instruments")
    with mcol2:
        manual_prog_only = st.checkbox("📊 Programme-level only", value=False, help="Show only programme-level metrics")

    # Search button
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
                        # Badges
                        badge_html = ""
                        if ins.get('domain'):
                            badge_html += f'<span class="badge badge-domain">📁 {ins.get("domain")}</span>'
                        if ins.get('programme_level') and str(ins.get('programme_level', '')).strip().lower() in ['yes', 'y', 'true']:
                            badge_html += '<span class="badge badge-programme">📊 Programme-level</span>'
                        if badge_html:
                            st.markdown(badge_html, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Information
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
    
    # Enhanced header with better styling
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
        st.caption("Powered by Hugging Face AI")
    
    load_environment()
    
    # Validate configuration
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
    
    try:
        if Config:
            excel_path = Config.EXCEL_FILE_PATH
            sheet_name = Config.EXCEL_SHEET_NAME
        else:
            excel_path = "measurement_instruments.xlsx"
            sheet_name = "Measurement Instruments"
        
        df = load_dataframe(excel_path, sheet_name)
    except FileNotFoundError:
        st.error(f"❌ Data file not found: {excel_path}")
        logger.error(f"Excel file not found: {excel_path}")
        return
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        logger.error(f"Error loading dataframe: {e}", exc_info=True)
        return

    # Enhanced sidebar navigation
    st.sidebar.markdown("### 🧭 Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["💬 Chat", "🔍 Manual Search"],
        key="nav_radio",
        label_visibility="collapsed"
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**📚 Database:** {len(df)} instruments")
    if 'Outcome Domain' in df.columns:
        domains = df['Outcome Domain'].dropna().unique()
        st.sidebar.markdown(f"**📁 Domains:** {len(domains)}")
    
    if page == "💬 Chat":
        render_chat_page(agent, df)
    else:
        render_manual_search_page(agent, df)


if __name__ == "__main__":
    main()