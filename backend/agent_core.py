import pandas as pd
import os
import difflib
import re
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Lazy import for sentence-transformers (optional dependency)
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False
    SentenceTransformer = None


def _validated_in_hk_text(x: object) -> bool:
    s = str(x or '').lower()
    s = s.strip()
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

class InstrumentSearcher:
    def __init__(self, excel_file_path, sheet_name=None, header_row=None, use_semantic_search=True):
        read_kwargs = {}
        if sheet_name is not None:
            read_kwargs['sheet_name'] = sheet_name
        if header_row is not None:
            read_kwargs['header'] = header_row
        self.df = pd.read_excel(excel_file_path, **read_kwargs).fillna('')
        if 'combined_text' not in self.df.columns:
            cols_to_combine = []
            for c in ('Measurement Instrument', 'Acronym', 'Purpose', 'Target Group(s)', 'Outcome Domain'):
                if c in self.df.columns:
                    cols_to_combine.append(self.df[c].astype(str))
            if cols_to_combine:
                self.df['combined_text'] = (pd.Series([''] * len(self.df)) + ' ' +
                                            pd.concat(cols_to_combine, axis=1).apply(lambda row: ' '.join([str(x) for x in row if x]), axis=1))
            else:
                self.df['combined_text'] = ''
        self._client = None
        
        # Initialize semantic search
        self.use_semantic_search = use_semantic_search and SEMANTIC_SEARCH_AVAILABLE
        self._embedding_model = None
        self._instrument_embeddings = None
        if self.use_semantic_search:
            try:
                # Use a lightweight, multilingual model that works well for search
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                # Pre-compute embeddings for all instruments (lazy loading)
                self._instrument_embeddings = None
            except Exception as e:
                if os.getenv('DEBUG_HF') == '1':
                    print(f'[DEBUG] Failed to initialize semantic search: {e}')
                self.use_semantic_search = False
    
    def _get_client(self):
        if self._client is None:
            hf = os.environ.get('HF_TOKEN')
            if isinstance(hf, str):
                hf = hf.strip().strip('\"\'')
            if not hf:
                raise RuntimeError('HF_TOKEN not set in environment; cannot call Hugging Face APIs')
            self._client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf)
        return self._client
    
    def _get_instrument_embeddings(self, df_local=None):
        """Lazy load and cache instrument embeddings"""
        df_to_use = df_local if df_local is not None else self.df
        
        # Check if we need to recompute (different dataframe or not computed yet)
        if self._instrument_embeddings is None or df_local is not None:
            if not self.use_semantic_search or self._embedding_model is None:
                return None
            
            # Build searchable text for each instrument
            texts = []
            for _, row in df_to_use.iterrows():
                parts = []
                if row.get('Measurement Instrument'):
                    parts.append(str(row.get('Measurement Instrument')))
                if row.get('Acronym'):
                    parts.append(str(row.get('Acronym')))
                if row.get('Purpose'):
                    parts.append(str(row.get('Purpose')))
                if row.get('Outcome Domain'):
                    parts.append(str(row.get('Outcome Domain')))
                if row.get('Target Group(s)'):
                    parts.append(str(row.get('Target Group(s)')))
                texts.append(' '.join(parts))
            
            # Compute embeddings
            try:
                self._instrument_embeddings = self._embedding_model.encode(texts, show_progress_bar=False)
            except Exception as e:
                if os.getenv('DEBUG_HF') == '1':
                    print(f'[DEBUG] Failed to compute embeddings: {e}')
                return None
        
        return self._instrument_embeddings
    
    def semantic_search(self, query, max_results=5, df_override=None, min_similarity=0.3):
        """Perform semantic search using embeddings"""
        if not self.use_semantic_search or self._embedding_model is None:
            return []
        
        df_local = df_override if df_override is not None else self.df
        embeddings = self._get_instrument_embeddings(df_local)
        
        if embeddings is None:
            return []
        
        try:
            # Encode query
            query_embedding = self._embedding_model.encode([query], show_progress_bar=False)
            
            # Compute cosine similarity
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:max_results]
            
            results = []
            for idx in top_indices:
                similarity = float(similarities[idx])
                if similarity >= min_similarity:
                    row = df_local.iloc[idx]
                    results.append({
                        'instrument': row,
                        'similarity_score': similarity,
                        'semantic_score': similarity
                    })
            
            return results
        except Exception as e:
            if os.getenv('DEBUG_HF') == '1':
                print(f'[DEBUG] Semantic search error: {e}')
            return []
    
    def search(self, query, max_results=5, df_override=None, use_semantic=True):
        """Hybrid search: semantic search + LLM fallback"""
        df_local = df_override if df_override is not None else self.df
        results = []
        
        # Try semantic search first if available
        if use_semantic and self.use_semantic_search:
            semantic_results = self.semantic_search(query, max_results=max_results, df_override=df_override)
            if semantic_results:
                # Use semantic results if we have good matches
                results = semantic_results
                if os.getenv('DEBUG_HF') == '1':
                    print(f'[DEBUG] Semantic search found {len(results)} results')
        
        # If semantic search didn't return enough results, try LLM-based search
        if len(results) < max_results:
            instrument_names = df_local['Measurement Instrument'].tolist()
            
            prompt = f"""You are a professional assistant to help users find suitable measurement instruments. Your total max output is 300 words at anytime. Available measurement instruments:
{chr(10).join([f'- {name}' for name in instrument_names if name])}

User query: "{query}"

Return ONLY the names of the most relevant instruments (max {max_results}) that match the query, one per line. No explanations, just the instrument names:"""

            response = None
            try:
                client = self._get_client()
            except RuntimeError:
                if os.getenv('DEBUG_HF') == '1':
                    print('\n[DEBUG_HF] HF_TOKEN not set; skipping HF call and using local fallback')
                client = None

            if client is not None:
                try:
                    response = client.chat.completions.create(
                        model="moonshotai/Kimi-K2-Instruct-0905",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,
                        temperature=0.1
                    )
                except Exception:
                    if os.getenv('DEBUG_HF') == '1':
                        print('\n[DEBUG_HF] Exception while calling HF/OpenAI router:')
                        import traceback
                        traceback.print_exc()
                    response = None

            suggested_names = []
            if response is not None:
                try:
                    llm_output = response.choices[0].message.content.strip()
                    if os.getenv('DEBUG_HF') == '1':
                        print('\n[DEBUG_HF] raw LLM output:')
                        print(llm_output)

                    import json
                    try:
                        parsed = json.loads(llm_output)
                        if isinstance(parsed, list):
                            suggested_names = [str(x).strip() for x in parsed if x]
                    except Exception:
                        suggested_names = [line.strip(' -"') for line in llm_output.split('\n') if line.strip()]
                    if os.getenv('DEBUG_HF') == '1':
                        print('\n[DEBUG_HF] parsed suggested_names:', suggested_names)
                except Exception:
                    suggested_names = []

            # Add LLM results that aren't already in semantic results
            existing_names = {r['instrument'].get('Measurement Instrument', '') for r in results}
            for name in suggested_names[:max_results]:
                if name in existing_names:
                    continue
                try:
                    match = df_local[
                        df_local['Measurement Instrument'].str.contains(name, case=False, na=False, regex=False)
                    ]
                except TypeError:
                    match = df_local[
                        df_local['Measurement Instrument'].str.contains(re.escape(name), case=False, na=False)
                    ]
                if match.empty:
                    candidates = difflib.get_close_matches(name, instrument_names, n=1, cutoff=0.6)
                    if candidates:
                        cand = candidates[0]
                        try:
                            match = df_local[
                                df_local['Measurement Instrument'].str.contains(cand, case=False, na=False, regex=False)
                            ]
                        except TypeError:
                            match = df_local[
                                df_local['Measurement Instrument'].str.contains(re.escape(cand), case=False, na=False)
                            ]
                        if os.getenv('DEBUG_HF') == '1':
                            print(f"[DEBUG_HF] Fuzzy-matched '{name}' -> '{cand}'")
                if not match.empty and len(results) < max_results:
                    row = match.iloc[0]
                    results.append({'instrument': row, 'similarity_score': None, 'semantic_score': None})
        
        # Fallback to keyword-based search if still not enough results
        if len(results) < max_results:
            qtokens = [t.strip().lower() for t in str(query).split() if t.strip()]
            if qtokens:
                df_for_scoring = df_local.reset_index(drop=True)
                scores = []
                existing_indices = {df_local.index.get_loc(r['instrument'].name) for r in results if hasattr(r['instrument'], 'name')}
                for i, row in df_for_scoring.iterrows():
                    if i in existing_indices:
                        continue
                    text = str(row.get('combined_text', '')).lower()
                    if not text:
                        text = ' '.join([str(v) for v in row.values if isinstance(v, (str, int, float))]).lower()
                    score = sum(1 for t in qtokens if t in text)
                    if score > 0:
                        scores.append((score, i))
                scores.sort(reverse=True)
                for score, idx in scores[:max_results - len(results)]:
                    results.append({'instrument': df_for_scoring.iloc[idx], 'similarity_score': float(score), 'semantic_score': None})
        
        # Sort results by semantic score (if available) or similarity score
        results.sort(key=lambda x: x.get('semantic_score', x.get('similarity_score', 0)), reverse=True)
        return results[:max_results]

    def search_instruments(self, query, top_k=3, df_override=None):
        return self.search(query, max_results=top_k, df_override=df_override)

    def manual_search(self, beneficiaries=None, measure=None, validated='both', prog_level='both', top_k=10):
        parts = []
        if measure:
            parts.append(f"Measure: {measure}")
        if beneficiaries:
            if isinstance(beneficiaries, list):
                parts.append("Beneficiaries: " + ", ".join(beneficiaries))
            else:
                parts.append(f"Beneficiaries: {beneficiaries}")
        if validated and validated != 'both':
            parts.append(f"Validated in Hong Kong: {validated}")
        if prog_level and prog_level != 'both':
            parts.append(f"Program-level metric: {prog_level}")

        q = '; '.join(parts).strip() or (measure or '')

        df_override = None
        try:
            df_override = self.df
            if prog_level and prog_level != 'both':
                want = str(prog_level).strip().lower()
                if want in ('yes', 'y', 'true', '1'):
                    df_override = df_override[df_override.get('Programme-level metric?', '').astype(str).str.strip().str.lower() == 'yes']
                elif want in ('no', 'n', 'false', '0'):
                    df_override = df_override[df_override.get('Programme-level metric?', '').astype(str).str.strip().str.lower() == 'no']

            if validated and validated != 'both' and 'Validated in Hong Kong' in self.df.columns:
                want_v = str(validated).strip().lower()
                if want_v in ('yes', 'y', 'true', '1'):
                    df_override = df_override[df_override['Validated in Hong Kong'].apply(_validated_in_hk_text)]
                elif want_v in ('no', 'n', 'false', '0'):
                    df_override = df_override[~df_override['Validated in Hong Kong'].apply(_validated_in_hk_text)]
        except Exception:
            df_override = None

        results = self.search(q, max_results=top_k, df_override=df_override)

        formatted = {'query': q, 'recommendations': []}
        for r in results:
            ins = r['instrument']
            formatted['recommendations'].append({
                'name': ins.get('Measurement Instrument', ''),
                'acronym': ins.get('Acronym', ''),
                'purpose': ins.get('Purpose', ''),
                'target_group': ins.get('Target Group(s)', ''),
                'domain': ins.get('Outcome Domain', ''),
                'programme_level': ins.get('Programme-level metric?', ''),
                'similarity_score': r.get('similarity_score'),
            })
        return formatted

    def format_response(self, results):
        if isinstance(results, dict) and 'recommendations' in results:
            recs = results.get('recommendations', [])
            query = results.get('query')
            header = f"Results for query: {query}\n\n" if query else ""
            body = self.format_results(recs)
            return header + body

        return self.format_results(results)

    def format_results(self, results):
        if not results:
            return "No matching instruments found."
        
        output = f"Found {len(results)} matching instruments:\n\n"
        for i, instrument in enumerate(results, 1):
            output += f"{i}. {instrument['name']}"
            if instrument['acronym']:
                output += f" ({instrument['acronym']})"
            output += f"\n   Purpose: {instrument['purpose']}\n"
            output += f"   Target: {instrument['target_group']}\n"
            output += f"   Domain: {instrument['domain']}\n\n"
        
        return output

MeasurementInstrumentAgent = InstrumentSearcher
