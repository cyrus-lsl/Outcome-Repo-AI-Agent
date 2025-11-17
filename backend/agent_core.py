import pandas as pd
import os
import difflib
import re
from openai import OpenAI


def _validated_in_hk_text(x: object) -> bool:
    """Heuristic: return True when the text indicates validation in Hong Kong.

    The spreadsheet entries are heterogeneous ("Yes. Refer to...", long
    sentences, "Developed and validated in HK."). This tries to detect
    positive signals and avoids false positives when the text explicitly
    says "Not validated in Hong Kong".
    """
    s = str(x or '').lower()
    s = s.strip()
    if not s or s in ('-', 'na', 'n/a'):
        return False
    # Negative patterns first
    if 'not validated' in s or 'not validated in hong' in s:
        return False
    # phrases like 'not in Hong Kong' or 'not ... hong' should be negative
    if 'not in hong' in s or 'not in hong kong' in s or re.search(r'not .*hong', s):
        return False
    # explicit 'no' as whole word is negative
    if re.search(r"\bno\b", s):
        return False
    # Positive when explicit yes or mentions HK/Hong Kong together with 'valid'
    if s.startswith('yes'):
        return True
    if ('hong' in s or 'hk' in s or 'hong kong' in s) and ('valid' in s or 'develop' in s or 'refer' in s):
        return True
    # fallback: any explicit 'validated' mention counts if not negated
    if 'validated' in s and 'hong' in s:
        return True
    return False

class InstrumentSearcher:
    def __init__(self, excel_file_path, sheet_name=None, header_row=None):
        read_kwargs = {}
        if sheet_name is not None:
            read_kwargs['sheet_name'] = sheet_name
        if header_row is not None:
            read_kwargs['header'] = header_row
        self.df = pd.read_excel(excel_file_path, **read_kwargs).fillna('')
        # Ensure we have a combined_text column for local keyword scoring. If the
        # spreadsheet already provides one, keep it; otherwise construct it from
        # commonly useful fields so local fallback works even without precomputed text.
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
    
    def _get_client(self):
        if self._client is None:
            hf = os.environ.get('HF_TOKEN')
            # If the token is stored with surrounding quotes in .env, strip them.
            if isinstance(hf, str):
                hf = hf.strip().strip('\"\'')
            if not hf:
                raise RuntimeError('HF_TOKEN not set in environment; cannot call Hugging Face APIs')
            self._client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf)
        return self._client
    
    def search(self, query, max_results=5, df_override=None):
        """Search for instruments matching the query. Optional df_override can
        be provided to restrict the search to a subset of the dataset (e.g.
        programme-level instruments only)."""
        df_local = df_override if df_override is not None else self.df
        instrument_names = df_local['Measurement Instrument'].tolist()
        
        prompt = f"""You are a professional assistant to help users find suitable measurement instruments. Your total max output is 300 words at anytime. Available measurement instruments:
{chr(10).join([f'- {name}' for name in instrument_names if name])}

User query: "{query}"

Return ONLY the names of the most relevant instruments (max {max_results}) that match the query, one per line. No explanations, just the instrument names:"""

        response = None
        # Only attempt an HF call if HF_TOKEN is available; otherwise skip to
        # local fallback scoring. This avoids crashing when the environment
        # doesn't provide a token.
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
                # Print error when debugging is enabled so the user can see HF call issues
                if os.getenv('DEBUG_HF') == '1':
                    print('\n[DEBUG_HF] Exception while calling HF/OpenAI router:')
                    import traceback
                    traceback.print_exc()
                response = None

        results = []
        suggested_names = []
        if response is not None:
            try:
                llm_output = response.choices[0].message.content.strip()
                # Optional debug: print raw LLM output when DEBUG_HF=1
                if os.getenv('DEBUG_HF') == '1':
                    print('\n[DEBUG_HF] raw LLM output:')
                    print(llm_output)

                # Try JSON parse first
                import json
                try:
                    parsed = json.loads(llm_output)
                    if isinstance(parsed, list):
                        suggested_names = [str(x).strip() for x in parsed if x]
                except Exception:
                    # Fallback to line-splitting
                    suggested_names = [line.strip(' -"') for line in llm_output.split('\n') if line.strip()]
                if os.getenv('DEBUG_HF') == '1':
                    print('\n[DEBUG_HF] parsed suggested_names:', suggested_names)
            except Exception:
                suggested_names = []

        if suggested_names:
            # Try to map suggested names to actual spreadsheet rows. We prefer
            # case-insensitive substring matches, but as a fallback use
            # fuzzy matching (difflib) to handle small name variations from the LLM.
            for name in suggested_names[:max_results]:
                # direct substring match first
                # Use literal substring matching to avoid regex pitfalls when
                # instrument names contain special characters. pandas' str.contains
                # supports regex=False for literal matching.
                try:
                    match = df_local[
                        df_local['Measurement Instrument'].str.contains(name, case=False, na=False, regex=False)
                    ]
                except TypeError:
                    # Older pandas versions may not accept regex kw; fallback to
                    # escaping the pattern for regex matching.
                    match = df_local[
                        df_local['Measurement Instrument'].str.contains(re.escape(name), case=False, na=False)
                    ]
                # if not found, try fuzzy match against available instrument names
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
                if not match.empty:
                    row = match.iloc[0]
                    results.append({'instrument': row, 'similarity_score': None})
            return results

        qtokens = [t.strip().lower() for t in str(query).split() if t.strip()]
        if not qtokens:
            return []
        # Reset index so we can use positional iloc safely on the filtered df
        df_for_scoring = df_local.reset_index(drop=True)
        scores = []
        for i, row in df_for_scoring.iterrows():
            # Use combined_text if provided, otherwise join all string fields in the row
            text = str(row.get('combined_text', '')).lower()
            if not text:
                text = ' '.join([str(v) for v in row.values if isinstance(v, (str, int, float))]).lower()
            score = sum(1 for t in qtokens if t in text)
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        for score, idx in scores[:max_results]:
            results.append({'instrument': df_for_scoring.iloc[idx], 'similarity_score': float(score)})
        return results

    def search_instruments(self, query, top_k=3, df_override=None):
        return self.search(query, max_results=top_k, df_override=df_override)

    def manual_search(self, beneficiaries=None, measure=None, validated='both', prog_level='both', top_k=10):
        # Build a natural-language query including the provided filters and
        # let the Hugging Face LLM pick the best instruments from the dataset.
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

        # Determine whether we should restrict the dataset to programme-level
        # instruments or HK-validated instruments before delegating selection to
        # the LLM.
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

        # Delegate selection to the LLM (search() uses HF-first). Pass df_override
        # so the LLM is only given instruments from the requested subset.
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
        """Compatibility wrapper: accept either a list of recommendations or the
        dict returned by `manual_search` and return a human-friendly string.
        """
        if isinstance(results, dict) and 'recommendations' in results:
            recs = results.get('recommendations', [])
            query = results.get('query')
            header = f"Results for query: {query}\n\n" if query else ""
            body = self.format_results(recs)
            return header + body

        return self.format_results(results)

    def format_results(self, results):
        """Format search results nicely"""
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