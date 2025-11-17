import pandas as pd
import os
from openai import OpenAI

class InstrumentSearcher:
    def __init__(self, excel_file_path, sheet_name=None, header_row=None):
        read_kwargs = {}
        if sheet_name is not None:
            read_kwargs['sheet_name'] = sheet_name
        if header_row is not None:
            read_kwargs['header'] = header_row
        self.df = pd.read_excel(excel_file_path, **read_kwargs).fillna('')
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            hf = os.environ.get('HF_TOKEN')
            if not hf:
                raise RuntimeError('HF_TOKEN not set in environment; cannot call Hugging Face APIs')
            self._client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf)
        return self._client
    
    def search(self, query, max_results=5):
        """Search for instruments matching the query"""
        instrument_names = self.df['Measurement Instrument'].tolist()
        
        prompt = f"""You are a professional assistant to help users find suitable measurement instruments. Your total max output is 300 words at anytime. Available measurement instruments:
{chr(10).join([f'- {name}' for name in instrument_names if name])}

User query: "{query}"

Return ONLY the names of the most relevant instruments (max {max_results}) that match the query, one per line. No explanations, just the instrument names:"""

        client = self._get_client()
        response = None
        try:
            response = client.chat.completions.create(
                model="moonshotai/Kimi-K2-Instruct-0905",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
        except Exception as e:
            print(e)

        results = []
        suggested_names = []
        if response is not None:
            try:
                llm_output = response.choices[0].message.content.strip()
                suggested_names = [line.strip(' -') for line in llm_output.split('\n') if line.strip()]
            except Exception:
                suggested_names = []

        if suggested_names:
            for name in suggested_names[:max_results]:
                match = self.df[
                    self.df['Measurement Instrument'].str.contains(name, case=False, na=False)
                ]
                if not match.empty:
                    row = match.iloc[0]
                    results.append({'instrument': row, 'similarity_score': None})
            return results

        qtokens = [t.strip().lower() for t in str(query).split() if t.strip()]
        if not qtokens:
            return []
        scores = []
        for i, row in self.df.iterrows():
            text = str(row.get('combined_text', '')).lower()
            score = sum(1 for t in qtokens if t in text)
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        for score, idx in scores[:max_results]:
            results.append({'instrument': self.df.iloc[idx], 'similarity_score': float(score)})
        return results

    def search_instruments(self, query, top_k=3):
        return self.search(query, max_results=top_k)

    def manual_search(self, beneficiaries=None, measure=None, validated='both', prog_level='both', top_k=10):
        parts = []
        if measure:
            parts.append(str(measure))
        if beneficiaries:
            if isinstance(beneficiaries, list):
                parts.extend([str(b) for b in beneficiaries])
            else:
                parts.append(str(beneficiaries))
        q = ' '.join(parts).strip() or (measure or '')
        results = self.search(q, max_results=top_k)
        formatted = {'query': q, 'recommendations': []}
        for r in results:
            ins = r['instrument']
            formatted['recommendations'].append({
                'name': ins.get('Measurement Instrument', ''),
                'acronym': ins.get('Acronym', ''),
                'purpose': ins.get('Purpose', ''),
                'target_group': ins.get('Target Group(s)', ''),
                'domain': ins.get('Outcome Domain', ''),
                'similarity_score': None,
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