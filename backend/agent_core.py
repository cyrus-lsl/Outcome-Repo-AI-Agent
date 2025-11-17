import pandas as pd
import os
import numpy as np

class MeasurementInstrumentAgent:
    def __init__(self, excel_file_path, sheet_name=None, header_row=None):
        read_kwargs = {}
        if sheet_name is not None:
            read_kwargs['sheet_name'] = sheet_name
        if header_row is not None:
            read_kwargs['header'] = header_row
        self.df = pd.read_excel(excel_file_path, **read_kwargs).fillna('')
        # Prepare a combined text field used for embedding-based matching
        text_cols = ['Measurement Instrument', 'Acronym', 'Purpose', 'Target Group(s)', 'Outcome Domain']
        combined = []
        for _, row in self.df.iterrows():
            parts = [str(row.get(c, '')) for c in text_cols if row.get(c, '')]
            combined.append(' \n '.join([p for p in parts if p]))
        self.df['combined_text'] = combined

        self._embeddings = None
        self._embed_model = None
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            hf = os.environ.get('HF_TOKEN')
            if not hf:
                raise RuntimeError('HF_TOKEN not set in environment; cannot call Hugging Face APIs')
            self._client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf)
        return self._client

    def _ensure_embeddings(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """Lazily load the embedding model and compute embeddings for the dataset."""
        if self._embeddings is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("sentence-transformers is required for embeddings-based search. Install with 'pip install sentence-transformers'") from e

        if self._embed_model is None:
            self._embed_model = SentenceTransformer(model_name)

        texts = self.df['combined_text'].astype(str).tolist()
        embs = self._embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # Normalize for cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        self._embeddings = embs
    
    def search(self, query, max_results=5):
        """Search for instruments matching the natural-language query using embeddings.

        This method uses a small sentence-transformers model to compute a
        semantic embedding for the query and returns the top matching rows from
        the Excel file. Results are guaranteed to be rows from the spreadsheet.
        """
        if not query or not str(query).strip():
            return []

        # Ensure dataset embeddings exist
        self._ensure_embeddings()

        # Compute embedding for the query
        qvec = self._embed_model.encode([str(query)], convert_to_numpy=True)
        qnorm = np.linalg.norm(qvec)
        if qnorm == 0:
            qnorm = 1.0
        qvec = qvec / qnorm

        # Cosine similarities
        sims = (self._embeddings @ qvec[0])
        # Get top indices
        top_idx = np.argsort(-sims)[:max_results]

        results = []
        for idx in top_idx:
            score = float(sims[idx])
            row = self.df.iloc[int(idx)]
            results.append({'instrument': row, 'similarity_score': score})
        return results

    def search_instruments(self, query, top_k=3):
        return self.search(query, max_results=top_k)

    def manual_search(self, beneficiaries=None, measure=None, validated='both', prog_level='both', top_k=10):
        """Simple manual search wrapper: constructs a short natural-language query
        from the beneficiaries/measure inputs and uses the HF-backed `search` to
        find matching instruments. Returns the same formatted dict structure the
        frontend expects: { 'query': ..., 'recommendations': [...] }
        """
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