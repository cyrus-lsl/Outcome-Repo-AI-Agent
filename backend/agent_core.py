import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib


class MeasurementInstrumentAgent:
    def __init__(self, excel_file_path, sheet_name=None, header_row=None):
        """Initialize the agent with the Excel data"""
        self.excel_file_path = excel_file_path
        self.sheet_name = sheet_name or 'Measurement Instruments'
        self.header_row = 0 if header_row is None else header_row

        if isinstance(self.excel_file_path, (str,)):
            if not os.path.exists(self.excel_file_path):
                raise FileNotFoundError(f"Excel file not found: {self.excel_file_path}")

        read_kwargs = {'sheet_name': self.sheet_name, 'header': self.header_row}
        self.df = pd.read_excel(excel_file_path, **read_kwargs)
        self.preprocess_data()
        self.setup_similarity_engine()

    def preprocess_data(self):
        """Clean and preprocess the data"""
        self.df = self.df.fillna('').astype(object)

        expected_cols = [
            'Measurement Instrument', 'Acronym', 'Outcome Domain',
            'Outcome Keywords', 'Purpose', 'Target Group(s)',
            'Cost', 'Permission to Use', 'Data Collection', 'Validated in Hong Kong',
            'No. of Questions / Statements', 'Scale', 'Scoring',
            'Download (Eng)', 'Download (Chi)', 'Citation',
            'Repository of Impact Measurement Instruments'
        ]

        for i in range(1, 4):
            expected_cols.append(f'Sample Question / Statement - {i}')

        lc_map = {c.lower(): c for c in self.df.columns}

        for col in expected_cols:
            if col not in self.df.columns:
                match = difflib.get_close_matches(col.lower(), lc_map.keys(), n=1, cutoff=0.6)
                if match:
                    matched_col = lc_map[match[0]]
                    self.df[col] = self.df[matched_col]
                else:
                    self.df[col] = ''

        self.df['combined_text'] = (
            self.df['Measurement Instrument'].astype(str) + ' ' +
            self.df['Acronym'].astype(str) + ' ' +
            self.df['Outcome Domain'].astype(str) + ' ' +
            self.df['Outcome Keywords'].astype(str) + ' ' +
            self.df['Purpose'].astype(str) + ' ' +
            self.df['Target Group(s)'].astype(str)
        )

    def setup_similarity_engine(self):
        """Set up TF-IDF for semantic search"""
        try:
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=5000
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_text'])
        except Exception:
            self.vectorizer = None
            self.tfidf_matrix = None

    def search_instruments(self, query, top_k=3):
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                instrument_data = self.df.iloc[idx]
                results.append({'instrument': instrument_data, 'similarity_score': similarities[idx]})
        return results

    def format_response(self, processed_results):
        """Format the response in a user-friendly way"""
        if isinstance(processed_results, str):
            return processed_results
        response = f"Recommendations for: \"{processed_results['query']}\"\n\n"
        for i, rec in enumerate(processed_results['recommendations'], 1):
            response += f"{i}. {rec['name']} ({rec['acronym']}) - Score: {rec['similarity_score']}\n"
            response += f"   Purpose: {rec['purpose']}\n"
            response += f"   Target Group: {rec['target_group']}\n"
            response += f"   Domain: {rec['domain']}\n\n"
        return response

    def manual_search(self, beneficiaries=None, measure=None, validated='both', prog_level='both', top_k=10):
        """
        Manual search allowing users to select criteria.

        beneficiaries: list of strings (e.g., ['youth','elderly']) - matches 'Target Group(s)'
        measure: string - free-text about what you're trying to measure (matches domain/keywords/purpose)
        validated: 'yes' | 'no' | 'both' - whether instrument is validated in Hong Kong
        prog_level: 'yes' | 'no' | 'both' - whether it is program-level metrics
        top_k: number of results to return
        """
        df = self.df.copy()

        # beneficiaries filter
        if beneficiaries:
            if isinstance(beneficiaries, str):
                beneficiaries = [b.strip() for b in beneficiaries.split(',')]
            mask = False
            for b in beneficiaries:
                mask = mask | df['Target Group(s)'].astype(str).str.contains(str(b), case=False, na=False)
            df = df[mask]

        # measure filter
        if measure and str(measure).strip():
            q = str(measure)
            cols = ['Outcome Domain', 'Outcome Keywords', 'Purpose', 'Measurement Instrument']
            submask = False
            for c in cols:
                if c in df.columns:
                    submask = submask | df[c].astype(str).str.contains(q, case=False, na=False)
            df = df[submask]

        # validated filter
        if validated in ('yes', 'no'):
            if validated == 'yes':
                df = df[df['Validated in Hong Kong'].astype(str).str.strip().ne('') & ~df['Validated in Hong Kong'].astype(str).str.lower().isin(['-', 'no'])]
            else:
                df = df[df['Validated in Hong Kong'].astype(str).str.strip().eq('') | df['Validated in Hong Kong'].astype(str).str.lower().isin(['-', 'no'])]

        # program-level filter (best-effort)
        prog_cols = [c for c in df.columns if 'program' in c.lower() or 'prog' in c.lower()]
        if prog_level in ('yes', 'no'):
            if prog_cols:
                col = prog_cols[0]
                if prog_level == 'yes':
                    df = df[df[col].astype(str).str.contains('yes|true|y', case=False, na=False)]
                else:
                    df = df[~df[col].astype(str).str.contains('yes|true|y', case=False, na=False)]
            else:
                if prog_level == 'yes':
                    df = df[df['Purpose'].astype(str).str.contains('program', case=False, na=False) | df['Outcome Keywords'].astype(str).str.contains('program', case=False, na=False)]
                else:
                    df = df[~(df['Purpose'].astype(str).str.contains('program', case=False, na=False) | df['Outcome Keywords'].astype(str).str.contains('program', case=False, na=False))]

        # ranking
        recommendations = []
        if getattr(self, 'vectorizer', None) is not None and not df.empty:
            parts = []
            if measure:
                parts.append(measure)
            if beneficiaries:
                parts.extend(beneficiaries)
            qtext = ' '.join(parts)
            if qtext:
                qvec = self.vectorizer.transform([qtext])
                subset = self.vectorizer.transform(df['combined_text'].astype(str))
                sims = cosine_similarity(qvec, subset).flatten()
                df = df.reset_index(drop=True)
                df['_sim'] = sims
                df2 = df.sort_values('_sim', ascending=False).head(top_k)
                for _, row in df2.iterrows():
                    recommendations.append({'instrument': row, 'similarity_score': float(row.get('_sim', 0.0))})
        if not recommendations:
            for _, row in df.head(top_k).iterrows():
                recommendations.append({'instrument': row, 'similarity_score': None})

        # format
        formatted = {'query': measure or (', '.join(beneficiaries) if beneficiaries else ''), 'recommendations': []}
        for r in recommendations:
            ins = r['instrument']
            formatted['recommendations'].append({
                'name': ins.get('Measurement Instrument', ''),
                'acronym': ins.get('Acronym', ''),
                'purpose': ins.get('Purpose', ''),
                'target_group': ins.get('Target Group(s)', ''),
                'domain': ins.get('Outcome Domain', ''),
                'similarity_score': round(r['similarity_score'], 3) if r['similarity_score'] is not None else None,
            })

        return formatted
