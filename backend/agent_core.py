"""Instrument database loader and keyword-based manual search."""
import pandas as pd

from backend.utils import ensure_combined_text, keyword_search_scored, validated_in_hk


class InstrumentSearcher:
    def __init__(self, excel_file_path, sheet_name=None, header_row=None):
        read_kwargs = {}
        if sheet_name is not None:
            read_kwargs['sheet_name'] = sheet_name
        if header_row is not None:
            read_kwargs['header'] = header_row
        self.df = pd.read_excel(excel_file_path, **read_kwargs).fillna('')
        self.df = ensure_combined_text(self.df)

    def keyword_search(self, query, max_results=10, df_override=None):
        df_local = df_override if df_override is not None else self.df
        return keyword_search_scored(df_local, query, top_n=max_results)

    def manual_search(self, beneficiaries=None, measure=None, validated='both', prog_level='both', top_k=10):
        parts = []
        if measure:
            parts.append(f'Measure: {measure}')
        if beneficiaries:
            text = ', '.join(beneficiaries) if isinstance(beneficiaries, list) else str(beneficiaries)
            parts.append(f'Beneficiaries: {text}')
        query = '; '.join(parts).strip() or (measure or '')

        df_filtered = self._apply_filters(validated, prog_level)
        results = self.keyword_search(query, max_results=top_k, df_override=df_filtered)

        return {
            'query': query,
            'recommendations': [
                {
                    'name': r['instrument'].get('Measurement Instrument', ''),
                    'acronym': r['instrument'].get('Acronym', ''),
                    'purpose': r['instrument'].get('Purpose', ''),
                    'target_group': r['instrument'].get('Target Group(s)', ''),
                    'domain': r['instrument'].get('Outcome Domain', ''),
                    'programme_level': r['instrument'].get('Programme-level metric?', ''),
                    'similarity_score': r['score'],
                }
                for r in results
            ],
        }

    def _apply_filters(self, validated, prog_level):
        df = self.df
        if prog_level and prog_level != 'both':
            col = 'Programme-level metric?'
            want = str(prog_level).strip().lower()
            if want in ('yes', 'y', 'true', '1'):
                df = df[df.get(col, '').astype(str).str.strip().str.lower() == 'yes']
            elif want in ('no', 'n', 'false', '0'):
                df = df[df.get(col, '').astype(str).str.strip().str.lower() == 'no']

        if validated and validated != 'both' and 'Validated in Hong Kong' in df.columns:
            want_v = str(validated).strip().lower()
            if want_v in ('yes', 'y', 'true', '1'):
                df = df[df['Validated in Hong Kong'].apply(validated_in_hk)]
            elif want_v in ('no', 'n', 'false', '0'):
                df = df[~df['Validated in Hong Kong'].apply(validated_in_hk)]
        return df

    def format_response(self, results):
        if isinstance(results, dict) and 'recommendations' in results:
            header = f"Results for query: {results.get('query')}\n\n" if results.get('query') else ''
            return header + self.format_results(results['recommendations'])
        return self.format_results(results)

    def format_results(self, results):
        if not results:
            return 'No matching instruments found.'
        lines = [f'Found {len(results)} matching instruments:\n']
        for i, ins in enumerate(results, 1):
            line = f"{i}. {ins['name']}"
            if ins.get('acronym'):
                line += f" ({ins['acronym']})"
            line += f"\n   Purpose: {ins.get('purpose', '')}\n"
            line += f"   Target: {ins.get('target_group', '')}\n"
            line += f"   Domain: {ins.get('domain', '')}\n"
            lines.append(line)
        return '\n'.join(lines)


MeasurementInstrumentAgent = InstrumentSearcher
