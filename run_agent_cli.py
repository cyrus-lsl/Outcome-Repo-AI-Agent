#!/usr/bin/env python3
import argparse
from backend.agent_core import MeasurementInstrumentAgent


def main():
    parser = argparse.ArgumentParser(description='Instrument Search CLI')
    parser.add_argument('excel_path', nargs='?', default='measurement_instruments.xlsx')
    args = parser.parse_args()

    try:
        agent = MeasurementInstrumentAgent(args.excel_path, sheet_name='Measurement Instruments')
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return
    
    print("Type your search query or 'manual' for advanced search. 'quit' to exit.")
    
    while True:
        query = input("\nQuery> ").strip()
        
        if query.lower() in ('quit', 'exit'):
            break
        elif query.lower() == 'manual':
            beneficiaries = input("Beneficiaries: ").strip() or None
            measure = input("Measure: ").strip() or None
            results = agent.manual_search(beneficiaries=beneficiaries, measure=measure)
            print("\n" + agent.format_response(results))
        else:
            results = agent.search_instruments(query)
            if isinstance(results, dict) and 'recommendations' in results:
                print(agent.format_response(results))
            else:
                recs = []
                for r in results:
                    ins = r.get('instrument') if isinstance(r, dict) else None
                    if ins is None:
                        continue
                    recs.append({
                        'name': ins.get('Measurement Instrument', ''),
                        'acronym': ins.get('Acronym', ''),
                        'purpose': ins.get('Purpose', ''),
                        'target_group': ins.get('Target Group(s)', ''),
                        'domain': ins.get('Outcome Domain', ''),
                        'similarity_score': r.get('similarity_score')
                    })
                print(agent.format_results(recs))


if __name__ == '__main__':
    main()