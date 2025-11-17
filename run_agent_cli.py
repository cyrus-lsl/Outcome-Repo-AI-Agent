#!/usr/bin/env python3
import argparse
from backend.agent_core import MeasurementInstrumentAgent


def main():
    parser = argparse.ArgumentParser(description='Instrument Search CLI')
    parser.add_argument('excel_path', nargs='?', default='measurement_instruments.xlsx')
    args = parser.parse_args()

    agent = MeasurementInstrumentAgent(args.excel_path)
    
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
            print(agent.format_response(results))


if __name__ == '__main__':
    main()