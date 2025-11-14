#!/usr/bin/env python3
import argparse
from backend.outcome_repo_agent import MeasurementInstrumentAgent

def main():
    parser = argparse.ArgumentParser(description='Run MeasurementInstrumentAgent in interactive CLI mode')
    parser.add_argument('excel_path', nargs='?', default='measurement_instruments.xlsx')
    parser.add_argument('--sheet', default='Measurement Instruments')
    args = parser.parse_args()

    agent = MeasurementInstrumentAgent(args.excel_path, sheet_name=args.sheet)
    print('Interactive CLI mode. Type your query and press Enter. Type "quit" to exit.')
    while True:
        q = input('Query> ')
        if q.strip().lower() in ('quit','exit','q'):
            break
        res = agent.process_query(q)
        print(agent.format_response(res))

if __name__ == '__main__':
    main()
