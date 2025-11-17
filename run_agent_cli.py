#!/usr/bin/env python3
import argparse
from backend.agent_core import MeasurementInstrumentAgent


def prompt_manual_search(agent):
    print('\nManual search — answer the prompts (leave blank to skip)')
    beneficiaries = input('Who are your target beneficiaries? (comma-separated, e.g. youth, elderly)\n> ').strip()
    measure = input('What are you trying to measure?\n> ').strip()
    print('Validated in HK? [y]es / [n]o / [b]oth')
    v = input('> ').strip().lower()
    validated = 'both'
    if v in ('y', 'yes'):
        validated = 'yes'
    elif v in ('n', 'no'):
        validated = 'no'
    print('Program-level metrics? [y]es / [n]o / [b]oth')
    p = input('> ').strip().lower()
    prog_level = 'both'
    if p in ('y', 'yes'):
        prog_level = 'yes'
    elif p in ('n', 'no'):
        prog_level = 'no'

    beneficiaries_arg = beneficiaries if beneficiaries else None
    measure_arg = measure if measure else None

    results = agent.manual_search(beneficiaries=beneficiaries_arg, measure=measure_arg, validated=validated, prog_level=prog_level)
    print('\n' + agent.format_response(results))


def main():
    parser = argparse.ArgumentParser(description='Run MeasurementInstrumentAgent in interactive CLI mode')
    parser.add_argument('excel_path', nargs='?', default='measurement_instruments.xlsx')
    parser.add_argument('--sheet', default='Measurement Instruments')
    args = parser.parse_args()

    agent = MeasurementInstrumentAgent(args.excel_path, sheet_name=args.sheet)
    print('Interactive CLI mode. Type your query and press Enter. Type "quit" to exit.')
    print('Type "manual" to run the manual search form.')
    while True:
        q = input('\nQuery> ')
        if q.strip().lower() in ('quit', 'exit', 'q'):
            break
        if q.strip().lower() == 'manual':
            prompt_manual_search(agent)
            continue
        res = agent.search_instruments(q) if hasattr(agent, 'search_instruments') else agent.process_query(q)
        # adapt output if search_instruments was used (returns list)
        if isinstance(res, list):
            # wrap to processed format
            processed = {'query': q, 'recommendations': []}
            for r in res:
                ins = r['instrument']
                processed['recommendations'].append({
                    'name': ins.get('Measurement Instrument', ''),
                    'acronym': ins.get('Acronym', ''),
                    'purpose': ins.get('Purpose', ''),
                    'target_group': ins.get('Target Group(s)', ''),
                    'domain': ins.get('Outcome Domain', ''),
                    'similarity_score': round(r.get('similarity_score', 0), 3) if r.get('similarity_score') is not None else None,
                })
            print(agent.format_response(processed))
        else:
            print(agent.format_response(res))


if __name__ == '__main__':
    main()
