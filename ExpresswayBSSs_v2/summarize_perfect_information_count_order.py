"""Independently check ledgers and summarize the completed count-order trial."""
import argparse
from collections import Counter
import gzip
import json
from math import fsum
from pathlib import Path
import re

from run_perfect_information import digest, read_json, write_json


def search_progress(text, checkpoints=(600, 1800, 3600)):
    number = r'[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?'
    pattern = re.compile(r'^\s*[*+]?\s*(\d+)\s+(\d+)\s+\S+\s+(\d+)\s+('
        + number + r')\s+(' + number + r')\s+(' + number + r')%\s+('
        + number + r')s\s*$', re.MULTILINE)
    rows = [dict(nodes=int(m[1]), active_nodes=int(m[2]), integer_infeasibilities=int(m[3]),
                 upper_bound=float(m[4]), profit=float(m[5]), relative_gap=float(m[6])/100.,
                 logged_seconds=float(m[7])) for m in pattern.finditer(text)]
    sampled = {}
    for time in checkpoints:
        eligible = [row for row in rows if row['logged_seconds'] <= time]
        sampled[str(time)] = eligible[-1] if eligible else None
    root = next((r for r in rows if r['nodes'] == 0 and r['integer_infeasibilities'] > 0), None)
    branch = next((r for r in rows if r['nodes'] > 1 or (r['nodes'] >= 1 and r['active_nodes'] > 1)), None)
    return dict(checkpoints=sampled, first_reported_root_relaxation=root,
                first_reported_branch_search=branch,
                interpretation='Rounded solver-log values; checkpoints use the last report at or before the requested time. Absence of branch evidence does not prove no branching occurred.')


def inspect(root, name):
    case = root / name
    summary = read_json(case / 'result.json')
    text = (case / 'solver.log').read_text(encoding='utf-8')
    nodes = re.search(r'Solve node\s*:\s*(\d+)', text)
    reduced = re.search(r'The presolved problem has:\s+(\d+) rows, (\d+) columns.*?\n\s+(\d+) binaries(?: and (\d+) integers)?', text)
    result = dict(status=summary['status'], solve_seconds=summary['solve_seconds'],
                  build_seconds=summary['build_seconds'], diagnostics=summary['diagnostics'],
                  nodes=int(nodes[1]) if nodes else None,
                  presolved=dict(rows=int(reduced[1]), columns=int(reduced[2]), binaries=int(reduced[3]),
                                 other_integers=int(reduced[4] or 0)) if reduced else None)
    if name.endswith('_mip'):
        result['search_progress'] = search_progress(text)
    if summary.get('has_verified_incumbent'):
        replay = json.loads(gzip.decompress((case / 'replay.json.gz').read_bytes()))
        events = replay['ledger']
        kinds = Counter(e['type'] for e in events)
        users = Counter(u['status'] for u in replay['final_state']['users'].values())
        if (len(replay['final_state']['users']) != 200 or replay['final_state']['waiting']
                or users['completed']+users['failed'] != 200 or users['failed'] != kinds['reservation_failure']
                or kinds['random_service']+kinds['random_timeout'] != 200):
            raise ValueError('demand conservation failure')
        income = fsum(e.get('income_reservation', 0.)+e.get('income_random', 0.) for e in events)
        charge = fsum(e.get('charging_cost', 0.) for e in events)
        penalty = fsum(e.get('reservation_failure_cost', 0.) for e in events)
        adjustment = fsum(e.get('adjustment_cost', 0.) for e in events)
        profit = income-charge-penalty-adjustment
        residual = abs(profit-summary['incumbent_objective'])
        if residual > max(1e-4, abs(profit)*1e-8):
            raise ValueError('independent ledger does not reconcile')
        result.update(profit=summary['incumbent_objective'], replay_profit=profit,
            charging_cost=charge, income=income, failure_cost=penalty, adjustment_cost=adjustment,
            reservations_completed=users['completed'], reservation_failures=users['failed'],
            random_served=kinds['random_service'], random_timeouts=kinds['random_timeout'],
            independent_accounting_residual=residual, audit_status=replay['status'],
            best_bound=summary.get('best_bound'), relative_gap=summary.get('relative_gap'),
            bound_scope=summary['bound_scope'])
    else:
        result.update(lp_bound=summary.get('lp_bound'), timed_out='Status: Timeout' in text)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--prior-full-result', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    run, report = read_json(root / 'run.json'), read_json(root / 'report.json')
    if report['status'] != 'passed':
        raise ValueError('comparison must finish and pass its checks')
    cases = {name: inspect(root, name) for name in run['identity']['cases']}
    base = read_json(root / 'baseline_mip/result.json')
    candidate = read_json(root / 'count_mip/result.json')
    if base['initial_solution_sha256'] != candidate['initial_solution_sha256']:
        raise ValueError('the two MIPs did not use the exact same initial solution')
    prior_run = read_json(args.prior_full_result.resolve().parents[2] / 'run.json')
    prior = read_json(args.prior_full_result)
    if (prior_run['identity']['inputs'] != run['identity']['inputs']
            or prior_run['identity']['failure_penalty'] != 200 or prior['day_id'] != 1):
        raise ValueError('previous full-model bound is for a different problem')
    bounds = [prior['best_bound']]
    for name in ['baseline_mip', 'count_mip']:
        if cases[name]['bound_scope'] != 'original_full_problem':
            raise ValueError('full-model trial returned a conditional bound')
        if cases[name]['best_bound'] is not None:
            bounds.append(cases[name]['best_bound'])
    upper = min(bounds)
    best_case = max([name for name in cases if 'profit' in cases[name]], key=lambda n: cases[n]['profit'])
    lower = cases[best_case]['profit']
    if upper + 1e-4 < lower:
        raise ValueError('original-problem bound is below verified profit')
    result = dict(status='passed', cases=cases, initial_objective=report['original_objective'],
        common_start_sha256=base['initial_solution_sha256'],
        original_problem=dict(best_verified_profit=lower, best_source=best_case, best_known_upper_bound=upper,
            combined_relative_gap=max(0., (upper-lower)/max(abs(upper), abs(lower))),
            prior_full_result=str(args.prior_full_result.resolve()), prior_full_result_sha256=digest(args.prior_full_result)),
        summarizer_sha256=digest(Path(__file__)))
    write_json(root / 'comparison.json', result)
    # Standard layout for reusing the best feasible solution in the existing runner.
    export = root / 'best_feasible'
    destination = export / 'days/day_01'
    destination.mkdir(parents=True, exist_ok=True)
    write_json(export / 'run.json', dict(run, role='verified_feasible_solution_export', selected_case=best_case))
    write_json(destination / 'solver_result.json', read_json(root / best_case / 'solver_result.json'))
    write_json(destination / 'result.json', read_json(root / best_case / 'result.json'))
    lines = []
    if 'charging_lp' in cases:
        lines.append(f"Charging LP: {cases['charging_lp']['solve_seconds']:.3f}s; cost={cases['charging_lp']['charging_cost']:.6f}")
    if 'reused_comparison' in run['identity']:
        lines.append('Validated common start and model audits reused from '+run['identity']['reused_comparison']['source'])
    for name in [n for n in cases if n != 'charging_lp']:
        v = cases[name]
        lines.append(f"{name}: {v['status']}; solve={v['solve_seconds']:.3f}s; "
                     f"profit={v.get('profit')}; bound={v.get('best_bound',v.get('lp_bound'))}; "
                     f"gap={v.get('relative_gap')}; nodes={v['nodes']}")
        if name.endswith('_mip'):
            for time, row in v['search_progress']['checkpoints'].items():
                lines.append(f'  checkpoint {time}s (last rounded log observation): {row}')
            lines.append(f"  first reported branch search: {v['search_progress']['first_reported_branch_search']}")
    lines.append(f'Best original-problem interval: [{lower:.6f}, {upper:.6f}]')
    (root / 'comparison.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
