"""Read-only cross-check and comparison of two completed diagnostic cases."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import re

from run_perfect_information import digest, read_json, write_json


def inspect_case(directory, original):
    result = read_json(directory / 'result.json')
    replay = json.loads(gzip.decompress((directory / 'replay.json.gz').read_bytes()))
    ledger = replay['ledger']
    events = Counter(e['type'] for e in ledger)
    counts = Counter(u['status'] for u in replay['final_state']['users'].values())
    if (len(replay['final_state']['users']) != 200 or replay['final_state']['waiting']
            or counts['completed'] + counts['failed'] != 200 or counts['active']
            or events['random_service'] + events['random_timeout'] != 200
            or events['reservation_failure'] != counts['failed']):
        raise ValueError('independent demand conservation check failed')
    fields = ['income_reservation', 'income_random', 'charging_cost',
              'adjustment_cost', 'reservation_failure_cost', 'reward_delta']
    sums = {name: math.fsum(e.get(name, 0.) for e in ledger) for name in fields}
    profit = sums['income_reservation'] + sums['income_random'] - sums['charging_cost'] \
             - sums['adjustment_cost'] - sums['reservation_failure_cost']
    residual = max(abs(profit - result['incumbent_objective']), abs(profit - sums['reward_delta']))
    if residual > max(1e-4, abs(profit) * 1e-8):
        raise ValueError('independent financial check failed')
    if result['bound_scope'] != 'restricted_subproblem_only' or result['audit']['status'] != 'passed':
        raise ValueError('invalid bound scope or replay')
    text = (directory / 'solver.log').read_text(encoding='utf-8')
    presolved = re.search(r'The presolved problem has:\s+(\d+) rows, (\d+) columns.*?\n\s+(\d+) binaries', text)
    nodes = re.search(r'Solve node\s*:\s*(\d+)', text)
    station_cost = [math.fsum(e.get('charging_cost', 0.) for e in ledger if e.get('station') == i)
                    for i in range(11)]
    improvement = profit-original
    if abs(improvement) <= max(1e-4, abs(profit) * 1e-8):
        improvement = 0.
    return dict(status=result['status'], profit=profit, improvement=improvement,
        restricted_bound=result['best_bound'], restricted_gap=result['relative_gap'],
        solve_seconds=result['solve_seconds'], build_seconds=result['build_seconds'],
        solve_nodes=int(nodes[1]) if nodes else None,
        presolved=dict(rows=int(presolved[1]), columns=int(presolved[2]), binaries=int(presolved[3])) if presolved else None,
        financial_sums=sums, charging_cost_by_station=station_cost, events=dict(events),
        completed_reservations=counts['completed'], failed_reservations=counts['failed'],
        served_random=events['random_service'], timed_out_random=events['random_timeout'],
        independent_accounting_residual=residual, replay_status=replay['status'],
        fixed_decisions_check=result['fixed_decisions_check'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--full-reference', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    report, run = read_json(output / 'report.json'), read_json(output / 'run.json')
    if report['status'] != 'passed':
        raise ValueError('both diagnostics must finish and pass replay before comparison')
    reference = read_json(args.full_reference)
    full_run = read_json(args.full_reference.resolve().parents[2] / 'run.json')
    if (full_run['identity']['inputs'] != run['identity']['inputs']
            or full_run['identity']['failure_penalty'] != 200 or reference['day_id'] != 1
            or reference.get('bound_scope', 'original_full_problem') != 'original_full_problem'):
        raise ValueError('reference bound does not describe the same unrestricted instance')
    baseline = report['original_objective']
    cases = {name: inspect_case(output / name, baseline) for name in ['routes', 'services']}
    lower, upper = max(baseline, *(v['profit'] for v in cases.values())), reference['best_bound']
    combined_file = output / 'stationwise/days/day_01/result.json'
    stationwise = None
    if combined_file.exists():
        stationwise = read_json(combined_file)
        if stationwise['audit']['status'] != 'passed' or not stationwise['has_verified_incumbent']:
            raise ValueError('unverified stationwise combination')
        lower = max(lower, stationwise['incumbent_objective'])
    if upper is None or upper + 1e-4 < lower:
        raise ValueError('missing or inconsistent original-problem bound')
    combined_gap = max(0., (upper-lower) / max(abs(upper), abs(lower)))
    comparison = dict(status='passed', original_objective=baseline, cases=cases, stationwise=stationwise,
        original_problem=dict(best_verified_feasible_profit=lower, known_full_problem_bound=upper,
            combined_relative_gap=combined_gap, bound_source=str(args.full_reference.resolve()),
            bound_source_sha256=digest(args.full_reference)),
        warning='Restricted subproblem bounds are not original-problem upper bounds.',
        summarizer_sha256=digest(Path(__file__)))
    write_json(output / 'comparison.json', comparison)
    rows = ['# 两项固定决策诊断结果', '',
        '测试日 1；seed=1；失败惩罚 200 元；每项限时 600 秒，目标 gap=1%。', '',
        f'共同初始收益：{baseline:.6f} 元。两项分别从同一个原始解出发。', '',
        '| 诊断 | 收益/元 | 改善/元 | 子问题上界/元 | 子问题 gap | 求解/s | 节点 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, label in [('routes', '固定路径'), ('services', '固定路径与服务计划')]:
        v = cases[name]
        rows.append(f"| {label} | {v['profit']:.2f} | {v['improvement']:.2f} | {v['restricted_bound']:.2f} | "
                    f"{v['restricted_gap']:.4%} | {v['solve_seconds']:.2f} | {v['solve_nodes']} |")
    rows += ['', '**表中上界和 gap 仅属于受限子问题。** 两项完整执行回放及独立财务、用户守恒核对均通过。', '']
    if stationwise:
        rows += [f'另将服务计划相同的原解与新解按站择优合并，无额外求解，得到收益 '
                 f'{stationwise["incumbent_objective"]:.2f} 元（比原解增加 {stationwise["improvement"]:.2f} 元）。'
                 '合成方案已通过完整回放，位于 `stationwise/days/day_01/solver_result.json`。', '']
    rows += [
        f'用新的最佳可行收益与此前完整模型上界 {upper:.2f} 元合并，原问题目前的已知 gap 为 '
        f'{combined_gap:.4%}。这不是重新求解完整模型获得的 gap。', '',
        '详细事件计数、费用、预处理规模和核验残差见 `comparison.json`；逐轮日志、解和电池轨迹在各子目录。', '']
    (output / 'comparison.md').write_text('\n'.join(rows), encoding='utf-8')
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
