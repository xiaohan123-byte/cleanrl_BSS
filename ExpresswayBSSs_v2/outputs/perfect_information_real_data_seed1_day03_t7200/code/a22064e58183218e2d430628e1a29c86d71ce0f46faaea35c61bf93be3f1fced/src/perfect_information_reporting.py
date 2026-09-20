"""Manual aggregation only; never run by the experiment controller."""
import csv
import json
from pathlib import Path

import numpy as np

from .experiment_control import atomic_json


def aggregate(records, expected_days):
    """A missing day/value invalidates its seven-day mean, not its denominator."""
    indexed = {r['day_id']: r for r in records}
    if len(indexed) != len(records) or not set(indexed) <= set(expected_days):
        raise ValueError('duplicate or unexpected day')
    full = len(expected_days) == 7 and set(indexed) == set(expected_days)

    def mean(field):
        values = [indexed[d].get(field) for d in expected_days if d in indexed]
        return float(np.mean(values)) if full and all(v is not None for v in values) else None

    flat = []
    for day in expected_days:
        record = indexed.get(day, {})
        verified = record.get('has_verified_incumbent', False)
        metrics = (record.get('audit') or {}).get('metrics', {}) if verified else {}
        flat.append(dict(day_id=day, status=record.get('status', 'missing'),
                         verified=verified, incumbent_objective=record.get('incumbent_objective') if verified else None,
                         best_bound=record.get('best_bound'), relative_gap=record.get('relative_gap'),
                         solve_seconds=record.get('solve_seconds'), build_seconds=record.get('build_seconds'),
                         reservation_failure_rate=metrics.get('reservation_failure_rate'),
                         random_service_rate=metrics.get('random_service_rate'),
                         mean_wait_minutes=metrics.get('mean_wait_minutes'),
                         path_adjustments=metrics.get('path_adjustments')))
    indexed = {r['day_id']: r for r in flat}
    keys = ['incumbent_objective', 'best_bound', 'reservation_failure_rate', 'random_service_rate',
            'mean_wait_minutes', 'path_adjustments', 'solve_seconds', 'build_seconds']
    means = {key: mean(key) for key in keys}
    times = [r['solve_seconds'] for r in flat if r['solve_seconds'] is not None]
    gaps = [r['relative_gap'] for r in flat if r['relative_gap'] is not None]
    return dict(expected_days=expected_days, present_days=sorted(set(r['day_id'] for r in records)),
                complete_seven_days=full, verified_incumbent_days=sum(r['verified'] for r in flat),
                means=means, solver_p95_seconds=float(np.percentile(times, 95)) if full and len(times) == 7 else None,
                max_relative_gap=max(gaps) if len(gaps) == 7 and full else None,
                gap_target_met_days=sum(g <= .0001 for g in gaps), rows=flat)


def report(directory):
    directory = Path(directory)
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    if (directory / 'controller.lock').exists():
        raise ValueError('controller is active or left a lock; inspect its status before final reporting')
    records = []
    for day in run['identity']['days']:
        path = directory / 'days' / f'day_{day:02d}' / 'result.json'
        status_path = path.with_name('status.json')
        if path.exists() and json.loads(status_path.read_text(encoding='utf-8'))['state'] == 'finished':
            record = json.loads(path.read_text(encoding='utf-8'))
            if record['run_id'] != run['run_id']:
                raise ValueError('mixed experiment identities')
            records.append(record)
    result = aggregate(records, run['identity']['days'])
    result['environment'] = run['environment']
    result['solver_configuration'] = run['identity']['solver']
    atomic_json(directory / 'summary.json', result)
    with (directory / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['rows'][0]))
        writer.writeheader()
        writer.writerows(result['rows'])

    def number(value, scale=1):
        return '待填' if value is None else f'{value * scale:.2f}'

    means = result['means']
    timing = number(means['solve_seconds']) + '/' + number(result['solver_p95_seconds'])
    rows = [
        '完美信息可行方案 & ' + ' & '.join([number(means['incumbent_objective']),
            number(means['reservation_failure_rate'], 100), number(means['random_service_rate'], 100),
            number(means['mean_wait_minutes']), number(means['path_adjustments']), timing]) + r' \\',
        '完美信息收益上界 & ' + number(means['best_bound']) + r' & --- & --- & --- & --- & --- \\',
    ]
    (directory / 'paper_rows.tex').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    text = ('# 完美信息实验结果\n\n'
            f"已保存日期：{len(records)}/7；通过回放的可行解：{result['verified_incumbent_days']}/7。\n\n"
            f"可行收益七日均值：{number(means['incumbent_objective'])} 元；"
            f"求解器收益上界七日均值：{number(means['best_bound'])} 元。\n\n"
            '服务指标只对应已通过物理及账务回放的可行方案。缺失日期或指标不会被剔除后求均值。'
            '求解耗时为七次全天 MILP 的统计，不是 MPC 单轮统计。\n\n'
            f"相对 gap 达标日期：{result['gap_target_met_days']}/7。计算环境与逐日残差见 summary.json 及各日 result.json。\n\n"
            '本报告由显式汇总命令生成；论文主文件需在复核后更新。\n')
    (directory / 'REPORT.md').write_text(text, encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False, indent=2))
    return result
