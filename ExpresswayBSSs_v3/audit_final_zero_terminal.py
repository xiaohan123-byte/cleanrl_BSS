"""Audit the frozen 35 results without solving, then export reproducible tables.

Run from the project root with Python 3.10. Raw results and manifests are read
only; derived artifacts go to outputs/zero_terminal_real_data_seed1_v1/reports.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.experiment_control import fingerprint
from src.candidate_network import generate_candidate_network, build_user_arcs
from src.experiment_metrics import trajectory_metrics
from src.forecast import deterministic_random_requests, road_segments
from src.parameters import BusinessParameters, execution_period_limit
from src.real_data_experiment import load_inputs, make_day, rng
from src.result_statistics import build_result_statistics

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'outputs/zero_terminal_real_data_seed1_v1'
DATA = ROOT / 'data/final_real_data_v1'
REPORTS = BASE / 'reports'
SLOTS = [22, 22, 22, 18, 20, 25, 23, 20, 28, 28, 22]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def key(value):
    return value if isinstance(value, str) else ':'.join(map(str, value))


class Audit:
    def __init__(self):
        self.checked = Counter()
        self.violations = Counter()
        self.examples = defaultdict(list)

    def check(self, condition, category, detail=''):
        self.checked[category] += 1
        if not condition:
            self.violations[category] += 1
            if len(self.examples[category]) < 12:
                self.examples[category].append(str(detail))

    def close(self, actual, expected, category, detail='', tolerance=1e-6):
        self.check(math.isclose(float(actual), float(expected), rel_tol=1e-8, abs_tol=tolerance), category, detail)

    def result(self):
        return {'checks': dict(self.checked), 'violations': dict(self.violations),
                'examples': dict(self.examples), 'passed': not self.violations}


def verify_dataset(audit):
    manifest = read(DATA / 'manifest.json')
    base, means, hashes = load_inputs(ROOT / 'data')
    audit.check(hashes == manifest['source_hashes'], 'source_csv_hashes')
    for name, digest in hashes.items():
        audit.check(sha(DATA / 'sources' / name) == digest, 'archived_source_hashes', name)
    poisson = read(DATA / 'poisson_70days.json')
    audit.check(fingerprint(poisson) == manifest['poisson_dataset_hash'], 'poisson_hash')
    split = rng(301)
    selected = [int(split.choice(np.arange(w, 71, 7))) for w in range(1, 8)]
    audit.check(selected == manifest['test_day_ids_by_weekday'] == [1,37,3,4,19,13,14], 'fixed_test_split')
    audit.check(manifest['training_validation_pool'] == [d for d in range(1,71) if d not in selected], '63_day_pool')
    scenes = {}
    for day in poisson['days']:
        d = day['day_id']
        samples = rng(300, d).poisson(means[(d-1) % 7])
        audit.check(np.array_equal(samples, day['counts']), 'poisson_reproduction', d)
        scenario, plans = make_day(base, means, samples, d)
        frozen = read(DATA / 'scenarios' / f'day_{d:02d}.json')
        audit.check(fingerprint(frozen) == manifest['scenario_hashes'][f'scenarios/day_{d:02d}.json'], 'scenario_hash', d)
        audit.check(scenario.to_dict() == frozen, 'scenario_reproduction', d)
        audit.check(plans == read(DATA / 'dayahead' / f'day_{d:02d}.json'), 'dayahead_reproduction', d)
        audit.check(len(frozen['reservations']) == len(frozen['actual_random_requests']) == 200, 'daily_input_counts', d)
        if d in selected:
            scenes[d] = frozen
    return manifest, scenes


def verify_rounds(result, scenario, audit, job, round_rows):
    p = BusinessParameters.from_dict(result['parameter_snapshot'])
    actual = {v['request_id']: v for v in scenario['actual_random_requests']}
    truth = {key(v['user_key']): v for v in scenario['reservations']}
    od = {v.od_id: v for v in p.od_pairs}
    audit.check([len(s) for s in result['initial_state']['slot_soc']] == SLOTS, '250_real_slots', job)
    audit.check(all(s == 1 for row in result['initial_state']['slot_soc'] for s in row), 'initial_full_inventory', job)
    audit.check(result['method']['terminal_kind'] == 'zero' and result['run_mode'] == 'discrete_mpc_no_terminal', 'zero_terminal', job)
    round_forecast = deterministic_random_requests(p, -1, 24)
    ledger_random = {}
    reservation_arrivals = {}
    last_swap = {}
    failed = set()
    network = generate_candidate_network(p)
    arc_cache = {}
    short_actual_legs = 0
    short_route_legs = 0
    max_bound_residual = 0.
    max_station_residual = 0.
    energy_residual = 0.
    after_midnight = defaultdict(float)
    for row in result['rounds']:
        n, state, sol = row['period'], row['state_before'], row['solution']
        now = n * .25
        audit.check(row['horizon'] == p.horizon and row['forecast']['end_time'] == now + p.horizon*.25,
                    'untruncated_H', (job,n))
        expected_forecast = [r for r in round_forecast if now < r['arrival_time'] < now+p.horizon*.25]
        audit.check(row['forecast']['random_requests'] == expected_forecast, 'historical_mean_forecast_only', (job,n))
        audit.check(row['path_update_allowed'] == (n % 2 == 0), 'path_update_clock', (job,n))
        audit.check(sol['status'] in ('optimal','time_limit') and math.isfinite(sol['mip_gap']), 'feasible_solver_status', (job,n))
        audit.close(sol.get('terminal_value',0), 0, 'zero_terminal', (job,n))
        terms = sol['objective_terms']
        audit.close(sol['objective'], terms['income']-terms['charging_cost']-terms['adjustment_cost']-terms['failure_cost'],
                    'optimizer_objective', (job,n), tolerance=1e-4)
        eligible = {rid: r for rid,r in state['waiting'].items() if r['arrival_time'] <= now <= r['deadline']}
        service_events = [e for e in row['events'] if e['type'] in ('random_service','reservation_service')]
        served = {e['request_id'] for e in service_events}
        for e in service_events:
            rid = e['request_id']
            audit.check(rid in eligible, 'only_observed_service', (job,n,rid))
            for qid,q in eligible.items():
                if q['station'] != e['station'] or qid == rid:
                    continue
                current = eligible.get(rid)
                if current is None:
                    continue
                prior = (q['kind']=='reservation' and current['kind']=='random') or (
                    q['kind']==current['kind'] and q['arrival_time']<current['arrival_time'])
                audit.check(not prior or qid in served, 'reservation_priority_and_fcfs', (job,n,rid,qid))
            audit.check(0 <= now-e['arrival_time'] <= .5+1e-9, '30_minute_wait', (job,n,rid))
        adjustments = {key(e['user_key']): e for e in row['events'] if e['type']=='path_adjustment'}
        expected_adjustments = set()
        for k, route in sol['paths'].items():
            user = state['users'][k]
            o = od[user['user_key'][0]]
            direction = 1 if o.exit_km > o.entry_km else -1
            waiting = state['waiting'].get(user['waiting_request_id'])
            excluded = set(user['completed_stations']) | ({waiting['station']} if waiting else set())
            old = [i for i in (user['published_plan'] if user['entered'] else user['retained_plan'])
                   if i not in excluded and direction*(p.station.positions_km[i]-user['position_km']) >= -1e-9]
            audit.check(n % 2 == 0 or route == old, 'frozen_paths_between_updates', (job,n,k))
            if user['entered'] and user['published_plan'] is not None and route != old:
                expected_adjustments.add(k)
            audit.check(user['status']=='active' and len(route)==len(set(route)) and
                        all(i in o.station_indices and i not in excluded for i in route), 'valid_path_stations', (job,n,k))
            pos = user['position_km']
            soc = 1. if waiting else user['soc']
            anchor = pos if waiting else user['last_swap_position_km']
            if not user['entered']:
                pos = o.entry_km
                soc = max(.5,user['entry_soc']-p.entry_soc_error)
                anchor = o.entry_km
            origin = 'entry' if not user['entered'] else waiting['station'] if waiting else 'origin'
            cache_key = (o.od_id,origin,pos,soc,anchor,tuple(old))
            if cache_key not in arc_cache:
                arc_cache[cache_key] = set(build_user_arcs(network,p.od_index(o.od_id),origin,pos,soc,
                                          last_swap_position_km=anchor,protected_stations=old))
            sequence = [origin,*route,'exit']
            audit.check(all(arc in arc_cache[cache_key] for arc in zip(sequence,sequence[1:])),
                        'spacing_pruning_with_feasible_and_protected_paths',(job,n,k))
            for i in route:
                target = p.station.positions_km[i]
                distance = direction*(target-pos)
                audit.check(distance >= -1e-7 and soc-distance/300 >= -1e-7, 'path_soc_reachability', (job,n,k,i))
                short_route_legs += int(anchor is not None and direction*(target-anchor) < 100-1e-7)
                pos, soc, anchor = target, 1., target
            audit.check(soc-abs(o.exit_km-pos)/300 >= .1-1e-7, 'exit_soc_feasible', (job,n,k))
        audit.check(set(adjustments) == expected_adjustments, 'only_actual_publication_charged', (job,n))
        for k,e in adjustments.items():
            audit.check(e['old_path'] != e['new_path'] and n%2==0 and state['users'][k]['entered'],
                        'only_actual_publication_charged', (job,n,k))
        for e in row['events']:
            typ = e['type']
            if typ == 'random_arrival':
                r = actual.get(e['request_id'])
                audit.check(r is not None and e['request_id'] not in ledger_random and 0 <= e['time'] < 24,
                            'all_random_truth_once', (job,n))
                ledger_random[e['request_id']] = e
                if r:
                    for field in ('return_soc','station'):
                        audit.close(e[field], r[field], 'random_truth_match', (job,n,field))
                    audit.close(e['time'],r['arrival_time'],'random_truth_match',(job,n))
            elif typ == 'reservation_entry':
                k = key(e['user_key'])
                audit.close(e['time'],truth[k]['actual_entry_time'],'reservation_actual_entry',(job,k))
                audit.close(e['actual_entry_soc'],truth[k]['actual_entry_soc'],'reservation_actual_entry',(job,k))
                audit.check(0 <= e['time'] < 24, 'no_new_demand_after_24h', (job,k))
            elif typ == 'reservation_arrival':
                k = key(e['user_key'])
                audit.check(k not in failed, 'no_service_after_failure', (job,k))
                o = od[truth[k]['user_key'][0]]
                departed, position, soc = last_swap.get(k, (truth[k]['actual_entry_time'],o.entry_km,truth[k]['actual_entry_soc']))
                target = p.station.positions_km[e['station']]
                short_actual_legs += int(abs(target-position)<100-1e-7)
                travel = sum(abs(b-a)/75 * truth[k]['segment_time_multipliers'][idx]
                             for idx,a,b in road_segments(p,position,target))
                audit.close(e['time'],departed+travel,'actual_downstream_service_dependency',(job,n,k))
                audit.close(e['return_soc'],soc-abs(target-position)/300,'actual_vehicle_soc',(job,n,k))
                reservation_arrivals[e['request_id']] = e
            elif typ == 'reservation_service':
                k = key(e['user_key'])
                audit.check(k not in failed and e['request_id'] in reservation_arrivals, 'no_service_after_failure', (job,n,k))
                last_swap[k] = (e['time'],p.station.positions_km[e['station']],1.)
            elif typ == 'reservation_failure':
                k = key(e['user_key'])
                audit.check(k not in failed, 'one_failure_per_user', (job,n,k))
                failed.add(k)
                audit.check(e['time']==e['deadline'] and e['time'] < (n+1)*.25, 'deadline_boundary_rule', (job,n))
            elif typ == 'random_timeout':
                audit.check(e['time']==e['deadline'] and e['time'] < (n+1)*.25, 'deadline_boundary_rule', (job,n))
            elif typ == 'charging':
                i,b = e['station'],e['slot']
                residual = abs(e['power_kw']-sol['power'][i][b][0])
                max_bound_residual = max(max_bound_residual,residual)
                energy_residual = max(energy_residual,abs(e['end_soc']-e['start_soc']-.95*.25*e['power_kw']/100))
        for i in range(11):
            executed_power = sum(e['power_kw'] for e in row['events'] if e['type']=='charging' and e['station']==i)
            max_station_residual = max(max_station_residual,executed_power-960)
        if n >= 96:
            after_midnight['net_profit_yuan'] += row['reward']
        diag = sol['diagnostics']
        round_rows.append({'job_id':job,'horizon':p.horizon,'period':n,'demand_period':n<96,
                           'status':sol['status'],'gap':sol['mip_gap'],'solver_seconds':sol['solve_seconds'],
                           'model_wall_seconds':row['model_wall_seconds'],'variables':diag['model_variables'],
                           'binary_variables':diag['model_binary_variables'],'constraints':diag['model_constraints']})
    audit.check(set(ledger_random)==set(actual),'all_random_truth_once',job)
    for k,user in result['final_state']['users'].items():
        if user['status']=='completed':
            o = od[user['user_key'][0]]
            audit.check(not user['retained_plan'] and user['waiting_request_id'] is None and
                        user['soc']-abs(user['position_km']-o.exit_km)/300 >= .1-1e-7,
                        'completed_service_exit_feasible', (job,k))
        else:
            audit.check(user['status']=='failed' and k in failed, 'final_failure_coverage', (job,k))
    p_grid = np.array([e['power_kw'] for e in result['ledger'] if e['type']=='charging'])
    audit.check(np.all(p_grid>=-1e-8) and np.all(p_grid<=60+1e-7), 'executed_slot_power_bounds', job)
    return {'max_power_correction_kw':max_bound_residual,'max_station_excess_kw':max_station_residual,
            'max_soc_balance_residual':energy_residual,'cleanup_net_profit_yuan':after_midnight['net_profit_yuan'],
            'actual_legs_below_pruning_threshold':short_actual_legs,'selected_legs_below_pruning_threshold':short_route_legs}


def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    REPORTS.mkdir(exist_ok=True)
    audit = Audit()
    manifest, scenes = verify_dataset(audit)
    print('Reproduced all 70 scenarios and fixed split.', flush=True)
    plan = read(BASE/'plan.json')
    status = read(BASE/'status.json')
    audit.check(status['state']=='complete' and len(set(status['completed_jobs']))==35, '35_completed_jobs')
    audit.check(fingerprint(manifest)==plan['dataset_manifest_hash'], 'frozen_dataset_identity')
    initial_by_day = {}
    daily, rounds, provenance = [], [], []
    for job in plan['jobs']:
        job_id, d, h = job['job_id'],job['day_id'],job['horizon']
        directory = BASE/'runs'/job_id
        start_violations = sum(audit.violations.values())
        try:
            worker = read(directory/'worker_status.json')
            audit.check(worker['state']=='complete','complete_worker',job_id)
            with gzip.open(directory/'result.json.gz','rt',encoding='utf8') as stream:
                result = json.load(stream)
            p = BusinessParameters.from_dict(result['parameter_snapshot'])
            expected = BusinessParameters.from_dict(scenes[d]['params'])
            expected.horizon = h
            audit.check(p.to_dict()==expected.to_dict(),'frozen_run_parameters',job_id)
            initial = fingerprint({'state':result['initial_state'],'plan':result['dayahead_plan']})
            audit.check(initial_by_day.setdefault(d,initial)==initial,'shared_scenario_initial_state_and_paths',job_id)
            audit.check(result['dayahead_plan']==read(DATA/'dayahead'/f'day_{d:02d}.json'),'frozen_dayahead_plan',job_id)
            audit.check(p.num_periods==96 and p.interval_hours==.25 and p.max_wait_hours==.5 and
                        p.path_update_interval==2 and p.seed==1 and p.solver.time_limit_sec==120 and
                        p.solver.mip_gap==.0001 and p.solver.threads==16,'run_settings',job_id)
            statistics = build_result_statistics(result)
            audit.check(statistics==read(directory/'statistics.json'),'rebuilt_statistics_exact',job_id)
            metrics = trajectory_metrics(result,scenes[d])
            saved = read(directory/'metrics.json')
            for name,value in metrics.items():
                audit.check(value==saved[name],'rebuilt_metrics_exact',(job_id,name))
            count = 0
            with (directory/'journal/rounds.jsonl').open(encoding='utf8') as stream:
                for count,line in enumerate(stream,1):
                    audit.check(json.loads(line)==result['rounds'][count-1],'journal_equals_result',(job_id,count))
            audit.check(count==len(result['rounds']),'journal_length',job_id)
            audit.check(len(result['rounds']) < execution_period_limit(p), 'cleanup_under_safety_bound', job_id)
            checks = verify_rounds(result,scenes[d],audit,job_id,rounds)
            n = metrics['completed_periods']
            s = result['summary']
            audit.check(s['completed_reservations']+s['reservation_failures']==200 and
                        s['random_services']+s['random_timeouts']==200 and
                        not s['active_reservations'] and not s['pending_requests'],'all_400_final_outcomes',job_id)
            audit.close(s['income']-s['charging_cost']-s['adjustment_cost']-s['reservation_failure_cost'],s['total_reward'],
                        'realized_accounting',job_id)
            audit.close(s['reservation_failure_cost'],1000*s['reservation_failures'],'failure_cost_1000_once',job_id)
            audit.close(s['adjustment_cost'],10*s['path_adjustments'],'adjustment_cost_10',job_id)
            dayrow = {'job_id':job_id,'horizon':h,'weekday':job['weekday'],'day_id':d}
            dayrow.update({k:v for k,v in metrics.items() if isinstance(v,(int,float)) and not isinstance(v,bool)})
            dayrow.update(checks)
            rr = result['rounds'][:96]
            dayrow.update(demand_solver_seconds_mean=float(np.mean([r['solution']['solve_seconds'] for r in rr])),
                          demand_solver_seconds_p95=float(np.percentile([r['solution']['solve_seconds'] for r in rr],95)),
                          demand_time_limit_fraction=float(np.mean([r['solution']['status']=='time_limit' for r in rr])),
                          demand_gap_met_fraction=float(np.mean([r['solution']['mip_gap']<=.0001+1e-10 for r in rr])),
                          mean_model_binary_variables=float(np.mean([r['solution']['diagnostics']['model_binary_variables'] for r in rr])),
                          final_full_batteries=sum(abs(s-1)<=1e-7 for row in result['final_state']['slot_soc'] for s in row),
                          final_mean_soc=metrics['final_inventory_kwh']/25000)
            for prefix,kind in [('service_income','income'),('charging_cost','charging_cost'),
                                ('adjustment_cost','adjustment_cost'),('failure_cost','reservation_failure_cost')]:
                if prefix=='service_income':
                    amount = sum(e['income_reservation']+e['income_random'] for e in result['ledger'] if e['period']>=96)
                else:
                    amount = sum(e[kind] for e in result['ledger'] if e['period']>=96)
                dayrow['cleanup_'+prefix+'_yuan'] = amount
            daily.append(dayrow)
            snapshot = Path(worker['source_snapshot'])
            hashes = read(snapshot/'manifest.json')
            for name,digest in hashes.items():
                audit.check(sha(snapshot/name)==digest,'worker_source_snapshot_hashes',(job_id,name))
            changes = [name for name,digest in hashes.items() if plan['source_hashes'].get(name)!=digest]
            audit.check(set(changes)<= {'src\\atomic_io.py','src\\execution.py'},'mathematical_model_source_unchanged',job_id)
            provenance.append({'job_id':job_id,'result_sha256':sha(directory/'result.json.gz'),
                               'journal_sha256':sha(directory/'journal/rounds.jsonl'),'source_snapshot':str(snapshot),
                               'source_changes_from_original':changes,'worker_wall_seconds':saved.get('worker_wall_seconds'),
                               'runtime_timing_incomplete':saved.get('runtime_timing_incomplete'),
                               'attempt_history_count':len(list((directory/'attempt_history').glob('*.json')))})
        except Exception as exc:
            audit.check(False,'audit_exception',f'{job_id}: {type(exc).__name__}: {exc}')
        print(f'{job_id}: {sum(audit.violations.values())-start_violations} violations',flush=True)
    summary = []
    for h in (4,5,6,7,8):
        rows = [r for r in daily if r['horizon']==h]
        audit.check(len(rows)==7 and [r['day_id'] for r in rows]==manifest['test_day_ids_by_weekday'],'seven_paired_days',h)
        if not rows:
            continue
        values = {'horizon':h,'window_minutes':h*15,'days':len(rows)}
        for k in rows[0]:
            if k not in ('job_id','horizon','weekday','day_id'):
                values[k]=float(np.mean([r[k] for r in rows]))
        values['worst_mip_gap']=max(r['mip_gap_max'] for r in rows)
        values['max_cleanup_hours']=max(r['cleanup_hours'] for r in rows)
        summary.append(values)
    result_audit = {'audited_at_utc':datetime.now(timezone.utc).isoformat(),
                    'experiment_finished_at_utc':status['finished_at'],'auditor_sha256':sha(__file__),
                    'dataset_manifest_hash':fingerprint(manifest),'audit':audit.result(),
                    'job_count':len(daily),'total_rounds':len(rounds),
                    'test_days_monday_to_sunday':manifest['test_day_ids_by_weekday'],
                    'aggregation':'arithmetic mean of seven per-day metrics; demand solve statistics use 96 periods; other solver metrics include cleanup',
                    'provenance':provenance}
    write(REPORTS/'audit.json',result_audit)
    write(REPORTS/'daily_metrics.json',daily)
    write(REPORTS/'summary_by_horizon.json',summary)
    csv_write(REPORTS/'daily_metrics.csv',daily)
    csv_write(REPORTS/'summary_by_horizon.csv',summary)
    csv_write(REPORTS/'solve_rounds.csv',rounds)
    csv_write(REPORTS/'test_days.csv',[{'weekday':w,'day_id':d,'week':(d-1)//7+1} for w,d in enumerate(manifest['test_day_ids_by_weekday'],1)])
    print(json.dumps(result_audit['audit'],ensure_ascii=False),flush=True)
    return 0 if audit.result()['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
