"""Postprocess completed databases; Gamma-7 definitions are imported unchanged."""
import csv
import json
import math
import sqlite3
from pathlib import Path

BOUNDARIES = {'cascading': [3500,4000,7000,8000,14000,17000,24000,28000,30000],
              'throughput_shock': [3500,4000,8000,10000],
              'energy_scarcity': [3500,4000,7000,10000],
              'faulted_recovery': [3500,4000,7000,8000,10000]}

def save_csv(path, rows):
    if rows:
        path = Path(path)
        temp = path.with_suffix(path.suffix + '.tmp')
        with temp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]

def weighted_alpha(checks, initial, start, end):
    current, last, integral = initial, start, 0.
    for row in checks:
        t = row['time']
        if t <= start:
            current = row['new_alpha']
        elif t < end:
            integral += (t - last) * current
            last, current = t, row['new_alpha']
    return (integral + (end - last) * current) / (end - start)

def extract(db, folder, scenario, initial):
    from evaluate_gamma7 import evaluation_for, expected_jobs_by_type, finish_time_expression
    folder = Path(folder)
    checks = read_jsonl(folder / 'checks.jsonl')
    telemetry = read_jsonl(folder / 'telemetry.jsonl')
    bounds = BOUNDARIES[scenario]
    horizon = bounds[-1]
    ev = evaluation_for(scenario, horizon)
    con = sqlite3.connect(db)
    cursor = con.cursor()
    finish = finish_time_expression(cursor)
    def summarize(starts, ends, filename):
        clauses = ' '.join(f'WHEN generated_at >= {a} AND generated_at < {b} THEN {i}'
                          for i, (a,b) in enumerate(zip(starts,ends)))
        end_clauses = ' '.join(f'WHEN generated_at >= {a} AND generated_at < {b} THEN {b}'
                          for a,b in zip(starts,ends))
        bucket_expr = f'CASE {clauses} ELSE -1 END'
        end_expr = f'CASE {end_clauses} ELSE 0 END'
        if filename == 'bins100.csv':
            bucket_expr = f'CAST((generated_at-{starts[0]})/100 AS INTEGER)'
            end_expr = f'({starts[0]}+100*({bucket_expr}+1))'
        query = f'''SELECT bucket, COUNT(*), SUM(rejected=1),
          SUM(executed=1 AND rejected=0 AND finished < bucket_end),
          AVG(CASE WHEN executed=1 AND rejected=0 AND finished < bucket_end THEN time_total END),
          SUM(node_uid=0 AND type IN (0,1,2) AND executed=1 AND rejected=0
              AND over_deadline=0 AND finished < bucket_end)
          FROM (SELECT *, {finish} AS finished, {bucket_expr} AS bucket,
                {end_expr} AS bucket_end FROM jobs
                WHERE generated_at >= {starts[0]} AND generated_at < {ends[-1]})
          GROUP BY bucket'''
        aggregates = {int(r[0]): r[1:] for r in cursor.execute(query)}
        rows = []
        for i, (a,b) in enumerate(zip(starts, ends)):
            total, rejected, completed, latency, timely = aggregates.get(i, (0,0,0,None,0))
            expected = sum(expected_jobs_by_type(a,b,ev.periodic_rates,ev.rate_multipliers))
            snapshots = [r for r in telemetry if a <= r['time'] < b]
            batteries = [min(w['battery_fraction'] for w in r['workers']) for r in snapshots]
            rows.append({'start': a, 'end': b, 'duration': b-a, 'job_rows': total,
                'rejected_rows': rejected or 0, 'rejection_rate': (rejected or 0)/total if total else None,
                'completed_by_window_end': completed or 0, 'throughput_per_second': (completed or 0)/(b-a),
                'avg_latency': latency, 'timely_periodic': timely or 0, 'expected_periodic': expected,
                'gamma': (timely or 0)/expected if expected else None,
                'alpha_time_mean': weighted_alpha(checks, initial, a,b),
                'min_battery_fraction_mean': sum(batteries)/len(batteries) if batteries else None,
                'min_battery_fraction_min': min(batteries) if batteries else None,
                'checks': sum(a <= r['time'] < b for r in checks),
                'alpha_changes': sum(a <= r['time'] < b and r['new_alpha'] != r['previous_alpha'] for r in checks)})
        save_csv(folder / filename, rows)
        return rows
    phases = summarize(bounds[:-1],bounds[1:],'phases.csv')
    starts = list(range(3500,horizon,100))
    summarize(starts, [min(t+100,horizon) for t in starts], 'bins100.csv')
    responses = []
    for t,end in zip(bounds[1:-1],bounds[2:]):
        following = [r for r in checks if t <= r['time'] < end]
        previous = [r for r in checks if r['time'] < t]
        previous = previous[-1] if previous else None
        first = following[0] if following else None
        moved = next((r for r in following if r['applied_delta'] != 0),None)
        attempted = next((r for r in following if abs(r['requested_delta']) > 1e-9),None)
        reason_changed = next((r for r in following if previous and
            (r['native_load_delta'],r['native_health_delta']) !=
            (previous['native_load_delta'],previous['native_health_delta'])),None)
        base_alpha = previous['new_alpha'] if previous else initial
        substantial = next((r for r in following if abs(r['new_alpha']-base_alpha) >= .05-1e-9),None)
        responses.append({'event_time': t, 'phase_end': end, 'observation_horizon': end-t,
             'first_check_delay_s': first['time']-t if first else None,
             'first_requested_update_delay_s': attempted['time']-t if attempted else None,
             'first_alpha_change_delay_s': moved['time']-t if moved else None,
             'alpha_change_observed': moved is not None,
             'first_feedback_regime_change_delay_s': reason_changed['time']-t if reason_changed else None,
             'alpha_shift_005_delay_s': substantial['time']-t if substantial else None,
             'alpha_before_event': base_alpha,
             'first_check_boundary_blocked': bool(first and first['requested_delta'] and not first['applied_delta']),
             'interpretation': 'Observed post-event latency; not causal attribution or QoS recovery time'})
    (folder / 'responses.json').write_text(json.dumps(responses, indent=2))
    events = [{'time': t,'type': typ,'target': target,'value': json.loads(value),'details': json.loads(details)}
              for t,typ,target,value,details in cursor.execute('SELECT * FROM scenario_events ORDER BY time')]
    (folder / 'scenario-events.json').write_text(json.dumps(events, indent=2))
    # Independent cross-check: alpha changes persisted by the scheduler match the checks.
    persisted = cursor.execute('SELECT time,new_alpha FROM dynamic_alpha_events WHERE trigger_reason != ? ORDER BY time',('initial',)).fetchall()
    traced = [(r['time'],r['new_alpha']) for r in checks if r['applied_delta'] != 0]
    assert persisted == traced, (persisted[:3], traced[:3])
    assert all(0 <= r['new_alpha'] <= 1 for r in checks)
    assert all(abs(r['new_alpha']-round(r['previous_alpha']+r['requested_delta'],4)) < 1e-8
               or r['new_alpha'] in (0.,1.) for r in checks)
    con.close()
    return {'checks': len(checks), 'telemetry_samples': len(telemetry), 'phases': phases,
            'responses': responses, 'alpha_trace_matches_database': True}
