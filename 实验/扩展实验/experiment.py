"""Experiment runner; each finished run is validated and archived."""
import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sqlite3
import subprocess
import sys
import tarfile
import time

ROOT = Path(os.environ['RL_EXPERIMENT_OUTPUT']).resolve()
CODE = Path(__file__).resolve().parents[2] / 'code'
os.environ['RL_CPN_RAW_ROOT'] = str(ROOT / 'raw')
DATA = ROOT / 'raw/_log/learning/D_SARSA/WORKERS_OR_CLOUD'
SCENARIOS = ['cascading', 'throughput_shock', 'energy_scarcity', 'faulted_recovery']
SEEDS = list(range(20261020, 20261030))
from controller import CONFIGS, BEST
STRATEGIES = CONFIGS

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()

def ident(seed, scenario, strategy, smoke=False):
    return f'{"smoke" if smoke else "supp20261017"}_s{seed}_{scenario}_{strategy}'

def validate(db, seed, scenario, strategy, horizon):
    base = strategy.split('__', 1)[0]
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as con:
        assert con.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
        metadata = {k: json.loads(v) for k, v in con.execute('SELECT key,value_json FROM run_metadata')}
        terminal = con.execute('SELECT terminal_time,stop_reason FROM simulation_terminal').fetchall()
        assert terminal == [(float(horizon), 'mission_end')], terminal
        assert int(metadata['seed']) == seed and metadata['scenario'] == scenario, metadata
        assert float(metadata['mission_end']) == horizon
        assert metadata['strategy'].split('__', 1)[0] == ('fixed_alpha' if base.startswith('fixed_') else base)
        if base.startswith('fixed_'):
            expected_alpha = (BEST[scenario] if base == 'fixed_selected' else
                              .35 if base == 'fixed_initial' else float(base[6:]))
            assert math.isclose(float(metadata['alpha']), expected_alpha)
        count = con.execute('SELECT COUNT(*) FROM dynamic_alpha_events').fetchone()[0]
        if not base.startswith('fixed_'):
            assert count > 0, 'missing dynamic alpha trajectory'
        return {'terminal': terminal, 'metadata': metadata, 'dynamic_alpha_events': count}

def worker(seed, scenario, strategy, smoke=False, native=False):
    from types import SimpleNamespace
    from controller import install, BEST
    started = time.time()
    task_id = ident(seed, scenario, strategy, smoke) + ('_native' if native else '')
    record_path = ROOT / ('analysis/smoke' if smoke or native else 'analysis/runs') / f'{task_id}.json'
    if record_path.exists():
        return
    os.chdir(CODE)
    sys.path.insert(0, str(CODE))
    import run_simulation_d_sarsa_cloud as sim
    from node import Node
    Node.BASE_DIR_MODELS = str(ROOT / 'models')
    from dynamic_alpha import DynamicAlphaPolicy
    from scenarios import ScenarioName
    from evaluate_gamma7 import read_metrics, AlphaMetrics, load_score_reference, gamma7_score
    from collect import extract, read_jsonl
    sim.RANDOM_SEED = str(seed)
    horizon = 30000 if scenario == 'cascading' else 10000
    if smoke:
        assert scenario == 'throughput_shock'
        sim.SIMULATION_TOTAL_TIME = horizon = 120
    initial = BEST[scenario] if strategy.split('__', 1)[0] == 'fixed_selected' else .35
    trace = ROOT / 'traces' / task_id
    db = DATA / f'{task_id}_0.00' / 'log.db'
    assert not db.exists(), 'Existing unrecorded database requires manual validation; will not overwrite'
    close = None if native else install(sim, strategy, trace, scenario, seed=seed)
    sim.run_simulation(0., scenario=ScenarioName(scenario), session_id_prefix=task_id,
                       tiling_max_size=65536, logging_info=False,
                       dynamic_alpha_policy=DynamicAlphaPolicy.REACTIVE_HYBRID if native else SimpleNamespace(value=strategy),
                       battery_scaling=3.6 if scenario == 'cascading' else 1.0)
    control = close() if close else {}
    with sqlite3.connect(db) as con:
        assert con.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
        assert con.execute('SELECT terminal_time,stop_reason FROM simulation_terminal').fetchall() == [(float(horizon),'mission_end')]
        metadata = {k: json.loads(v) for k,v in con.execute('SELECT key,value_json FROM run_metadata')}
        assert int(metadata['seed']) == seed and metadata['scenario'] == scenario
        alpha_events = con.execute('SELECT COUNT(*) FROM dynamic_alpha_events').fetchone()[0]
        assert alpha_events > 0 or strategy.startswith('fixed_')
    metrics = dataclasses.asdict(read_metrics(db, initial, warmup=0 if smoke else 3500,
                              scenario=scenario, mission_end=horizon))
    assert all(math.isfinite(float(metrics[k])) for k in
        ['gamma','total_throughput','rejection_rate','avg_latency','min_lifespan','total_service_time','battery_variance'])
    detail = {}
    if not smoke and not native:
        detail = extract(db, trace, scenario, initial)
        # Compact diagnostics are backed up every cycle, full records in each archive.
        target = ROOT / 'analysis/trajectories' / f'{task_id}.json'
        telemetry = read_jsonl(trace / 'telemetry.jsonl')
        detail['grid_columns'] = ['time','alpha','rejection_rate','min_battery_fraction','available_workers','battery_dead_workers']
        detail['grid'] = [[r['time'],r['alpha'],r['rejection_rate'],
            min(w['battery_fraction'] for w in r['workers']),
            sum(w['available'] for w in r['workers']),sum(w['battery_dead'] for w in r['workers'])] for r in telemetry]
        atomic(target, detail)
    archive = ROOT / 'archives' / f'{task_id}.tar.gz'
    archive.parent.mkdir(parents=True, exist_ok=True)
    temp = archive.with_suffix('.partial')
    with tarfile.open(temp, 'w:gz', compresslevel=1) as tar:
        tar.add(db.parent, arcname=db.parent.name)
        if trace.exists():
            tar.add(trace, arcname='diagnostics')
    temp.replace(archive)
    reference = load_score_reference(ROOT / 'analysis/references' / f'{scenario}.json')
    record = {'id':task_id,'seed':seed,'scenario':scenario,'strategy':strategy,
        'status':'completed','utc':utc(),'wall_seconds':time.time()-started,
        'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'database':str(db.relative_to(ROOT)),'database_sha256':digest(db),'database_bytes':db.stat().st_size,
        'archive':str(archive.relative_to(ROOT)),'archive_bytes':archive.stat().st_size,'archive_sha256':digest(archive),
        'validation':{'terminal_time':horizon,'dynamic_alpha_events':alpha_events,
                      'alpha_trace_matches_database':detail.get('alpha_trace_matches_database')},
        'metrics':metrics,'score':gamma7_score(AlphaMetrics(**metrics),reference,scenario=scenario),
        'control':control}
    atomic(record_path, record)
    print(json.dumps({'completed':task_id,'wall_seconds':record['wall_seconds']}),flush=True)

def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return 0

def aggregate(records):
    from collect import save_csv
    all_rows = [{'seed':r['seed'],'scenario':r['scenario'],'strategy':r['strategy'],
                **r['metrics'],'score':r['score'],**r['control']} for r in records]
    rankings = []
    for scenario in SCENARIOS:
        for strategy in STRATEGIES:
            rows = [r for r in all_rows if r['scenario']==scenario and r['strategy']==strategy]
            if rows:
                mean = sum(r['score'] for r in rows)/len(rows)
                sd = (sum((r['score']-mean)**2 for r in rows)/max(1,len(rows)-1))**.5
                rankings.append({'scenario':scenario,'strategy':strategy,'n_seeds':len(rows),
                                 'mean_score':mean,'std_score':sd})
    save_csv(ROOT/'analysis/metrics.csv',all_rows)
    save_csv(ROOT/'analysis/rankings.csv',sorted(rankings,key=lambda r:(r['scenario'],-r['mean_score'])))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['worker'])
    p.add_argument('--seed', type=int)
    p.add_argument('--scenario', choices=SCENARIOS)
    p.add_argument('--strategy', choices=STRATEGIES)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--native', action='store_true')
    a = p.parse_args()
    for name in ['analysis/runs', 'analysis/smoke', 'logs', 'archives']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    if a.mode == 'worker':
        worker(a.seed, a.scenario, a.strategy, a.smoke, a.native)
