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
SEEDS = list(range(20260715, 20260720))
STRATEGIES = [f'fixed_{i / 20:.2f}' for i in range(21)] + ['reactive_rejection', 'reactive_battery', 'reactive_hybrid']

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
    return f'{"smoke" if smoke else "rerun20261014"}_s{seed}_{scenario}_{strategy}'

def validate(db, seed, scenario, strategy, horizon):
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as con:
        assert con.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
        metadata = {k: json.loads(v) for k, v in con.execute('SELECT key,value_json FROM run_metadata')}
        terminal = con.execute('SELECT terminal_time,stop_reason FROM simulation_terminal').fetchall()
        assert terminal == [(float(horizon), 'mission_end')], terminal
        assert int(metadata['seed']) == seed and metadata['scenario'] == scenario, metadata
        assert float(metadata['mission_end']) == horizon
        assert metadata['strategy'] == ('fixed_alpha' if strategy.startswith('fixed_') else strategy)
        if strategy.startswith('fixed_'):
            assert math.isclose(float(metadata['alpha']), float(strategy[6:]))
        count = con.execute('SELECT COUNT(*) FROM dynamic_alpha_events').fetchone()[0]
        if not strategy.startswith('fixed_'):
            assert count > 0, 'missing dynamic alpha trajectory'
        return {'terminal': terminal, 'metadata': metadata, 'dynamic_alpha_events': count}

def worker(seed, scenario, strategy, smoke=False):
    started = time.time()
    task_id = ident(seed, scenario, strategy, smoke)
    record_path = ROOT / ('analysis/smoke' if smoke else 'analysis/runs') / f'{task_id}.json'
    if record_path.exists():
        return
    os.chdir(CODE)
    sys.path.insert(0, str(CODE))
    import run_simulation_d_sarsa_cloud as sim
    from node import Node
    Node.BASE_DIR_MODELS = str(ROOT / 'models')
    from dynamic_alpha import DynamicAlphaPolicy
    from scenarios import ScenarioName
    from evaluate_gamma7 import read_metrics
    sim.RANDOM_SEED = str(seed)
    horizon = 30000 if scenario == 'cascading' else 10000
    if smoke:
        assert scenario == 'throughput_shock'
        sim.SIMULATION_TOTAL_TIME = horizon = 120
    fixed = strategy.startswith('fixed_')
    alpha = float(strategy[6:]) if fixed else 0.0
    db = DATA / f'{task_id}_{alpha:.2f}' / 'log.db'
    # A database without a completion record may be an interrupted write.
    # Reuse it only after full validation; rerun invalid databases in a new session.
    if db.exists():
        try:
            checks = validate(db, seed, scenario, strategy, horizon)
        except Exception:
            db.parent.rename(db.parent.with_name(db.parent.name + f'_interrupted_{int(time.time())}'))
    if not db.exists():
        sim.run_simulation(alpha, scenario=ScenarioName(scenario), session_id_prefix=task_id,
                           tiling_max_size=65536, logging_info=False,
                           dynamic_alpha_policy=None if fixed else DynamicAlphaPolicy(strategy),
                           battery_scaling=3.6 if scenario == 'cascading' else 1.0)
    checks = validate(db, seed, scenario, strategy, horizon)
    metrics = dataclasses.asdict(read_metrics(db, alpha, warmup=0 if smoke else 3500,
                                            scenario=scenario, mission_end=horizon))
    assert all(math.isfinite(float(metrics[k])) for k in
               ['gamma', 'total_throughput', 'rejection_rate', 'avg_latency', 'min_lifespan', 'total_service_time', 'battery_variance'])
    archive = ROOT / 'archives' / f'{task_id}.tar.gz'
    archive.parent.mkdir(parents=True, exist_ok=True)
    temp = archive.with_suffix('.partial')
    with tarfile.open(temp, 'w:gz', compresslevel=1) as tar:
        tar.add(db.parent, arcname=db.parent.name)
    temp.replace(archive)
    record = {'id': task_id, 'seed': seed, 'scenario': scenario, 'strategy': strategy,
              'status': 'completed', 'utc': utc(), 'wall_seconds': time.time() - started,
              'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              'database': str(db.relative_to(ROOT)), 'database_sha256': digest(db),
              'database_bytes': db.stat().st_size, 'archive': str(archive.relative_to(ROOT)),
              'archive_bytes': archive.stat().st_size, 'archive_sha256': digest(archive),
              'validation': checks, 'metrics': metrics}
    atomic(record_path, record)
    print(json.dumps({'completed': task_id, 'wall_seconds': record['wall_seconds']}), flush=True)

def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return 0

def aggregate(records):
    sys.path.insert(0, str(CODE))
    from evaluate_gamma7 import AlphaMetrics, build_score_reference, gamma7_score
    all_rows = []
    rankings = []
    for scenario in SCENARIOS:
        selected = [r for r in records if r['scenario'] == scenario]
        fixed = [AlphaMetrics(**r['metrics']) for r in selected if r['strategy'].startswith('fixed_')]
        if not fixed:
            continue
        reference = build_score_reference(fixed, dataset=ROOT.name)
        atomic(ROOT / 'analysis/references' / f'{scenario}.json', reference.to_dict())
        for r in selected:
            all_rows.append({'seed': r['seed'], 'scenario': scenario, 'strategy': r['strategy'],
                             **r['metrics'], 'score': gamma7_score(AlphaMetrics(**r['metrics']), reference, scenario=scenario)})
        for strategy in STRATEGIES:
            rows = [r for r in all_rows if r['scenario'] == scenario and r['strategy'] == strategy]
            if rows:
                mean = sum(r['score'] for r in rows) / len(rows)
                sd = (sum((r['score'] - mean)**2 for r in rows) / max(1, len(rows)-1))**0.5
                rankings.append({'scenario': scenario, 'strategy': strategy, 'n_seeds': len(rows), 'mean_score': mean, 'std_score': sd})
    for filename, rows in [('metrics.csv', all_rows), ('rankings.csv', sorted(rankings, key=lambda r: (r['scenario'], -r['mean_score'])))]:
        if rows:
            with (ROOT / 'analysis' / filename).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['worker'])
    p.add_argument('--seed', type=int)
    p.add_argument('--scenario', choices=SCENARIOS)
    p.add_argument('--strategy', choices=STRATEGIES)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    for name in ['analysis/runs', 'analysis/smoke', 'logs', 'archives']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    if a.mode == 'worker':
        worker(a.seed, a.scenario, a.strategy, a.smoke)
