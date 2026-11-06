#!/usr/bin/env python3
"""Run the six invalid cascade repairs and nine single-scenario dynamics.

Total tasks: 6 cascading fixed-alpha repairs +
3 reactive policies x 3 single scenarios = 15 tasks.
The default pool starts all 15 tasks concurrently.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from runtime_config import results_root as _resolve_results_root


CODE_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = _resolve_results_root()
SEED = 20260715
REPAIR_ALPHAS = (0.05, 0.10, 0.25, 0.40, 0.50, 0.55)
SINGLE_SCENARIOS = ("throughput_shock", "energy_scarcity", "faulted_recovery")
REACTIVE_POLICIES = ("reactive_rejection", "reactive_battery", "reactive_hybrid")


def sqlite_result(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as stream:
        return stream.read(16) == b"SQLite format 3\x00"


def tasks():
    repair = [("cascade_fixed", "cascading", alpha) for alpha in REPAIR_ALPHAS]
    dynamic = [("single_dynamic", scenario, policy) for scenario in SINGLE_SCENARIOS for policy in REACTIVE_POLICIES]
    return repair + dynamic


def target_path(task, repair_prefix: str, dynamic_prefix: str) -> Path:
    kind, value, parameter = task
    if kind == "cascade_fixed":
        return RESULTS_ROOT / f"{repair_prefix}_cascading_fixed_alpha_{parameter:.2f}" / "log.db"
    return RESULTS_ROOT / f"{dynamic_prefix}_single_{value}_dynamic_{parameter}" / "log.db"


def run_one(task, repair_prefix: str, dynamic_prefix: str, seed: int):
    kind, value, parameter = task
    os.chdir(CODE_DIR)
    sys.path.insert(0, str(CODE_DIR))
    import run_simulation_d_sarsa_cloud
    from dynamic_alpha import DynamicAlphaPolicy
    from scenarios import ScenarioName

    run_simulation_d_sarsa_cloud.RANDOM_SEED = str(seed)
    if kind == "cascade_fixed":
        run_simulation_d_sarsa_cloud.run_simulation(parameter, scenario=ScenarioName.CASCADING, session_id_prefix=f"{repair_prefix}_cascading_fixed_alpha", tiling_max_size=65536, logging_info=True, battery_scaling=3.6)
    else:
        policy = DynamicAlphaPolicy(parameter)
        raw_prefix = f"{dynamic_prefix}_single_{value}_dynamic_{parameter}_raw"
        run_simulation_d_sarsa_cloud.run_simulation(0.0, scenario=ScenarioName(value), session_id_prefix=raw_prefix, tiling_max_size=65536, logging_info=True, dynamic_alpha_policy=policy, battery_scaling=1.0)
        source = RESULTS_ROOT / f"{raw_prefix}_0.00"
        target = target_path(task, repair_prefix, dynamic_prefix).parent
        if target.exists():
            raise RuntimeError(f"dynamic target already exists: {target}")
        source.rename(target)
        with sqlite3.connect(target / "log.db") as connection:
            connection.execute("UPDATE run_metadata SET value_json = ? WHERE key = 'session_id'", (json.dumps(target.name),))
            connection.commit()
    return task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repair-prefix", default="20260804")
    parser.add_argument("--dynamic-prefix", default="20260805")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 30:
        raise SystemExit("--workers must be between 1 and 30")

    queue = tasks()
    pending = []
    for task in queue:
        database = target_path(task, args.repair_prefix, args.dynamic_prefix)
        if task[0] == "cascade_fixed" and database.exists() and not sqlite_result(database):
            if args.dry_run:
                print(f"would-remove-invalid-result={database.parent}", flush=True)
            else:
                shutil.rmtree(database.parent)
                print(f"removed-invalid-result={database.parent}", flush=True)
        if not sqlite_result(database):
            pending.append(task)

    print(f"queued={len(pending)} total={len(queue)} workers={args.workers} seed={args.seed}", flush=True)
    for task in pending:
        print(f"pending={task}", flush=True)
    if args.dry_run:
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, task, args.repair_prefix, args.dynamic_prefix, args.seed) for task in pending]
        for future in as_completed(futures):
            print(f"completed={future.result()}", flush=True)

    missing = [task for task in queue if not sqlite_result(target_path(task, args.repair_prefix, args.dynamic_prefix))]
    if missing:
        raise RuntimeError(f"incomplete tasks: {missing}")
    print("all 15 requested experiments are complete", flush=True)


if __name__ == "__main__":
    main()
