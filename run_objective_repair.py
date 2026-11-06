#!/usr/bin/env python3
"""Rerun the corrected objective policy and unrecoverable cascade baselines."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from runtime_config import results_root as _resolve_results_root


CODE_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = _resolve_results_root()
SEED = 20260715
OBJECTIVE_PREFIX = "20260805"
FIXED_PREFIX = "20260804"
POLICY = "objective_staged"
SINGLE_SCENARIOS = ("throughput_shock", "energy_scarcity", "faulted_recovery")
REPAIR_ALPHAS = (0.05, 0.10, 0.25, 0.40, 0.50, 0.55)


def valid_database(database: Path) -> bool:
    if not database.is_file() or database.stat().st_size < 1024:
        return False
    with database.open("rb") as stream:
        return stream.read(16) == b"SQLite format 3\x00"


def database_for(task) -> Path:
    kind, value = task
    if kind == "fixed":
        return RESULTS_ROOT / f"{FIXED_PREFIX}_cascading_fixed_alpha_{value:.2f}" / "log.db"
    if value == "cascading":
        name = f"{OBJECTIVE_PREFIX}_cascading_dynamic_{POLICY}"
    else:
        name = f"{OBJECTIVE_PREFIX}_single_{value}_dynamic_{POLICY}"
    return RESULTS_ROOT / name / "log.db"


def run_one(task, seed: int):
    kind, value = task
    os.chdir(CODE_DIR)
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    import run_simulation_d_sarsa_cloud
    from dynamic_alpha import DynamicAlphaPolicy
    from scenarios import ScenarioName

    run_simulation_d_sarsa_cloud.RANDOM_SEED = str(seed)
    if kind == "fixed":
        run_simulation_d_sarsa_cloud.run_simulation(
            value,
            scenario=ScenarioName.CASCADING,
            session_id_prefix=f"{FIXED_PREFIX}_cascading_fixed_alpha",
            tiling_max_size=65536,
            logging_info=False,
            battery_scaling=3.6,
        )
        return task

    scenario = ScenarioName(value)
    target = database_for(task).parent
    raw_prefix = f"{target.name}_raw"
    run_simulation_d_sarsa_cloud.run_simulation(
        0.0,
        scenario=scenario,
        session_id_prefix=raw_prefix,
        tiling_max_size=65536,
        logging_info=False,
        dynamic_alpha_policy=DynamicAlphaPolicy.OBJECTIVE_STAGED,
        battery_scaling=3.6 if value == "cascading" else 1.0,
    )
    source = RESULTS_ROOT / f"{raw_prefix}_0.00"
    if target.exists():
        raise RuntimeError(f"target result already exists: {target}")
    source.rename(target)
    with sqlite3.connect(target / "log.db") as connection:
        connection.execute(
            "UPDATE run_metadata SET value_json = ? WHERE key = 'session_id'",
            (json.dumps(target.name),),
        )
        connection.commit()
    return task


def main():
    parser = argparse.ArgumentParser(description="Rerun objective-staged scenarios and damaged cascade fixed alphas.")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 30:
        raise SystemExit("--workers must be between 1 and 30")

    tasks = [("objective", scenario) for scenario in (*SINGLE_SCENARIOS, "cascading")]
    tasks.extend(("fixed", alpha) for alpha in REPAIR_ALPHAS)
    pending = [task for task in tasks if not valid_database(database_for(task))]
    print(f"queued={len(pending)} skipped={len(tasks) - len(pending)} workers={args.workers}", flush=True)
    for task in pending:
        print(f"pending={task} target={database_for(task)}", flush=True)
    if args.dry_run:
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, task, args.seed) for task in pending]
        for future in as_completed(futures):
            print(f"completed={future.result()}", flush=True)

    missing = [task for task in tasks if not valid_database(database_for(task))]
    if missing:
        raise RuntimeError(f"missing result databases: {missing}")
    print("objective and cascade repair experiments complete", flush=True)


if __name__ == "__main__":
    main()
