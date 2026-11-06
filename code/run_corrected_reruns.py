#!/usr/bin/env python3
"""Run the corrected 2026-08-04 experiment reruns with a shared worker pool."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CODE_DIR.parent / "results" / "data" / "_log" / "learning" / "D_SARSA" / "WORKERS_OR_CLOUD"
SEED = 20260715
ALPHAS = tuple(round(index * 0.05, 2) for index in range(21))
SINGLE_SCENARIOS = ("energy_scarcity", "faulted_recovery")
REACTIVE_POLICIES = ("reactive_rejection", "reactive_battery", "reactive_hybrid")


def result_dir(kind: str, value: str | float) -> Path:
    if kind == "single":
        return RESULTS_DIR / f"20260804_single_{value}_alpha"
    if kind == "cascading_fixed":
        return RESULTS_DIR / "20260804_cascading_fixed_alpha"
    return RESULTS_DIR / f"20260804_cascading_dynamic_{value}"


def database_path(kind: str, value: str | float, alpha: float) -> Path:
    if kind == "dynamic":
        return result_dir(kind, value) / "log.db"
    return Path(f"{result_dir(kind, value)}_{alpha:.2f}") / "log.db"


def build_tasks():
    tasks = []
    for scenario in SINGLE_SCENARIOS:
        tasks.extend(("single", scenario, alpha) for alpha in ALPHAS)
    tasks.extend(("cascading_fixed", "cascading", alpha) for alpha in ALPHAS)
    tasks.extend(("dynamic", policy, 0.0) for policy in REACTIVE_POLICIES)
    return tasks


def run_one(task: tuple[str, str, float]) -> tuple[str, str, float]:
    kind, value, alpha = task
    os.chdir(CODE_DIR)
    sys.path.insert(0, str(CODE_DIR))
    import run_simulation_d_sarsa_cloud
    from dynamic_alpha import DynamicAlphaPolicy
    from scenarios import ScenarioName

    run_simulation_d_sarsa_cloud.RANDOM_SEED = str(SEED)
    if kind == "single":
        run_simulation_d_sarsa_cloud.run_simulation(
            alpha,
            scenario=ScenarioName(value),
            session_id_prefix=result_dir(kind, value).name,
            tiling_max_size=65536,
            logging_info=True,
            battery_scaling=1.0,
        )
    elif kind == "cascading_fixed":
        run_simulation_d_sarsa_cloud.run_simulation(
            alpha,
            scenario=ScenarioName.CASCADING,
            session_id_prefix=result_dir(kind, value).name,
            tiling_max_size=65536,
            logging_info=True,
            battery_scaling=3.6,
        )
    else:
        policy = DynamicAlphaPolicy(value)
        raw_prefix = f"{result_dir(kind, value).name}_raw"
        run_simulation_d_sarsa_cloud.run_simulation(
            0.0,
            scenario=ScenarioName.CASCADING,
            session_id_prefix=raw_prefix,
            tiling_max_size=65536,
            logging_info=True,
            dynamic_alpha_policy=policy,
            battery_scaling=3.6,
        )
        source = RESULTS_DIR / f"{raw_prefix}_0.00"
        target = result_dir(kind, value)
        if target.exists():
            raise RuntimeError(f"dynamic target already exists: {target}")
        source.rename(target)
        with sqlite3.connect(target / "log.db") as connection:
            connection.execute(
                "UPDATE run_metadata SET value_json = ? WHERE key = 'session_id'",
                (json.dumps(target.name),),
            )
            connection.commit()
    return task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 30:
        raise SystemExit("--workers must be between 1 and 30")

    pending = [task for task in build_tasks() if not database_path(*task).is_file()]
    print(f"queued={len(pending)} workers={args.workers} seed={SEED}", flush=True)
    for kind, value, alpha in pending:
        print(f"pending kind={kind} value={value} alpha={alpha:.2f}", flush=True)
    if args.dry_run:
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, task) for task in pending]
        for future in as_completed(futures):
            kind, value, alpha = future.result()
            print(f"completed kind={kind} value={value} alpha={alpha:.2f}", flush=True)


if __name__ == "__main__":
    main()
