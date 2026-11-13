#!/usr/bin/env python3
"""Multi-seed neighborhood sweep for fixed-alpha lock-in (workers only).

Default neighborhoods follow V2 Pareto recommendations ±0.05:
  throughput_shock: 0.35, 0.40, 0.45
  energy_scarcity:  0.30, 0.35, 0.40
  faulted_recovery: 0.20, 0.25, 0.30
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

from runtime_config import results_root
from scenarios import CASCADING_BATTERY_SCALING, ScenarioName

RESULTS_BASE = str(results_root())

DEFAULT_NEIGHBORHOODS = {
    ScenarioName.THROUGHPUT_SHOCK.value: [0.35, 0.40, 0.45],
    ScenarioName.ENERGY_SCARCITY.value: [0.30, 0.35, 0.40],
    ScenarioName.FAULTED_RECOVERY.value: [0.20, 0.25, 0.30],
}

DEFAULT_SEEDS = [20260715, 20260716, 20260717, 20260718, 20260719]
FULL_ALPHA_SWEEP = [round(index * 0.05, 2) for index in range(21)]


def run_one(config):
    alpha, scenario, session_id, seed, tiling_max_size, battery_scaling = config
    import run_simulation_d_sarsa_cloud

    run_simulation_d_sarsa_cloud.RANDOM_SEED = str(seed)
    run_simulation_d_sarsa_cloud.run_simulation(
        alpha,
        scenario=ScenarioName(scenario),
        session_id_prefix=session_id,
        tiling_max_size=tiling_max_size,
        logging_info=True,
        battery_scaling=battery_scaling,
    )
    return alpha, seed, session_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scenario",
        choices=[item.value for item in ScenarioName],
        nargs="?",
        default=None,
        help="One scenario, or omit with --all-scenarios",
    )
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated random seeds (default: 5 seeds)",
    )
    parser.add_argument(
        "--alphas",
        default=None,
        help="Comma-separated alphas; default is V2 recommendation ±0.05 for the scenario",
    )
    parser.add_argument("--session-prefix", default="20260721_v2ms", help="Session id prefix")
    parser.add_argument("--omit-seed-from-session-id", action="store_true",
                        help="Use only when exactly one seed is selected; keeps dataset names concise.")
    parser.add_argument("--tiling-max-size", type=int, default=65536)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--battery-scaling", type=float, default=None,
                        help="Override battery multiplier; cascading defaults to 3.6, single scenarios to 1.0.")
    parser.add_argument("--full-sweep", action="store_true",
                        help="Sweep alpha=0.00..1.00 in 0.05 steps instead of a historical neighbourhood.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the queued runs without starting simulations.")
    args = parser.parse_args()

    if not args.all_scenarios and args.scenario is None:
        raise SystemExit("provide a scenario or --all-scenarios")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise SystemExit("--seeds must list at least one seed")
    if args.omit_seed_from_session_id and len(seeds) != 1:
        raise SystemExit("--omit-seed-from-session-id requires exactly one seed")

    scenarios = (
        [item.value for item in ScenarioName]
        if args.all_scenarios
        else [args.scenario]
    )

    os.chdir(Path(__file__).resolve().parent)

    def is_already_done(session_id, alpha):
        db_path = f"{RESULTS_BASE}/{session_id}_{alpha:.2f}/log.db"
        return os.path.exists(db_path)

    configs = []
    skipped = 0
    for scenario in scenarios:
        if args.alphas:
            alphas = [round(float(x.strip()), 10) for x in args.alphas.split(",") if x.strip()]
        elif args.full_sweep:
            alphas = FULL_ALPHA_SWEEP
        else:
            if scenario not in DEFAULT_NEIGHBORHOODS:
                raise SystemExit(
                    f"scenario {scenario!r} has no historical alpha neighbourhood; "
                    "pass --full-sweep (alpha 0.00..1.00) or --alphas explicitly. "
                    "Note: --all-scenarios therefore requires --full-sweep or --alphas."
                )
            alphas = DEFAULT_NEIGHBORHOODS[scenario]
        for seed in seeds:
            for alpha in alphas:
                session_id = (
                    f"{args.session_prefix}_{scenario}_alpha"
                    if args.omit_seed_from_session_id
                    else f"{args.session_prefix}_{scenario}_s{seed}"
                )
                if is_already_done(session_id, alpha):
                    skipped += 1
                    continue
                scaling = args.battery_scaling
                if scaling is None:
                    scaling = CASCADING_BATTERY_SCALING if scenario in {
                        ScenarioName.CASCADING.value
                    } else 1.0
                configs.append((alpha, scenario, session_id, seed, args.tiling_max_size, scaling))

    print(
        f"queued {len(configs)} runs ({skipped} already done), "
        f"(scenarios={scenarios}, seeds={seeds}, workers={args.workers}, "
        f"battery_scaling={'auto' if args.battery_scaling is None else args.battery_scaling})"
    )
    if args.dry_run:
        for alpha, scenario, session_id, seed, _, battery_scaling in configs:
            print(f"would run: scenario={scenario} alpha={alpha:.2f} seed={seed} "
                  f"battery_scaling={battery_scaling} session={session_id}")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for alpha, seed, session_id in executor.map(run_one, configs):
            print(f"completed alpha={alpha:.2f} seed={seed} session={session_id}")


if __name__ == "__main__":
    main()
