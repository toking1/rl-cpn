#!/usr/bin/env python3
"""Run the three reactive dynamic-alpha strategies for the canonical cascading scenario."""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(Path(__file__).resolve().parent)

from runtime_config import results_root
from scenarios import CASCADING_BATTERY_SCALING, ScenarioName
from dynamic_alpha import DynamicAlphaPolicy
import run_simulation_d_sarsa_cloud

PREFIX = "20260721_v4da"
SEEDS = [20260715, 20260716, 20260717, 20260718, 20260719]
RESULTS_BASE = str(results_root())
SCENARIO = ScenarioName.CASCADING
BATTERY_SCALING = CASCADING_BATTERY_SCALING

# Dynamic policies to evaluate
DYNAMIC_POLICIES = [
    DynamicAlphaPolicy.REACTIVE_REJECTION,
    DynamicAlphaPolicy.REACTIVE_BATTERY,
    DynamicAlphaPolicy.REACTIVE_HYBRID,
]


def session_id_for(seed: int, policy: DynamicAlphaPolicy, prefix=None, scenario=None) -> str:
    """Build a unique session identifier, including spawn-safe run settings."""
    use_prefix = PREFIX if prefix is None else prefix
    use_scenario = SCENARIO if scenario is None else scenario
    return f"{use_prefix}_{use_scenario.value}_s{seed}_{policy.value}"


def is_done(seed: int, policy: DynamicAlphaPolicy) -> bool:
    db = f"{RESULTS_BASE}/{session_id_for(seed, policy)}_0.00/log.db"
    return os.path.exists(db)


def run_one(config):
    policy, seed, prefix, scenario_value, battery_scaling = config
    import run_simulation_d_sarsa_cloud
    run_simulation_d_sarsa_cloud.RANDOM_SEED = str(seed)
    scenario = ScenarioName(scenario_value)
    run_simulation_d_sarsa_cloud.run_simulation(
        0.0,
        scenario=scenario,
        session_id_prefix=session_id_for(seed, policy, prefix, scenario),
        tiling_max_size=65536,
        logging_info=True,
        dynamic_alpha_policy=policy,
        battery_scaling=battery_scaling,
    )
    return policy, seed


def main():
    global PREFIX, SCENARIO, BATTERY_SCALING
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--policies", default=None,
                        help="comma-separated policy names (default: all)")
    parser.add_argument("--scenario", choices=[ScenarioName.CASCADING.value],
                        default=ScenarioName.CASCADING.value)
    parser.add_argument("--battery-scaling", type=float, default=None)
    parser.add_argument("--prefix", default="20260723_v6da")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    SCENARIO = ScenarioName(args.scenario)
    BATTERY_SCALING = (
        CASCADING_BATTERY_SCALING
        if args.battery_scaling is None and SCENARIO is ScenarioName.CASCADING
        else 1.0 if args.battery_scaling is None else args.battery_scaling
    )
    PREFIX = args.prefix

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    if args.policies:
        policies = [DynamicAlphaPolicy(p.strip()) for p in args.policies.split(",") if p.strip()]
    else:
        policies = list(DYNAMIC_POLICIES)

    configs = []
    skipped = 0
    for seed in seeds:
        for policy in policies:
            if is_done(seed, policy):
                skipped += 1
                continue
            configs.append((policy, seed, args.prefix, args.scenario, BATTERY_SCALING))

    total = len(policies) * len(seeds)
    print(f"queued {len(configs)} runs ({skipped}/{total} already done), "
        f"policies={[p.value for p in policies]}, "
          f"seeds={seeds}, workers={args.workers}, scenario={SCENARIO.value}, "
          f"battery_scaling={BATTERY_SCALING}")

    if args.dry_run:
        for policy, seed, *_ in configs:
            print(f"would run: policy={policy.value} seed={seed}")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for policy, seed in executor.map(run_one, configs):
            print(f"completed policy={policy.value} seed={seed}")

    print("All dynamic-alpha runs done.")


if __name__ == "__main__":
    main()
