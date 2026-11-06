#!/usr/bin/env python3
"""run_cascading_reactive_reruns.py: rerun canonical cascading reactive policies.

This script reuses the fixed-alpha 20260804 baseline and replaces only the
three stale cascading reactive-policy databases. It records the controller
source fingerprint and runtime parameters in each output database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from runtime_config import results_root as _resolve_results_root


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
RESULTS_ROOT = _resolve_results_root()
DEFAULT_SEED = 20260715
REACTIVE_POLICIES = (
    "reactive_rejection",
    "reactive_battery",
    "reactive_hybrid",
)


def parse_policies(value: str) -> tuple[str, ...]:
    policies = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(policies) - set(REACTIVE_POLICIES))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unsupported reactive policy: {', '.join(invalid)}"
        )
    if not policies:
        raise argparse.ArgumentTypeError("at least one policy is required")
    return policies


def result_dir(prefix: str, policy: str) -> Path:
    return RESULTS_ROOT / f"{prefix}_cascading_dynamic_{policy}"


def controller_provenance() -> dict[str, object]:
    controller_path = CODE_DIR / "dynamic_alpha.py"
    digest = hashlib.sha256(controller_path.read_bytes()).hexdigest()

    os.chdir(CODE_DIR)
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    import dynamic_alpha

    parameters = {
        "alpha_mid": dynamic_alpha.ALPHA_MID,
        "default_window_s": dynamic_alpha.DEFAULT_WINDOW_S,
        "high_rejection_threshold": dynamic_alpha.HIGH_REJECTION_THRESHOLD,
        "low_rejection_threshold": dynamic_alpha.LOW_REJECTION_THRESHOLD,
        "battery_critical_pct": dynamic_alpha.BATTERY_CRITICAL_PCT,
        "battery_warning_pct": dynamic_alpha.BATTERY_WARNING_PCT,
        "period_normal_s": dynamic_alpha.PERIOD_NORMAL,
        "period_warning_s": dynamic_alpha.PERIOD_WARNING,
        "period_critical_s": dynamic_alpha.PERIOD_CRITICAL,
    }
    return {
        "dynamic_alpha_sha256": digest,
        "dynamic_alpha_mtime_ns": controller_path.stat().st_mtime_ns,
        "controller_parameters": parameters,
    }


def is_sqlite(database: Path) -> bool:
    if not database.is_file() or database.stat().st_size < 1024:
        return False
    with database.open("rb") as stream:
        return stream.read(16) == b"SQLite format 3\x00"


def write_provenance(
    database: Path, provenance: dict[str, object], policy: str, seed: int, prefix: str
) -> None:
    values = {
        **provenance,
        "rerun_contract": "current_cascading_reactive_v1",
        "rerun_policy": policy,
        "rerun_seed": seed,
        "rerun_dataset_prefix": prefix,
    }
    with sqlite3.connect(database) as connection:
        for key, value in values.items():
            connection.execute(
                "INSERT OR REPLACE INTO run_metadata (key, value_json) VALUES (?, ?)",
                (key, json.dumps(value, sort_keys=True)),
            )
        connection.commit()


def run_one(
    policy: str, prefix: str, seed: int, provenance: dict[str, object]
) -> tuple[str, Path]:
    os.chdir(CODE_DIR)
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    import run_simulation_d_sarsa_cloud
    from dynamic_alpha import DynamicAlphaPolicy
    from scenarios import ScenarioName

    run_simulation_d_sarsa_cloud.RANDOM_SEED = str(seed)
    raw_prefix = f"{prefix}_cascading_dynamic_{policy}_raw"
    run_simulation_d_sarsa_cloud.run_simulation(
        0.0,
        scenario=ScenarioName.CASCADING,
        session_id_prefix=raw_prefix,
        tiling_max_size=65536,
        logging_info=True,
        dynamic_alpha_policy=DynamicAlphaPolicy(policy),
        battery_scaling=3.6,
    )

    source = RESULTS_ROOT / f"{raw_prefix}_0.00"
    target = result_dir(prefix, policy)
    if not is_sqlite(source / "log.db"):
        raise RuntimeError(f"simulation did not create a valid SQLite database: {source}")
    source.rename(target)
    write_provenance(target / "log.db", provenance, policy, seed, prefix)
    return policy, target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cascading reactive reruns for the canonical scenario."
    )
    parser.add_argument("--prefix", default="20260804")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--policies",
        type=parse_policies,
        default=REACTIVE_POLICIES,
        help="comma-separated subset of reactive policies",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete existing policy result directories before rerunning them.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.workers <= len(args.policies):
        raise SystemExit(f"workers must be between 1 and {len(args.policies)}")

    provenance = controller_provenance()
    existing = [
        result_dir(args.prefix, policy)
        for policy in args.policies
        if result_dir(args.prefix, policy).exists()
    ]
    if existing and not args.replace_existing:
        paths = "\n".join(str(path) for path in existing)
        raise SystemExit(
            "existing result directories require --replace-existing:\n" + paths
        )

    print(
        f"policies={list(args.policies)} seed={args.seed} workers={args.workers} "
        f"prefix={args.prefix} controller_sha256={provenance['dynamic_alpha_sha256']}",
        flush=True,
    )
    for policy in args.policies:
        print(f"target={result_dir(args.prefix, policy)}", flush=True)

    if args.dry_run:
        return

    for path in existing:
        shutil.rmtree(path)
        print(f"removed-stale-result={path}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, policy, args.prefix, args.seed, provenance): policy
            for policy in args.policies
        }
        for future in as_completed(futures):
            policy, target = future.result()
            print(f"completed policy={policy} target={target}", flush=True)


if __name__ == "__main__":
    main()
