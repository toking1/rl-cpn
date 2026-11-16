from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SCENARIOS = ("throughput_shock", "energy_scarcity", "faulted_recovery", "cascading")
STRATEGIES = {
    "replication": [f"fixed_{i / 20:.2f}" for i in range(21)] + [
        "reactive_rejection", "reactive_battery", "reactive_hybrid"
    ],
    "comparison": [
        "full", "ucb_mab", "dqn_alpha", "ppo_alpha", "lydroo", "debo",
        "greedy_place", "all_edge", "all_local"
    ],
    "ablation": [
        "full", "no_load", "no_health", "period10", "period20", "period50",
        "symmetric", "proportional", "fixed_initial", "fixed_selected",
        "no_reject_gate__w3_wired", "serial_priority__w3_wired",
        "single_scale_adaptive__w3_wired", "symmetric_same_range__w3_wired"
    ],
    "extension": [
        "full__w3_wireless", "proportional__w3_wireless",
        "fixed_selected__w3_wireless"
    ],
}
BASELINE_PARAMETERS = {
    "PBARWS_ALPHA_PARAMS": json.dumps({
        "ucb_mab": {"c": 0.5},
        "dqn_alpha": {"lr": 0.001, "eps_decay": 250},
        "ppo_alpha": {"lr": 0.0003, "rollout": 32},
    }),
    "PBARWS_PLACEMENT_PARAMS": json.dumps({
        "lydroo": {"V": 1000}, "debo": {"c": 0.25}
    }),
}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(folder: Path):
    parser = argparse.ArgumentParser(description=f"Run the {folder.name} experiment")
    parser.add_argument("--seeds", required=True, type=int, nargs="+", help="Random seeds for each replication")
    parser.add_argument("--workers", required=True, type=positive_int, help="Number of concurrent workers")
    parser.add_argument("--output", required=True, type=Path, help="Output directory under results/, relative to repository root")
    parser.add_argument("--dry-run", action="store_true", help="Show the task matrix without running")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    if args.output.is_absolute() or ".." in args.output.parts:
        parser.error("--output must be a relative repository path")
    repo = folder.parents[1]
    output = (repo / args.output).resolve()
    results = (repo / "results").resolve()
    if not output.is_relative_to(results) or output == results:
        parser.error("--output must be a subdirectory of results/")
    scenarios = SCENARIOS if folder.name != "extension" else ("cascading", "throughput_shock")
    tasks = [(seed, scenario, strategy) for seed in args.seeds
             for scenario in scenarios for strategy in STRATEGIES[folder.name]]
    print(f"{folder.name}: {len(tasks)} tasks, output {args.output}", flush=True)
    if args.dry_run:
        for seed, scenario, strategy in tasks:
            print(seed, scenario, strategy)
        return
    if os.name != "posix":
        parser.error("The experiment runner requires Linux")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "experiment_manifest.json"
    manifest = {"experiment": folder.name, "seeds": args.seeds, "scenarios": scenarios,
                "strategies": STRATEGIES[folder.name]}
    canonical = json.loads(json.dumps(manifest))
    legacy_path = output / ("\u5b9e\u9a8c\u8bbe\u7f6e.json")
    legacy_names = {
        "\u91cd\u590d\u5b9e\u9a8c": "replication",
        "\u5bf9\u7167\u5b9e\u9a8c": "comparison",
        "\u6d88\u878d\u5b9e\u9a8c": "ablation",
        "\u6269\u5c55\u5b9e\u9a8c": "extension",
    }
    for existing_path in (manifest_path, legacy_path):
        if existing_path.exists():
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            legacy_id = existing.pop("\u5b9e\u9a8c", None)
            if legacy_id is not None:
                existing["experiment"] = legacy_names.get(legacy_id, legacy_id)
            if existing != canonical:
                parser.error("Output directory has different experiment settings")
    manifest_path.write_text(json.dumps(canonical, indent=2), encoding="utf-8")
    for path in ("analysis/runs", "analysis/smoke", "analysis/trajectories", "logs", "archives"):
        (output / path).mkdir(parents=True, exist_ok=True)
    if (folder / "references").exists():
        shutil.copytree(folder / "references", output / "analysis/references", dirs_exist_ok=True)
    env = os.environ.copy()
    env.update(BASELINE_PARAMETERS)
    env.update({"RL_EXPERIMENT_OUTPUT": str(output), "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg"})

    def run(task):
        seed, scenario, strategy = task
        log = output / "logs" / f"{seed}_{scenario}_{strategy}.log"
        command = [sys.executable, str(folder / "experiment.py"), "worker",
                   "--seed", str(seed), "--scenario", scenario, "--strategy", strategy]
        with log.open("a", encoding="utf-8") as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Task failed {seed} {scenario} {strategy}; log {log.relative_to(repo)}")
        return task

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        for future in as_completed(futures):
            try:
                print("Completed", *future.result(), flush=True)
            except Exception as exc:
                print(exc, file=sys.stderr, flush=True)
                errors.append(str(exc))
    if errors:
        raise SystemExit(f"{len(errors)} tasks failed; inspect logs")
    os.environ["RL_EXPERIMENT_OUTPUT"] = str(output)
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("experiment_worker", folder / "experiment.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in (output / "analysis/runs").glob("*.json")]
    if len(records) != len(tasks):
        raise SystemExit(f"Found {len(records)} records for {len(tasks)} tasks")
    worker.aggregate(records)
    print("Experiment complete", args.output, flush=True)
