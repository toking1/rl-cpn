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
    "重复实验": [f"fixed_{i / 20:.2f}" for i in range(21)] + [
        "reactive_rejection", "reactive_battery", "reactive_hybrid"
    ],
    "对照实验": [
        "full", "ucb_mab", "dqn_alpha", "ppo_alpha", "lydroo", "debo",
        "greedy_place", "all_edge", "all_local"
    ],
    "消融实验": [
        "full", "no_load", "no_health", "period10", "period20", "period50",
        "symmetric", "proportional", "fixed_initial", "fixed_selected",
        "no_reject_gate__w3_wired", "serial_priority__w3_wired",
        "single_scale_adaptive__w3_wired", "symmetric_same_range__w3_wired"
    ],
    "扩展实验": [
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
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def main(folder: Path):
    parser = argparse.ArgumentParser(description=f"运行{folder.name}")
    parser.add_argument("--seeds", required=True, type=int, nargs="+", help="每组使用的随机种子")
    parser.add_argument("--workers", required=True, type=positive_int, help="并行进程数")
    parser.add_argument("--output", required=True, type=Path, help="输出目录，必须位于 results/ 下，相对于仓库根目录")
    parser.add_argument("--dry-run", action="store_true", help="仅显示实验矩阵")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("随机种子不能重复")
    if args.output.is_absolute() or ".." in args.output.parts:
        parser.error("--output 必须是仓库内相对路径")
    repo = folder.parents[1]
    output = (repo / args.output).resolve()
    results = (repo / "results").resolve()
    if not output.is_relative_to(results) or output == results:
        parser.error("--output 必须是 results/ 下的子目录")
    scenarios = SCENARIOS if folder.name != "扩展实验" else ("cascading", "throughput_shock")
    tasks = [(seed, scenario, strategy) for seed in args.seeds
             for scenario in scenarios for strategy in STRATEGIES[folder.name]]
    print(f"{folder.name}，任务数 {len(tasks)}，输出 {args.output}", flush=True)
    if args.dry_run:
        for seed, scenario, strategy in tasks:
            print(seed, scenario, strategy)
        return
    if os.name != "posix":
        parser.error("实验运行工具需要 Linux 环境")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "实验设置.json"
    manifest = {"实验": folder.name, "seeds": args.seeds, "scenarios": scenarios,
                "strategies": STRATEGIES[folder.name]}
    canonical = json.loads(json.dumps(manifest, ensure_ascii=False))
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != canonical:
        parser.error("输出目录已有不同的实验设置，请指定新目录")
    manifest_path.write_text(json.dumps(canonical, ensure_ascii=False, indent=2), encoding="utf-8")
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
            raise RuntimeError(f"任务失败 {seed} {scenario} {strategy}，日志 {log.relative_to(repo)}")
        return task

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        for future in as_completed(futures):
            try:
                print("完成", *future.result(), flush=True)
            except Exception as exc:
                print(exc, file=sys.stderr, flush=True)
                errors.append(str(exc))
    if errors:
        raise SystemExit(f"{len(errors)} 个任务失败，请查看日志")
    os.environ["RL_EXPERIMENT_OUTPUT"] = str(output)
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("experiment_worker", folder / "experiment.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in (output / "analysis/runs").glob("*.json")]
    if len(records) != len(tasks):
        raise SystemExit(f"记录数 {len(records)} 与任务数 {len(tasks)} 不一致")
    worker.aggregate(records)
    print("实验完成", args.output, flush=True)
