#!/usr/bin/env python3
"""Unified entry point for the current Gamma-7 experiment set.

The default workflow runs single-scenario alpha sweeps, validates their
databases, runs cascading fixed and reactive experiments, validates their
databases, and exports statistics.

Workers default to 1 and can be overridden with --workers (1..30).
--plot-only validates existing databases and exports statistics without
simulation. Plotting is handled by plot_all_figures.py in the data branch.
--experiment and --scenario filter registered runnable experiment tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import experiment_registry
from runtime_config import resolve_raw_root, resolve_repo_path
from runtime_config import results_root as resolve_results_root
from statistics_export import export_statistics


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
RESULTS_ROOT = resolve_results_root()
ALPHAS = tuple(round(index * 0.05, 2) for index in range(21))
SINGLE_SCENARIOS = ("throughput_shock", "energy_scarcity", "faulted_recovery")
REACTIVE_POLICIES = (
    "reactive_rejection",
    "reactive_battery",
    "reactive_hybrid",
)

Task = tuple[str, str, float]


def result_database(
    prefix: str, kind: str, value: str, alpha: float = 0.0
) -> Path:
    if kind == "single":
        name = f"{prefix}_single_{value}_alpha_{alpha:.2f}"
    elif kind == "cascading_fixed":
        name = f"{prefix}_cascading_fixed_alpha_{alpha:.2f}"
    elif kind == "dynamic":
        name = f"{prefix}_cascading_dynamic_{value}"
    else:
        raise ValueError(kind)
    return RESULTS_ROOT / name / "log.db"


def task_database(task: Task, prefix: str) -> Path:
    return result_database(prefix, *task)


def is_valid_database(database: Path) -> bool:
    """Return true only for a materialized SQLite result, not an LFS pointer."""
    if not database.is_file() or database.stat().st_size < 1024:
        return False
    with database.open("rb") as stream:
        return stream.read(16) == b"SQLite format 3\x00"


def run_one(task: Task, prefix: str, seed: int) -> Task:
    kind, value, alpha = task
    os.chdir(CODE_DIR)
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    import run_simulation_d_sarsa_cloud
    from dynamic_alpha import DynamicAlphaPolicy
    from scenarios import ScenarioName

    run_simulation_d_sarsa_cloud.RANDOM_SEED = str(seed)
    if kind == "single":
        run_simulation_d_sarsa_cloud.run_simulation(
            alpha,
            scenario=ScenarioName(value),
            session_id_prefix=f"{prefix}_single_{value}_alpha",
            tiling_max_size=65536,
            logging_info=True,
            battery_scaling=1.0,
        )
    elif kind == "cascading_fixed":
        run_simulation_d_sarsa_cloud.run_simulation(
            alpha,
            scenario=ScenarioName.CASCADING,
            session_id_prefix=f"{prefix}_cascading_fixed_alpha",
            tiling_max_size=65536,
            logging_info=True,
            battery_scaling=3.6,
        )
    else:
        policy = DynamicAlphaPolicy(value)
        raw_prefix = f"{prefix}_cascading_dynamic_{value}_raw"
        run_simulation_d_sarsa_cloud.run_simulation(
            0.0,
            scenario=ScenarioName.CASCADING,
            session_id_prefix=raw_prefix,
            tiling_max_size=65536,
            logging_info=True,
            dynamic_alpha_policy=policy,
            battery_scaling=3.6,
        )
        source = RESULTS_ROOT / f"{raw_prefix}_0.00"
        target = result_database(prefix, "dynamic", value).parent
        if target.exists():
            raise RuntimeError(f"dynamic result already exists: {target}")
        source.rename(target)
        with sqlite3.connect(target / "log.db") as connection:
            connection.execute(
                "UPDATE run_metadata SET value_json = ? WHERE key = 'session_id'",
                (json.dumps(target.name),),
            )
            connection.commit()
    return task


def run_phase(
    name: str,
    tasks: list[Task],
    prefix: str,
    seed: int,
    workers: int,
    dry_run: bool,
    replace_invalid: bool,
) -> None:
    invalid = [
        task
        for task in tasks
        if task_database(task, prefix).is_file()
        and not is_valid_database(task_database(task, prefix))
    ]
    if invalid and not replace_invalid:
        paths = "\n".join(str(task_database(task, prefix)) for task in invalid)
        raise RuntimeError(
            f"phase={name} contains invalid or unmaterialized LFS databases:\n"
            f"{paths}\nUse --replace-invalid only after confirming these "
            "results must be rerun."
        )
    if replace_invalid and not dry_run:
        for task in invalid:
            result_dir = task_database(task, prefix).parent
            shutil.rmtree(result_dir)
            print(
                f"removed-invalid-result phase={name} path={result_dir}",
                flush=True,
            )
    elif replace_invalid:
        for task in invalid:
            print(
                f"would-remove-invalid-result phase={name} "
                f"path={task_database(task, prefix).parent}",
                flush=True,
            )
    pending = [
        task for task in tasks
        if not is_valid_database(task_database(task, prefix))
    ]
    print(
        f"phase={name} queued={len(pending)} "
        f"skipped={len(tasks) - len(pending)} "
        f"invalid-replaced="
        f"{len(invalid) if replace_invalid and not dry_run else 0} "
        f"workers={workers}",
        flush=True,
    )
    if dry_run:
        for task in pending:
            print(f"would-run phase={name} task={task}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_one, task, prefix, seed) for task in pending
        ]
        for future in as_completed(futures):
            print(
                f"completed phase={name} task={future.result()}",
                flush=True,
            )


def validate_databases(tasks: list[Task], prefix: str) -> None:
    missing = [
        str(task_database(task, prefix))
        for task in tasks
        if not is_valid_database(task_database(task, prefix))
    ]
    if missing:
        raise RuntimeError("missing result databases:\n" + "\n".join(missing))


def statistics_code_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        try:
            manifest = json.loads(
                (REPO_ROOT / "metadata/release-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            commit = manifest["source_sha"]
            if (
                isinstance(commit, str)
                and len(commit) in (40, 64)
                and all(character in "0123456789abcdef" for character in commit)
            ):
                return commit
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return ""


def export_task_statistics(
    groups: list[tuple[str, list[Task]]],
    args: argparse.Namespace,
) -> dict:
    commit = statistics_code_commit()
    summary: dict = {
        "written": [],
        "ok": 0,
        "failed": 0,
        "failures": [],
        "experiments": {},
    }
    for experiment_id, tasks in groups:
        sessions = [
            task_database(task, args.prefix).parent.name for task in tasks
        ]
        try:
            result = export_statistics(
                RESULTS_ROOT,
                args.statistics_dir,
                experiment_id=experiment_id,
                batch_id=args.prefix,
                prefix=args.prefix,
                seed=args.seed,
                sessions=sessions,
                simulation_code_commit=commit,
                statistics_code_commit=commit,
            )
        except Exception as exc:
            result = {
                "written": [],
                "ok": 0,
                "failed": 1,
                "failures": [
                    {
                        "session": "",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                ],
            }
        summary["experiments"][experiment_id] = result
        summary["written"].extend(str(path) for path in result["written"])
        summary["ok"] += result["ok"]
        summary["failed"] += result["failed"]
        summary["failures"].extend(
            {"experiment_id": experiment_id, **failure}
            for failure in result["failures"]
        )
        if result["failures"]:
            print(
                f"WARNING: statistics export for {experiment_id} "
                f"reported {len(result['failures'])} failure(s); "
                "completed simulation phases remain completed.",
                file=sys.stderr,
                flush=True,
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _add_arguments(
    parser: argparse.ArgumentParser, *, suppress_defaults: bool = False
) -> None:
    def default(value: object) -> object:
        return argparse.SUPPRESS if suppress_defaults else value

    parser.add_argument(
        "--prefix",
        default=default("20260804"),
        help="Dataset date/prefix used in result directory names.",
    )
    parser.add_argument("--seed", type=int, default=default(20260715))
    parser.add_argument("--workers", type=int, default=default(1))
    parser.add_argument(
        "--experiment",
        default=default(None),
        help="Select one registered runnable experiment ID.",
    )
    parser.add_argument(
        "--scenario",
        choices=(*SINGLE_SCENARIOS, "cascading"),
        default=default(None),
        help="Filter tasks by scenario.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=default(None),
        help="Override the raw simulation data root (relative to the repository root).",
    )
    parser.add_argument(
        "--statistics-dir",
        type=Path,
        default=default(Path(os.environ.get("RL_CPN_STATISTICS_ROOT", "statistics-out"))),
        help="Statistics output directory (relative to the repository root).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=default(False)
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        default=default(False),
        help="Validate existing databases and export statistics only.",
    )
    parser.add_argument(
        "--replace-invalid",
        action="store_true",
        default=default(False),
        help=(
            "Delete known invalid result directories "
            "(for example LFS pointers) and rerun them."
        ),
    )
    parser.add_argument(
        "--list",
        dest="list_registry",
        action="store_true",
        default=default(False),
        help="Print the experiment registry and exit.",
    )


def main() -> int:
    global RESULTS_ROOT

    parser = argparse.ArgumentParser(
        description="Run Gamma-7 experiments and export standard statistics."
    )
    _add_arguments(parser)
    subparsers = parser.add_subparsers(dest="command")
    for command, help_text in (
        ("single-alpha-sweep", "Run the three single-scenario alpha sweeps."),
        ("cascading", "Run cascading fixed-alpha and reactive experiments."),
        ("all", "Run the complete experiment workflow (default)."),
        ("statistics-only", "Export statistics from existing databases."),
        ("list", "Print the experiment registry."),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        # Accept options before or after the subcommand without overwriting
        # values already parsed by the top-level parser.
        _add_arguments(subparser, suppress_defaults=True)

    args = parser.parse_args()
    if not 1 <= args.workers <= 30:
        raise SystemExit("--workers must be between 1 and 30")

    command = args.command or "all"
    if command == "list" or args.list_registry:
        if args.experiment is not None or args.scenario is not None:
            parser.error("--experiment and --scenario cannot be used with list")
        for experiment in experiment_registry.list_experiments():
            print(
                f"{experiment.experiment_id}\t"
                f"runnable={experiment.runnable}\t{experiment.title}"
            )
        return 0

    RESULTS_ROOT = resolve_results_root(args.raw_root)
    # Use an absolute raw root so spawned imports remain consistent even
    # after run_one changes its working directory.
    os.environ["RL_CPN_RAW_ROOT"] = str(resolve_raw_root(args.raw_root))
    os.environ["SIMULATION_RANDOM_SEED"] = str(args.seed)
    args.statistics_dir = resolve_repo_path(args.statistics_dir)

    single_tasks = [
        ("single", scenario, alpha)
        for scenario in SINGLE_SCENARIOS
        for alpha in ALPHAS
    ]
    cascade_tasks = [
        ("cascading_fixed", "cascading", alpha) for alpha in ALPHAS
    ]
    cascade_tasks.extend(
        ("dynamic", policy, 0.0) for policy in REACTIVE_POLICIES
    )

    phases: list[tuple[str, list[Task]]] = []
    groups: list[tuple[str, list[Task]]] = []
    if command in {"single-alpha-sweep", "all", "statistics-only"}:
        phases.append(("single-alpha-sweep", single_tasks))
        groups.append(("single_alpha_sweep", single_tasks))
    if command in {"cascading", "all", "statistics-only"}:
        phases.append(("cascading-fixed-and-reactive", cascade_tasks))
        groups.extend([
            (
                "cascading_fixed_alpha",
                [task for task in cascade_tasks if task[0] == "cascading_fixed"],
            ),
            (
                "cascading_reactive",
                [task for task in cascade_tasks if task[0] == "dynamic"],
            ),
        ])

    # Filter at the orchestration layer only; simulation and evaluation
    # parameters retain their established definitions.
    if args.experiment is not None or args.scenario is not None:
        registry = {
            experiment.experiment_id: experiment
            for experiment in experiment_registry.list_experiments()
        }
        if args.experiment is not None:
            experiment = registry.get(args.experiment)
            if experiment is None:
                parser.error(f"Unknown experiment: {args.experiment!r}")
            if not experiment.runnable:
                parser.error(
                    f"Experiment {args.experiment!r} is not runnable"
                )
            if args.experiment not in {
                experiment_id for experiment_id, _ in groups
            }:
                parser.error(
                    f"--experiment {args.experiment!r} is incompatible "
                    f"with command {command!r}"
                )
            if (
                args.scenario is not None
                and args.scenario not in experiment.scenarios
            ):
                parser.error(
                    f"--scenario {args.scenario!r} is incompatible "
                    f"with experiment {args.experiment!r}"
                )

        filtered_groups: list[tuple[str, list[Task]]] = []
        for experiment_id, tasks in groups:
            experiment = registry.get(experiment_id)
            if experiment is None or not experiment.runnable:
                continue
            if args.experiment is not None and experiment_id != args.experiment:
                continue
            if args.scenario is not None:
                if args.scenario not in experiment.scenarios:
                    continue
                tasks = [
                    task for task in tasks
                    if (
                        task[1] if task[0] == "single" else "cascading"
                    ) == args.scenario
                ]
            if tasks:
                filtered_groups.append((experiment_id, tasks))
        if not filtered_groups:
            parser.error("The experiment/scenario selection contains no tasks")

        groups = filtered_groups
        allowed_tasks = {
            task for _, tasks in groups for task in tasks
        }
        phases = [
            (name, [task for task in tasks if task in allowed_tasks])
            for name, tasks in phases
        ]
        phases = [(name, tasks) for name, tasks in phases if tasks]

    selected_tasks = [task for _, tasks in phases for task in tasks]
    skip_simulation = args.plot_only or command == "statistics-only"

    if args.dry_run:
        if skip_simulation:
            if args.plot_only:
                print(
                    f"would-validate databases={len(selected_tasks)}",
                    flush=True,
                )
        else:
            for name, tasks in phases:
                run_phase(
                    name, tasks, args.prefix, args.seed,
                    args.workers, True, args.replace_invalid,
                )
        for experiment_id, tasks in groups:
            print(
                f"would-export-statistics experiment_id={experiment_id} "
                f"sessions={len(tasks)} "
                f"output={args.statistics_dir / experiment_id / args.prefix}",
                flush=True,
            )
        return 0

    if args.plot_only:
        validate_databases(selected_tasks, args.prefix)
    elif not skip_simulation:
        completed_tasks: list[Task] = []
        for name, tasks in phases:
            run_phase(
                name, tasks, args.prefix, args.seed,
                args.workers, False, args.replace_invalid,
            )
            completed_tasks.extend(tasks)
            validate_databases(completed_tasks, args.prefix)

    # statistics-only lets the exporter report missing or invalid databases
    # individually, while --plot-only preserves mandatory pre-validation.
    summary = export_task_statistics(groups, args)
    return 2 if summary["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
