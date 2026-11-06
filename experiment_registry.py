from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal


DEFAULT_SEED = 20260715
FIXED_ALPHAS = tuple(round(i * 0.05, 2) for i in range(21))
SINGLE_SCENARIOS = (
    "throughput_shock",
    "energy_scarcity",
    "faulted_recovery",
)
REACTIVE_STRATEGIES = (
    "reactive_rejection",
    "reactive_battery",
    "reactive_hybrid",
)


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    title: str
    kind: Literal["simulation", "analysis"]
    runnable: bool
    runner: str
    scenarios: tuple[str, ...]
    strategies: tuple[str, ...]
    alphas: tuple[float, ...] | None
    default_seed: int
    battery_scaling: float | None
    notes: str


# runner 仅为登记字符串；空字符串表示统一入口尚未确定。
# 空元组或 None 表示没有已确认的对应参数，不据此推导归档配置。
REGISTRY: dict[str, Experiment] = {
    "single_alpha_sweep": Experiment(
        experiment_id="single_alpha_sweep",
        title="单场景固定 α 扫描",
        kind="simulation",
        runnable=True,
        runner="run_experiments:run_single_alpha_sweep",
        scenarios=SINGLE_SCENARIOS,
        strategies=("fixed",),
        alphas=FIXED_ALPHAS,
        default_seed=DEFAULT_SEED,
        battery_scaling=1.0,
        notes="对应 reproduce_all_experiments.py 的 single：三个场景各运行 21 个 α。",
    ),
    "cascading_fixed_alpha": Experiment(
        experiment_id="cascading_fixed_alpha",
        title="级联场景固定 α 扫描",
        kind="simulation",
        runnable=True,
        runner="run_experiments:run_cascading_fixed_alpha",
        scenarios=("cascading",),
        strategies=("fixed",),
        alphas=FIXED_ALPHAS,
        default_seed=DEFAULT_SEED,
        battery_scaling=3.6,
        notes="对应 reproduce_all_experiments.py 的 cascading_fixed。",
    ),
    "cascading_reactive": Experiment(
        experiment_id="cascading_reactive",
        title="级联场景反应式动态 α",
        kind="simulation",
        runnable=True,
        runner="run_experiments:run_cascading_reactive",
        scenarios=("cascading",),
        strategies=REACTIVE_STRATEGIES,
        alphas=None,
        default_seed=DEFAULT_SEED,
        battery_scaling=3.6,
        notes="对应 reproduce_all_experiments.py 的 dynamic：三个反应式策略。",
    ),
    "single_dynamic": Experiment(
        experiment_id="single_dynamic",
        title="单场景动态 α",
        kind="simulation",
        runnable=False,
        runner="",
        scenarios=SINGLE_SCENARIOS,
        strategies=REACTIVE_STRATEGIES,
        alphas=None,
        default_seed=DEFAULT_SEED,
        battery_scaling=None,
        notes=(
            "执行链路由 run_dynamic_alpha.py 与 "
            "run_repair_and_single_dynamic.py 分散承载，待统一入口后续接入。"
            "20260806_reactive_single 对应的 9 个会话从未入 git；"
            "本登记不代表这些会话数据已随仓库提供。"
        ),
    ),
    "e2_multiseed": Experiment(
        experiment_id="e2_multiseed",
        title="E2 多种子实验",
        kind="simulation",
        runnable=False,
        runner="",
        scenarios=(),
        strategies=(),
        alphas=None,
        default_seed=DEFAULT_SEED,
        battery_scaling=None,
        notes=(
            "运行能力由 run_alpha_multiseed.py 承载；"
            "结果集合标识为 paper_e2_10seeds。"
            "种子列表须按归档登记，不自行生成种子列表；"
            "default_seed 仅记录历史通用默认值，不替代归档种子列表。"
        ),
    ),
    "baseline_comparison": Experiment(
        experiment_id="baseline_comparison",
        title="基线对比实验",
        kind="simulation",
        runnable=False,
        runner="",
        scenarios=(),
        strategies=(),
        alphas=None,
        default_seed=DEFAULT_SEED,
        battery_scaling=None,
        notes="仅有归档证据，承载入口待由数据分支源码快照定位。",
    ),
    "ablation": Experiment(
        experiment_id="ablation",
        title="消融实验",
        kind="simulation",
        runnable=False,
        runner="",
        scenarios=(),
        strategies=(),
        alphas=None,
        default_seed=DEFAULT_SEED,
        battery_scaling=None,
        notes="仅有归档证据，承载入口待由数据分支源码快照定位。",
    ),
    "weight_sensitivity": Experiment(
        experiment_id="weight_sensitivity",
        title="权重敏感性分析",
        kind="analysis",
        runnable=False,
        runner="",
        scenarios=(),
        strategies=(),
        alphas=None,
        default_seed=DEFAULT_SEED,
        battery_scaling=None,
        notes=(
            "由数据分支 evaluate_weight_sensitivity.py 消费统计表执行；"
            "当前仅登记分析任务，尚未接入统一入口。"
        ),
    ),
}


def get_experiment(experiment_id: str) -> Experiment:
    """Return an experiment, raising KeyError for an unknown ID."""
    try:
        return REGISTRY[experiment_id]
    except KeyError:
        raise KeyError(f"Unknown experiment: {experiment_id!r}") from None


def list_experiments() -> list[Experiment]:
    """Return all experiments in registration order."""
    return list(REGISTRY.values())


def runnable_experiments() -> list[Experiment]:
    """Return experiments with an established execution chain."""
    return [experiment for experiment in REGISTRY.values() if experiment.runnable]


def session_name(
    experiment_id: str,
    prefix: str,
    scenario: str | None = None,
    alpha: float | None = None,
    strategy: str | None = None,
) -> str:
    """Build a session name using the existing orchestration conventions."""
    experiment = get_experiment(experiment_id)

    if experiment_id == "single_alpha_sweep":
        if scenario not in experiment.scenarios:
            raise ValueError(
                f"scenario must be one of {experiment.scenarios!r}"
            )
        if alpha is None:
            raise ValueError("alpha is required for single_alpha_sweep")
        if strategy not in (None, "fixed"):
            raise ValueError("single_alpha_sweep requires the fixed strategy")
        return f"{prefix}_single_{scenario}_alpha_{alpha:.2f}"

    if experiment_id == "cascading_fixed_alpha":
        if scenario not in (None, "cascading"):
            raise ValueError("cascading_fixed_alpha requires scenario='cascading'")
        if alpha is None:
            raise ValueError("alpha is required for cascading_fixed_alpha")
        if strategy not in (None, "fixed"):
            raise ValueError("cascading_fixed_alpha requires the fixed strategy")
        return f"{prefix}_cascading_fixed_alpha_{alpha:.2f}"

    if experiment_id == "cascading_reactive":
        if scenario not in (None, "cascading"):
            raise ValueError("cascading_reactive requires scenario='cascading'")
        if strategy not in experiment.strategies:
            raise ValueError(
                f"strategy must be one of {experiment.strategies!r}"
            )
        return f"{prefix}_cascading_dynamic_{strategy}"

    raise ValueError(
        f"Session naming is not registered for {experiment_id!r}; "
        "consult the archived execution chain."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the experiment registry.")
    parser.add_argument(
        "experiment_id",
        nargs="?",
        choices=tuple(REGISTRY),
        help="Show one experiment",
    )
    parser.add_argument(
        "--runnable",
        action="store_true",
        help="Show only runnable experiments",
    )
    args = parser.parse_args()

    if args.experiment_id is not None:
        experiments = [get_experiment(args.experiment_id)]
    else:
        experiments = list_experiments()

    for experiment in experiments:
        if args.runnable and not experiment.runnable:
            continue
        print(
            f"{experiment.experiment_id}: {experiment.title} "
            f"(kind={experiment.kind}, runnable={experiment.runnable})"
        )
        print(f"  runner: {experiment.runner or '(pending)'}")
        print(f"  notes: {experiment.notes}")


if __name__ == "__main__":
    main()
