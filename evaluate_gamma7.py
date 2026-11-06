#!/usr/bin/env python3
"""Gamma-7 evaluation: extract the seven metrics, score and rank strategies.

This is the single entry point for the Gamma-7 protocol.  It unifies the two
scripts that used to split the job between them:

* metric extraction from a monitored ``log.db`` (``read_metrics``), and
* the scenario weight matrix, the frozen reference bounds, the composite score,
  the Pareto screen and the strategy ranking.

The seven immutable objectives and their directions:

  1. gamma              (maximize): timely periodic jobs / expected periodic jobs
  2. total_throughput   (maximize)
  3. rejection_rate     (minimize)
  4. avg_latency        (minimize)
  5. min_lifespan       (maximize)
  6. total_service_time (maximize)
  7. battery_variance   (minimize)

The metric definitions here are the ones every published score, reference bound
and figure in ``results/analysis`` was derived from, so the numbers are
unchanged; the module keeps the same public names for downstream plotting code.

Two CLI modes:

* sweep (when ``--session-id`` is given) - read a fixed-alpha sweep, freeze the
  reference from it, score and rank every run, and report the best alpha.  In a
  single scenario the highest score is the best fixed alpha.
* database (when a positional ``log.db`` is given) - evaluate one run, print its
  seven metrics, optionally against a frozen reference, with ``--phase-breakdown``
  for the per-phase trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_ROOT = REPO_ROOT / "results" / "data" / "_log" / "learning" / "D_SARSA"
DEFAULT_WARMUP = 3500.0
DEFAULT_MISSION_END = 10000.0
DEFAULT_PERIODIC_RATES = (60.0, 30.0, 15.0)
SCORE_VERSION = "Gamma-7"

# All score calculations use these immutable objectives and directions.  The
# reference bounds are derived only from the fixed-alpha baseline, never from
# the strategies being compared in a later ranking.
OBJECTIVES = (
    ("gamma", True),
    ("total_throughput", True),
    ("rejection_rate", False),
    ("avg_latency", False),
    ("min_lifespan", True),
    ("total_service_time", True),
    ("battery_variance", False),
)
OBJECTIVE_NAMES = tuple(name for name, _ in OBJECTIVES)
EQUAL_WEIGHTS = {name: 1.0 / len(OBJECTIVES) for name in OBJECTIVE_NAMES}


@dataclass(frozen=True)
class ScenarioEvaluation:
    name: str
    mission_end: float
    periodic_rates: tuple[float, ...] = DEFAULT_PERIODIC_RATES
    rate_multipliers: tuple[tuple[float, float, float], ...] = ()


SCENARIO_EVALUATIONS = {
    "throughput_shock": ScenarioEvaluation(
        "throughput_shock", 10000.0, rate_multipliers=((4000.0, 8000.0, 1.5),)
    ),
    "energy_scarcity": ScenarioEvaluation("energy_scarcity", 10000.0),
    "faulted_recovery": ScenarioEvaluation("faulted_recovery", 10000.0),
    "cascading": ScenarioEvaluation(
        "cascading", 30000.0, rate_multipliers=((24000.0, 28000.0, 1.5),)
    ),
}

# The metrics remain the same seven Gamma-7 objectives, but their relative
# importance follows the operational failure mode of each single scenario.
# Cascading keeps equal weights so fixed and dynamic strategies are compared
# on the same balanced mission objective.
SCENARIO_OBJECTIVE_WEIGHTS = {
    "throughput_shock": {
        # Throughput shock is scored as a burst-handling and continuity trade-off:
        # timely periodic service and total worker service time dominate, while
        # throughput/rejection remain guards rather than the whole objective.
        "gamma": 0.43,
        "total_throughput": 0.04,
        "rejection_rate": 0.05,
        "avg_latency": 0.01,
        "min_lifespan": 0.02,
        "total_service_time": 0.44,
        "battery_variance": 0.01,
    },
    "energy_scarcity": {
        "gamma": 0.15,
        "total_throughput": 0.10,
        "rejection_rate": 0.10,
        "avg_latency": 0.05,
        "min_lifespan": 0.25,
        "total_service_time": 0.15,
        "battery_variance": 0.20,
    },
    "faulted_recovery": {
        "gamma": 0.25,
        "total_throughput": 0.15,
        "rejection_rate": 0.20,
        "avg_latency": 0.15,
        "min_lifespan": 0.15,
        "total_service_time": 0.05,
        "battery_variance": 0.05,
    },
}


@dataclass
class AlphaMetrics:
    alpha: float
    gamma: float
    total_throughput: float
    rejection_rate: float
    avg_latency: float
    min_lifespan: float
    total_service_time: float
    battery_variance: float
    gamma_by_type: list[float]
    first_death_gap: float
    max_lifespan: float
    mission_end: float
    database: str

    @property
    def qos(self) -> float:
        """Compatibility alias for historical callers; new reports use gamma."""
        return self.gamma


@dataclass(frozen=True)
class ScoreReference:
    """Frozen normalization bounds for one Gamma-7 evaluation dataset."""

    low: dict[str, float]
    high: dict[str, float]
    source: str = "fixed_alpha_baseline"
    dataset: str | None = None
    score_version: str = SCORE_VERSION

    def __post_init__(self) -> None:
        low = {name: float(self.low[name]) for name in self.low}
        high = {name: float(self.high[name]) for name in self.high}
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        missing = (set(OBJECTIVE_NAMES) - set(low)) | (set(OBJECTIVE_NAMES) - set(high))
        if missing:
            raise ValueError(f"score reference is missing objectives: {sorted(missing)}")
        if self.score_version != SCORE_VERSION:
            raise ValueError(
                f"unsupported score reference version {self.score_version!r}; "
                f"expected {SCORE_VERSION!r}"
            )
        for name in OBJECTIVE_NAMES:
            if not math.isfinite(low[name]) or not math.isfinite(high[name]):
                raise ValueError(f"score reference for {name} must be finite")
            if high[name] < low[name]:
                raise ValueError(f"score reference high < low for {name}")

    def to_dict(self) -> dict:
        return {
            "score_version": self.score_version,
            "source": self.source,
            "dataset": self.dataset,
            "low": {name: float(self.low[name]) for name in OBJECTIVE_NAMES},
            "high": {name: float(self.high[name]) for name in OBJECTIVE_NAMES},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ScoreReference":
        nested = value.get("reference")
        if isinstance(nested, Mapping):
            value = nested
        low = value.get("low")
        high = value.get("high")
        if not isinstance(low, Mapping) or not isinstance(high, Mapping):
            raise ValueError("score reference must contain low and high mappings")
        return cls(
            low={name: float(low[name]) for name in OBJECTIVE_NAMES},
            high={name: float(high[name]) for name in OBJECTIVE_NAMES},
            source=str(value.get("source", "fixed_alpha_baseline")),
            dataset=None if value.get("dataset") is None else str(value["dataset"]),
            score_version=str(value.get("score_version", SCORE_VERSION)),
        )


def weights_for(scenario: str | None) -> dict[str, float]:
    """Return the versioned immutable Gamma-7 weights for a scenario."""
    weights = dict(SCENARIO_OBJECTIVE_WEIGHTS.get(scenario or "", EQUAL_WEIGHTS))
    if set(weights) != set(OBJECTIVE_NAMES):
        raise ValueError("Gamma-7 weights do not cover all objectives")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Gamma-7 weights must sum to 1")
    return weights


def _metric_value(metrics: AlphaMetrics, name: str) -> float:
    value = float(getattr(metrics, name))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def build_score_reference(
    metrics: Sequence[AlphaMetrics],
    *,
    source: str = "fixed_alpha_baseline",
    dataset: str | None = None,
) -> ScoreReference:
    """Freeze min/max normalization bounds from a fixed-alpha baseline sweep."""
    if not metrics:
        raise ValueError("at least one fixed-alpha result is required for a score reference")
    low = {name: min(_metric_value(item, name) for item in metrics) for name in OBJECTIVE_NAMES}
    high = {name: max(_metric_value(item, name) for item in metrics) for name in OBJECTIVE_NAMES}
    return ScoreReference(low=low, high=high, source=source, dataset=dataset)


def load_score_reference(path: Path) -> ScoreReference:
    return ScoreReference.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_score_reference(reference: ScoreReference, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized_objectives(candidate: AlphaMetrics, reference: ScoreReference) -> dict[str, float]:
    """Normalize metrics with frozen bounds; never derive bounds from candidates."""
    normalized: dict[str, float] = {}
    for name, maximize in OBJECTIVES:
        value = _metric_value(candidate, name)
        low = reference.low[name]
        high = reference.high[name]
        if high == low:
            normalized[name] = 0.5
        elif maximize:
            normalized[name] = (value - low) / (high - low)
        else:
            normalized[name] = (high - value) / (high - low)
    # Gamma-7 scores are defined on [0, 1]: freeze the ruler AND bound the reading.
    return {name: min(1.0, max(0.0, value)) for name, value in normalized.items()}


def score_breakdown(
    candidate: AlphaMetrics,
    reference: ScoreReference,
    *,
    scenario: str | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict[str, object]:
    chosen_weights = dict(weights) if weights is not None else weights_for(scenario)
    if set(chosen_weights) != set(OBJECTIVE_NAMES):
        raise ValueError("Gamma-7 weights do not cover all objectives")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in chosen_weights.values()):
        raise ValueError("Gamma-7 weights must be finite and non-negative")
    if not math.isclose(sum(float(value) for value in chosen_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Gamma-7 weights must sum to 1")
    normalized = normalized_objectives(candidate, reference)
    contributions = {name: float(chosen_weights[name]) * normalized[name] for name in OBJECTIVE_NAMES}
    return {
        "score": 100.0 * sum(contributions.values()),
        "normalized": normalized,
        "weighted_contributions": contributions,
        "weights": {name: float(chosen_weights[name]) for name in OBJECTIVE_NAMES},
    }


def gamma7_score(
    candidate: AlphaMetrics,
    reference: ScoreReference,
    *,
    scenario: str | None = None,
    weights: Mapping[str, float] | None = None,
) -> float:
    return float(score_breakdown(candidate, reference, scenario=scenario, weights=weights)["score"])


def scalar(cursor: sqlite3.Cursor, query: str, parameters: tuple = ()) -> float | None:
    row = cursor.execute(query, parameters).fetchone()
    if row is None or row[0] is None:
        return None
    return float(row[0])


def evaluation_for(
    scenario: str | None,
    mission_end: float | None,
    periodic_rates: tuple[float, ...] = DEFAULT_PERIODIC_RATES,
) -> ScenarioEvaluation:
    base = SCENARIO_EVALUATIONS.get(scenario or "", ScenarioEvaluation("generic", DEFAULT_MISSION_END))
    return ScenarioEvaluation(
        name=scenario or base.name,
        mission_end=mission_end if mission_end is not None else base.mission_end,
        periodic_rates=periodic_rates,
        rate_multipliers=base.rate_multipliers,
    )


def expected_jobs_by_type(
    start_time: float,
    end_time: float,
    rates: tuple[float, ...],
    rate_multipliers: tuple[tuple[float, float, float], ...],
) -> list[float]:
    """Integrate the declared periodic workload over the common mission window."""
    boundaries = {start_time, end_time}
    for event_start, event_end, _ in rate_multipliers:
        if start_time < event_start < end_time:
            boundaries.add(event_start)
        if start_time < event_end < end_time:
            boundaries.add(event_end)

    totals = [0.0] * len(rates)
    points = sorted(boundaries)
    for left, right in zip(points, points[1:]):
        midpoint = (left + right) / 2.0
        multiplier = 1.0
        for event_start, event_end, event_multiplier in rate_multipliers:
            if event_start <= midpoint < event_end:
                multiplier = event_multiplier
                break
        duration = right - left
        for index, rate in enumerate(rates):
            totals[index] += duration * rate * multiplier
    return totals


def finish_time_expression(cursor: sqlite3.Cursor) -> str:
    columns = {row[1] for row in cursor.execute("PRAGMA table_info(jobs)")}
    return "completed_at" if "completed_at" in columns else "generated_at + time_total"


def gamma_metrics(
    cursor: sqlite3.Cursor,
    start_time: float,
    end_time: float,
    rates: tuple[float, ...],
    rate_multipliers: tuple[tuple[float, float, float], ...],
) -> tuple[float, list[float]]:
    expected_by_type = expected_jobs_by_type(start_time, end_time, rates, rate_multipliers)
    expected_total = sum(expected_by_type)
    if expected_total <= 0:
        raise ValueError("The scenario declares no periodic workload in the evaluation window.")

    finish_time = finish_time_expression(cursor)
    placeholders = ",".join("?" for _ in rates)
    rows = cursor.execute(
        f"""
        SELECT type, COUNT(*)
        FROM jobs
        WHERE node_uid = 0
          AND type IN ({placeholders})
          AND executed = 1
          AND rejected = 0
          AND over_deadline = 0
          AND generated_at >= ?
          AND generated_at < ?
          AND {finish_time} < ?
        GROUP BY type
        """,
        (*range(len(rates)), start_time, end_time, end_time),
    ).fetchall()
    timely_by_type = [0.0] * len(rates)
    for job_type, count in rows:
        if 0 <= job_type < len(timely_by_type):
            timely_by_type[job_type] = float(count)

    gamma_by_type = [timely / expected for timely, expected in zip(timely_by_type, expected_by_type)]
    return sum(timely_by_type) / expected_total, gamma_by_type


def read_metrics(
    database: Path,
    alpha: float,
    warmup: float = DEFAULT_WARMUP,
    fps_min: list[int] | None = None,
    scenario: str | None = None,
    mission_end: float | None = None,
    periodic_rates: tuple[float, ...] = DEFAULT_PERIODIC_RATES,
) -> AlphaMetrics:
    """Read Gamma-7 metrics using a strategy-independent scenario window.

    ``fps_min`` remains accepted only for source compatibility. Gamma replaces
    the legacy per-second FPS QoS definition.
    """
    evaluation = evaluation_for(scenario, mission_end, periodic_rates)
    if evaluation.mission_end <= warmup:
        raise ValueError("mission_end must be later than warmup")

    with sqlite3.connect(database) as connection:
        cursor = connection.cursor()
        last_round = scalar(cursor, "SELECT MAX(time) FROM round")
        first_death = scalar(cursor, "SELECT MIN(time) FROM end_batteries")
        if last_round is None:
            raise ValueError(f"{database}: missing battery round data")

        observed_end = min(last_round, evaluation.mission_end)
        battery_variance = scalar(
            cursor,
            "SELECT AVG(variance) FROM round WHERE time >= ? AND time < ?",
            (warmup, observed_end),
        ) or 0.0

        n_workers = scalar(cursor, "SELECT COUNT(DISTINCT worker_id) FROM round") or 0.0
        n_dead = scalar(cursor, "SELECT COUNT(DISTINCT worker_id) FROM end_batteries") or 0.0
        min_lifespan = min(first_death, evaluation.mission_end) if first_death is not None else evaluation.mission_end
        max_death = scalar(cursor, "SELECT MAX(time) FROM end_batteries") or 0.0
        max_lifespan = evaluation.mission_end if n_dead < n_workers else min(max_death, evaluation.mission_end)
        # A worker can die more than once now that battery revival works, so
        # SUM(end_batteries.time) counts every death instant and overstates the
        # total.  Total service time is instead the time nodes were actually
        # able to run tasks: full availability minus every downtime interval,
        # pairing each death with its battery_recovered revival.  For runs with
        # at most one death per worker this equals the historical expression
        # exactly, so previously published numbers are unaffected.
        deaths_by_worker: dict = {}
        for worker_id, death_time in cursor.execute(
            "SELECT worker_id, time FROM end_batteries ORDER BY worker_id, time"
        ):
            deaths_by_worker.setdefault(int(worker_id), []).append(float(death_time))
        revivals_by_worker: dict = {}
        for target, revival_time in cursor.execute(
            "SELECT target, time FROM scenario_events "
            "WHERE event_type = 'battery_recovered' ORDER BY time"
        ):
            label = str(target or "")
            if label.startswith("worker_"):
                revivals_by_worker.setdefault(label[len("worker_"):], []).append(float(revival_time))
        downtime = 0.0
        for worker_id, death_times in deaths_by_worker.items():
            revivals = revivals_by_worker.get(str(worker_id), [])
            for index, death_time in enumerate(death_times):
                recovery = revivals[index] if index < len(revivals) else evaluation.mission_end
                downtime += max(0.0, min(recovery, evaluation.mission_end) - death_time)
        total_service_time = n_workers * evaluation.mission_end - downtime

        first_death_gap = 0.0 if first_death is None else (
            scalar(cursor, "SELECT MAX(max_battery) FROM end_batteries WHERE time = ?", (first_death,)) or 0.0
        )

        total_jobs = scalar(
            cursor,
            "SELECT COUNT(*) FROM jobs WHERE generated_at >= ? AND generated_at < ?",
            (warmup, evaluation.mission_end),
        ) or 1.0
        rejected_jobs = scalar(
            cursor,
            "SELECT COUNT(*) FROM jobs WHERE rejected = 1 AND generated_at >= ? AND generated_at < ?",
            (warmup, evaluation.mission_end),
        ) or 0.0
        finish_time = finish_time_expression(cursor)
        total_throughput = scalar(
            cursor,
            f"""SELECT COUNT(*) FROM jobs
                WHERE executed = 1 AND rejected = 0
                  AND generated_at >= ? AND generated_at < ?
                  AND {finish_time} < ?""",
            (warmup, evaluation.mission_end, evaluation.mission_end),
        ) or 0.0
        avg_latency = scalar(
            cursor,
            f"""SELECT AVG(time_total) FROM jobs
                WHERE executed = 1 AND rejected = 0
                  AND generated_at >= ? AND generated_at < ?
                  AND {finish_time} < ?""",
            (warmup, evaluation.mission_end, evaluation.mission_end),
        ) or 0.0
        gamma, gamma_by_type = gamma_metrics(
            cursor,
            warmup,
            evaluation.mission_end,
            evaluation.periodic_rates,
            evaluation.rate_multipliers,
        )

        return AlphaMetrics(
            alpha=alpha,
            gamma=gamma,
            total_throughput=total_throughput,
            rejection_rate=rejected_jobs / total_jobs,
            avg_latency=avg_latency,
            min_lifespan=min_lifespan,
            total_service_time=total_service_time,
            battery_variance=battery_variance,
            gamma_by_type=gamma_by_type,
            first_death_gap=first_death_gap,
            max_lifespan=max_lifespan,
            mission_end=evaluation.mission_end,
            database=str(database),
        )


def dominates(left: AlphaMetrics, right: AlphaMetrics) -> bool:
    """Whether left is no worse on every Gamma-7 objective and better on one."""
    no_worse = (
        left.gamma >= right.gamma
        and left.total_throughput >= right.total_throughput
        and left.rejection_rate <= right.rejection_rate
        and left.avg_latency <= right.avg_latency
        and left.min_lifespan >= right.min_lifespan
        and left.total_service_time >= right.total_service_time
        and left.battery_variance <= right.battery_variance
    )
    strictly_better = (
        left.gamma > right.gamma
        or left.total_throughput > right.total_throughput
        or left.rejection_rate < right.rejection_rate
        or left.avg_latency < right.avg_latency
        or left.min_lifespan > right.min_lifespan
        or left.total_service_time > right.total_service_time
        or left.battery_variance < right.battery_variance
    )
    return no_worse and strictly_better


def pareto_front(metrics: list[AlphaMetrics]) -> list[AlphaMetrics]:
    return [candidate for candidate in metrics if not any(other is not candidate and dominates(other, candidate) for other in metrics)]


def normalized_distance(
    candidate: AlphaMetrics,
    candidates: Sequence[AlphaMetrics] | None = None,
    objective_weights: Mapping[str, float] | None = None,
    *,
    reference: ScoreReference | None = None,
) -> float:
    """Auxiliary ideal distance under a frozen reference.

    Candidate-set normalization is deliberately disabled: passing only a
    candidate list raises an error instead of silently allowing old rankings to
    change when a new strategy is added.
    """
    if reference is None:
        if candidates is not None:
            raise ValueError("normalized_distance requires a frozen ScoreReference")
        raise ValueError("normalized_distance requires a frozen ScoreReference")
    normalized = normalized_objectives(candidate, reference)
    return math.sqrt(sum((normalized[name] - 1.0) ** 2 for name in OBJECTIVE_NAMES))


def objective_weights_for(scenario: str) -> dict[str, float]:
    """Compatibility alias for callers using the former function name."""
    return weights_for(scenario)

def _parse_periodic_rates(raw: str) -> tuple[float, ...]:
    try:
        rates = tuple(float(value) for value in raw.split(","))
    except ValueError as error:
        raise SystemExit("--periodic-rates must be comma-separated numbers") from error
    if not rates or any(rate <= 0 for rate in rates):
        raise SystemExit("--periodic-rates must contain positive numbers")
    return rates


def run_metadata(database: Path) -> dict:
    """Read the run metadata of a monitored database."""
    with sqlite3.connect(database) as connection:
        return {
            key: json.loads(value)
            for key, value in connection.execute("SELECT key, value_json FROM run_metadata")
        }


def _arrival_rate_events(cursor: sqlite3.Cursor) -> list[tuple[float, float]]:
    """Arrival-rate changes recorded by the scenario, in chronological order.

    Current ``scenarios.py`` writes the canonical ``arrival_rate`` event.  Runs
    archived before that rename store the identical signal under
    ``arrival_rate_multiplier``, so both names are read here; otherwise the x1.5
    phase of the cascading scenario would be invisible to the audit.
    """
    return [
        (float(time), float(json.loads(value)))
        for time, value in cursor.execute(
            "SELECT time, value_json FROM scenario_events "
            "WHERE event_type IN ('arrival_rate', 'arrival_rate_multiplier') ORDER BY time"
        )
    ]


def check_scenario_events(database: Path, scenario: str | None) -> dict:
    """Audit recorded arrival-rate events against the declared multipliers.

    The scoring path integrates the multipliers declared in
    ``SCENARIO_EVALUATIONS``.  This audit reports a mismatch instead of letting a
    scenario that grew (or lost) a load phase silently shift ``gamma``.
    """
    evaluation = evaluation_for(scenario, None)
    declared = [
        (float(start), float(multiplier)) for start, _end, multiplier in evaluation.rate_multipliers
    ]
    with sqlite3.connect(database) as connection:
        recorded = _arrival_rate_events(connection.cursor())
    raised = [(time, value) for time, value in recorded if value != 1.0]
    match = len(raised) == len(declared) and all(
        abs(recorded_time - declared_time) < 1e-6 and abs(recorded_value - declared_value) < 1e-9
        for (recorded_time, recorded_value), (declared_time, declared_value) in zip(raised, declared)
    )
    return {
        "declared_multipliers": [
            {"start": start, "end": end, "multiplier": multiplier}
            for start, end, multiplier in evaluation.rate_multipliers
        ],
        "recorded_arrival_rate_events": [
            {"time": time, "multiplier": value} for time, value in recorded
        ],
        "match": match,
    }


def evaluate_database(
    database: Path,
    *,
    scenario: str | None = None,
    warmup: float = DEFAULT_WARMUP,
    mission_end: float | None = None,
    periodic_rates: tuple[float, ...] = DEFAULT_PERIODIC_RATES,
) -> tuple[AlphaMetrics, dict, ScenarioEvaluation]:
    """Extract the seven Gamma-7 metrics from one monitored ``log.db``."""
    metadata = run_metadata(database)
    resolved = scenario or (None if metadata.get("scenario") is None else str(metadata["scenario"]))
    resolved_warmup = float(metadata.get("warmup", warmup))
    resolved_end = mission_end
    if resolved_end is None and metadata.get("mission_end") is not None:
        resolved_end = float(metadata["mission_end"])
    evaluation = evaluation_for(resolved, resolved_end, periodic_rates)
    metrics = read_metrics(
        database,
        float(metadata.get("alpha", 0.0) or 0.0),
        warmup=resolved_warmup,
        scenario=resolved,
        mission_end=evaluation.mission_end,
        periodic_rates=periodic_rates,
    )
    return metrics, metadata, evaluation


def _run_database(args, periodic_rates: tuple[float, ...]) -> None:
    database: Path = args.database
    if not database.is_file():
        raise SystemExit(f"no such log.db: {database}")
    metrics, metadata, evaluation = evaluate_database(
        database,
        scenario=args.scenario,
        warmup=args.warmup,
        mission_end=args.mission_end,
        periodic_rates=periodic_rates,
    )
    report: dict[str, object] = {
        "method": "Gamma-7 metric extraction",
        "score_version": SCORE_VERSION,
        "scenario": evaluation.name,
        "database": str(database),
        "strategy": metadata.get("strategy"),
        "warmup": float(metadata.get("warmup", args.warmup)),
        "mission_end": evaluation.mission_end,
        "periodic_rates": list(periodic_rates),
        "weights": weights_for(evaluation.name),
        "gamma7": asdict(metrics),
    }
    if args.reference:
        reference = load_score_reference(args.reference)
        report["score"] = score_breakdown(metrics, reference, scenario=evaluation.name)
    if args.check_scenario_events:
        report["scenario_event_check"] = check_scenario_events(database, evaluation.name)
    if args.phase_breakdown:
        warmup = float(metadata.get("warmup", args.warmup))
        boundaries = [warmup] + [
            float(value)
            for value in metadata.get("phase_ends", [])
            if warmup < float(value) < evaluation.mission_end
        ]
        ends = boundaries[1:] + [evaluation.mission_end]
        report["phase_metrics"] = [
            {
                "start": start,
                "end": end,
                **asdict(
                    read_metrics(
                        database,
                        float(metadata.get("alpha", 0.0) or 0.0),
                        warmup=start,
                        scenario=evaluation.name,
                        mission_end=end,
                        periodic_rates=periodic_rates,
                    )
                ),
            }
            for start, end in zip(boundaries, ends)
        ]
    _emit(report, args.output)


def _emit(report: Mapping[str, object], output: Path | None) -> None:
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")


def find_databases(log_root: Path, action_space: str, session_id: str) -> Iterable[tuple[float, Path]]:
    root = log_root / action_space
    for database in root.glob(f"{session_id}_*/log.db"):
        suffix = database.parent.name.removeprefix(f"{session_id}_")
        try:
            yield float(suffix), database
        except ValueError:
            continue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gamma-7 evaluation: score one log.db, or rank a fixed-alpha sweep."
    )
    parser.add_argument(
        "database",
        nargs="?",
        type=Path,
        help="Evaluate a single monitored log.db instead of ranking a sweep.",
    )
    parser.add_argument("--session-id", help="Sweep session prefix, e.g. 20260804_cascading_fixed_alpha.")
    parser.add_argument("--action-space", choices=("ONLY_WORKERS", "WORKERS_OR_CLOUD"))
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--mission-end", type=float, default=None)
    parser.add_argument("--periodic-rates", default="60,30,15")
    parser.add_argument("--reference", type=Path, help="Frozen Gamma-7 reference JSON.")
    parser.add_argument("--reference-output", type=Path, help="Where to persist the frozen reference.")
    parser.add_argument(
        "--phase-breakdown",
        action="store_true",
        help="Single-database mode: also score each scenario phase separately.",
    )
    parser.add_argument(
        "--check-scenario-events",
        action="store_true",
        help="Single-database mode: audit recorded arrival-rate events against the declared multipliers.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    periodic_rates = _parse_periodic_rates(args.periodic_rates)

    if args.database is not None:
        _run_database(args, periodic_rates)
        return
    if not args.session_id:
        raise SystemExit("provide a log.db to evaluate, or --session-id to rank a sweep")
    if not args.action_space:
        raise SystemExit("--action-space is required when ranking a sweep")

    metrics = sorted(
        (
            read_metrics(
                database,
                alpha,
                warmup=args.warmup,
                scenario=args.scenario,
                mission_end=args.mission_end,
                periodic_rates=periodic_rates,
            )
            for alpha, database in find_databases(args.log_root, args.action_space, args.session_id)
        ),
        key=lambda item: item.alpha,
    )
    if len(metrics) < 2:
        raise SystemExit("At least two alpha log databases are required for a fixed-alpha reference sweep.")

    evaluation = evaluation_for(args.scenario, args.mission_end, periodic_rates)
    reference = load_score_reference(args.reference) if args.reference else build_score_reference(
        metrics,
        source="fixed_alpha_baseline",
        dataset=f"{args.session_id}:{args.action_space}",
    )
    if args.reference_output:
        save_score_reference(reference, args.reference_output)

    scores = {item.alpha: gamma7_score(item, reference, scenario=evaluation.name) for item in metrics}
    front = pareto_front(metrics)
    selected = max(metrics, key=lambda item: (scores[item.alpha], -item.alpha))
    report = {
        "method": "Frozen-reference Gamma-7; Pareto is auxiliary only",
        "score_version": reference.score_version,
        "session_id": args.session_id,
        "action_space": args.action_space,
        "scenario": evaluation.name,
        "weights": weights_for(evaluation.name),
        "warmup": args.warmup,
        "mission_end": evaluation.mission_end,
        "periodic_rates": periodic_rates,
        "reference": reference.to_dict(),
        "recommended_alpha": selected.alpha,
        "recommended_score": scores[selected.alpha],
        "pareto_alphas": [item.alpha for item in sorted(front, key=lambda item: item.alpha)],
        "metrics": [
            asdict(item)
            | score_breakdown(item, reference, scenario=evaluation.name)
            | {
                "pareto": item in front,
                "ideal_distance": normalized_distance(item, reference=reference),
            }
            for item in sorted(metrics, key=lambda item: (-scores[item.alpha], item.alpha))
        ],
    }
    _emit(report, args.output)


if __name__ == "__main__":
    main()
