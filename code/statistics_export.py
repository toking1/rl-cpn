from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
CSV_FIELDS = (
    "experiment_id",
    "batch_id",
    "prefix",
    "session_id",
    "scenario",
    "strategy",
    "alpha",
    "strategy_parameters",
    "seed",
    "gamma",
    "total_throughput",
    "rejection_rate",
    "avg_latency",
    "min_lifespan",
    "total_service_time",
    "battery_variance",
    "mission_end",
    "battery_scaling",
    "simulation_code_commit",
    "statistics_code_commit",
    "generated_at",
    "raw_relative_path",
    "raw_sha256",
    "config_sha256",
)
METRIC_FIELDS = (
    "gamma",
    "total_throughput",
    "rejection_rate",
    "avg_latency",
    "min_lifespan",
    "total_service_time",
    "battery_variance",
)


def parse_session_name(name: str, prefix: str) -> dict[str, Any] | None:
    """Parse supported historical session names without importing simulations."""
    marker = f"{prefix}_"
    if not name.startswith(marker):
        return None
    suffix = name[len(marker):]

    single = re.fullmatch(
        r"single_(throughput_shock|energy_scarcity|faulted_recovery)"
        r"_alpha_(0\.\d{2}|1\.00)",
        suffix,
    )
    if single:
        return {
            "scenario": single.group(1),
            "strategy": "fixed_alpha",
            "alpha": float(single.group(2)),
            "battery_scaling": 1.0,
        }

    fixed = re.fullmatch(r"cascading_fixed_alpha_(0\.\d{2}|1\.00)", suffix)
    if fixed:
        return {
            "scenario": "cascading",
            "strategy": "fixed_alpha",
            "alpha": float(fixed.group(1)),
            "battery_scaling": 3.6,
        }

    dynamic = re.fullmatch(
        r"cascading_dynamic_(reactive_rejection|reactive_battery|reactive_hybrid)",
        suffix,
    )
    if dynamic:
        return {
            "scenario": "cascading",
            "strategy": dynamic.group(1),
            "alpha": None,
            "battery_scaling": 3.6,
        }

    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validate_database(database: Path) -> None:
    if not database.is_file():
        raise ValueError("Database does not exist or is not a regular file")
    if database.stat().st_size < 1024:
        raise ValueError("Database is smaller than 1024 bytes")
    with database.open("rb") as stream:
        if stream.read(16) != b"SQLite format 3\x00":
            raise ValueError("Database has an invalid SQLite header")


def _validate_component(value: str, label: str) -> None:
    if (
        not value
        or value in (".", "..")
        or "/" in value
        or "\\" in value
        or ":" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be a single non-empty path component")


def _reason(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def export_statistics(
    results_root: Path,
    output_root: Path,
    *,
    experiment_id: str,
    batch_id: str,
    prefix: str,
    seed: int,
    sessions: Iterable[str] | None = None,
    simulation_code_commit: str = "",
    statistics_code_commit: str = "",
    config: dict | None = None,
) -> dict[str, Any]:
    """Extract per-session metrics and write scenario CSVs and JSON manifests."""
    failures: list[dict[str, str]] = []
    written: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    meta_exists: dict[str, dict[str, bool]] = {}
    generated_at = datetime.now(timezone.utc).isoformat()

    def fail(session: str, error: Exception) -> None:
        failures.append({"session": session, "reason": _reason(error)})

    def summary(ok: int) -> dict[str, Any]:
        return {
            "written": written,
            "ok": ok,
            "failed": len(failures),
            "failures": failures,
        }

    try:
        _validate_component(experiment_id, "experiment_id")
        _validate_component(batch_id, "batch_id")
        results_root = Path(results_root).expanduser().absolute()
        output_root = Path(output_root).expanduser().absolute()
        config_sha256 = (
            hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
            if config is not None
            else ""
        )
    except Exception as error:
        fail("<configuration>", error)
        return summary(0)

    session_ids: list[str] = []
    try:
        if sessions is None:
            for directory in sorted(results_root.iterdir(), key=lambda p: p.name):
                if not directory.name.startswith(f"{prefix}_"):
                    continue
                try:
                    if directory.is_dir() and (directory / "log.db").exists():
                        session_ids.append(directory.name)
                except Exception as error:
                    fail(directory.name, error)
        else:
            session_ids = sorted(set(sessions))
    except Exception as error:
        fail("<session discovery>", error)

    for session_id in session_ids:
        try:
            _validate_component(session_id, "session_id")
            parsed = parse_session_name(session_id, prefix)
            if parsed is None:
                raise ValueError("Session name does not match a supported convention")

            scenario = parsed["scenario"]
            groups.setdefault(scenario, [])
            meta_exists.setdefault(scenario, {})

            database = results_root / session_id / "log.db"
            _validate_database(database)
            has_meta = (database.parent / "meta.txt").exists()
            raw_sha256 = _sha256(database)

            # Keep module import lightweight. Dynamic sessions have no single
            # fixed alpha; 0.0 supplies the required metric-label argument only.
            # Their exported alpha remains empty instead of asserting a value.
            from evaluate_gamma7 import read_metrics

            metric_alpha = parsed["alpha"]
            if metric_alpha is None:
                metric_alpha = 0.0
            metrics = read_metrics(
                database,
                metric_alpha,
                scenario=scenario,
                mission_end=None,
            )

            strategy = parsed["strategy"]
            parameters = (
                {} if strategy == "fixed_alpha" else {"policy": strategy}
            )
            row: dict[str, Any] = {
                "experiment_id": experiment_id,
                "batch_id": batch_id,
                "prefix": prefix,
                "session_id": session_id,
                "scenario": scenario,
                "strategy": strategy,
                "alpha": parsed["alpha"],
                "strategy_parameters": _canonical_json(parameters),
                "seed": seed,
                "mission_end": metrics.mission_end,
                "battery_scaling": parsed["battery_scaling"],
                "simulation_code_commit": simulation_code_commit,
                "statistics_code_commit": statistics_code_commit,
                "generated_at": generated_at,
                "raw_relative_path": os.path.relpath(database, results_root),
                "raw_sha256": raw_sha256,
                "config_sha256": config_sha256,
            }
            for field in METRIC_FIELDS:
                row[field] = getattr(metrics, field)

            groups[scenario].append(row)
            meta_exists[scenario][session_id] = has_meta
        except Exception as error:
            fail(session_id, error)

    # Finish all CSV writes before building manifests so every manifest includes
    # the same batch-wide extraction and CSV-write failure list.
    csv_outputs: dict[str, tuple[Path, str]] = {}
    for scenario, rows in sorted(groups.items()):
        try:
            directory = output_root / experiment_id / batch_id / scenario
            directory.mkdir(parents=True, exist_ok=True)
            csv_path = directory / "metrics.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            csv_outputs[scenario] = (csv_path, _sha256(csv_path))
        except Exception as error:
            fail(f"<output:{scenario}>", error)

    ok = 0
    for scenario, (csv_path, csv_sha256) in sorted(csv_outputs.items()):
        rows = groups[scenario]
        sessions_ok = [row["session_id"] for row in rows]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "batch_id": batch_id,
            "prefix": prefix,
            "seed": seed,
            "scenario": scenario,
            "fields": list(CSV_FIELDS),
            "row_count": len(rows),
            "sessions_ok": sessions_ok,
            "failures": list(failures),
            "complete": not failures and bool(sessions_ok),
            "generated_at": generated_at,
            "statistics_code_commit": statistics_code_commit,
            "csv_sha256": csv_sha256,
            "relative_to": "results_root",
            "meta_exists": meta_exists[scenario],
        }
        try:
            with csv_path.with_suffix(".json").open(
                "w", encoding="utf-8", newline="\n"
            ) as stream:
                json.dump(manifest, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            written.append(str(csv_path))
            ok += len(rows)
        except Exception as error:
            fail(f"<manifest:{scenario}>", error)

    # If a manifest write failed, refresh successful manifests so none claims
    # that this batch completed successfully.
    if any(item["session"].startswith("<manifest:") for item in failures):
        for csv_name in written:
            json_path = Path(csv_name).with_suffix(".json")
            try:
                with json_path.open("r", encoding="utf-8") as stream:
                    manifest = json.load(stream)
                manifest["failures"] = list(failures)
                manifest["complete"] = False
                with json_path.open("w", encoding="utf-8", newline="\n") as stream:
                    json.dump(manifest, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
            except Exception as error:
                fail(f"<manifest refresh:{json_path.parent.name}>", error)

    return summary(ok)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export simulation log databases to standard statistics tables."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sessions", help="Comma-separated session directory names")
    parser.add_argument("--simulation-code-commit", default="")
    parser.add_argument("--statistics-code-commit", default="")
    args = parser.parse_args()

    sessions = None
    if args.sessions is not None:
        sessions = [
            session.strip()
            for session in args.sessions.split(",")
            if session.strip()
        ]

    result = export_statistics(
        args.results_root,
        args.output_root,
        experiment_id=args.experiment_id,
        batch_id=args.batch_id,
        prefix=args.prefix,
        seed=args.seed,
        sessions=sessions,
        simulation_code_commit=args.simulation_code_commit,
        statistics_code_commit=args.statistics_code_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
