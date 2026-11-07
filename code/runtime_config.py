from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any


def repo_root() -> Path:
    """Return the repository root, the parent of code/."""
    return Path(__file__).absolute().parent.parent


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a repository-relative setting or an explicit absolute override."""
    return repo_root() / Path(value).expanduser()


def _load_minimal_config(text: str) -> dict[str, Any]:
    """Parse [paths] raw_root when tomllib is unavailable."""
    section = ""
    config: dict[str, Any] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        header = re.fullmatch(r"\[([^\]]+)\]\s*(?:#.*)?", line)
        if header:
            section = header.group(1).strip()
            continue

        if section != "paths":
            continue

        assignment = re.match(r"raw_root\s*=\s*(.*)", line)
        if not assignment:
            continue

        value = assignment.group(1)
        quoted = re.fullmatch(r'"((?:[^"\\]|\\.)*)"\s*(?:#.*)?', value)
        if quoted is None:
            raise ValueError("paths.raw_root must be a double-quoted string")

        # TOML supports eight-digit Unicode escapes in addition to JSON's.
        def replace_escape(match: re.Match[str]) -> str:
            escape = match.group(0)
            if escape.startswith("\\U"):
                return json.dumps(chr(int(escape[2:], 16)))[1:-1]
            return escape

        encoded = re.sub(
            r"\\(?:U[0-9a-fA-F]{8}|.)",
            replace_escape,
            quoted.group(1),
        )
        config.setdefault("paths", {})["raw_root"] = json.loads(
            '"' + encoded + '"'
        )

    return config


def load_local_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load local configuration without creating files or directories."""
    if config_path is None:
        config_path = os.environ.get("RL_CPN_CONFIG")
    if config_path is None:
        config_path = repo_root() / "runtime.local.toml"

    path = resolve_repo_path(config_path)
    try:
        with path.open("rb") as stream:
            content = stream.read()
    except FileNotFoundError:
        return {}

    try:
        import tomllib
    except ImportError:
        return _load_minimal_config(content.decode("utf-8"))

    return tomllib.loads(content.decode("utf-8"))


def resolve_raw_root(
    explicit: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """Resolve explicit, environment, local config, then repository default."""
    if explicit is not None:
        return resolve_repo_path(explicit)

    environment_root = os.environ.get("RL_CPN_RAW_ROOT")
    if environment_root is not None:
        return resolve_repo_path(environment_root)

    config = load_local_config(config_path)
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("Configuration entry 'paths' must be a table")

    configured_root = paths.get("raw_root")
    if configured_root is not None:
        if not isinstance(configured_root, str):
            raise ValueError("Configuration entry 'paths.raw_root' must be a string")
        return resolve_repo_path(configured_root)

    return resolve_repo_path("raw-out")


def results_root(
    explicit: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """Return the directory containing historical session directories."""
    return (
        resolve_raw_root(explicit, config_path)
        / "_log"
        / "learning"
        / "D_SARSA"
        / "WORKERS_OR_CLOUD"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resolve raw simulation paths.")
    parser.add_argument("--raw-root", type=Path, help="Explicit raw data root")
    parser.add_argument("--config-path", type=Path, help="Local TOML configuration")
    args = parser.parse_args()

    raw = resolve_raw_root(args.raw_root, args.config_path)
    print(f"repo_root: {repo_root()}")
    print(f"raw_root: {raw}")
    print(f"results_root: {results_root(explicit=raw)}")
