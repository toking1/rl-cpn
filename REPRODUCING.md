# Reproducing rl-cpn

## Local environment

Run setup from the repository root. Python 3.11 was used for local smoke checks;
`requirements-py313.txt` also contains the development project's Python 3.13
dependency adaptation.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r environment/requirements-py313.txt
python -m pip check
```

`environment/requirements.txt` preserves the development project's original
dependency definitions. Use the adapted list above for the current setup.

## Simulation entry point

The working directory is `code/`, as in the original simulator.

```bash
cd code
python run_experiments.py list
python run_experiments.py --dry-run
python run_experiments.py --workers 1
```

On Linux/macOS, `bash run --dry-run` and `bash run --workers 1` invoke the same
entry point. The default matrix contains 87 tasks: 63 single-scenario fixed-alpha
runs, 21 cascading fixed-alpha runs and 3 cascading reactive runs. Registry entries
marked `runnable=False` are not executed by this matrix.

## Output directories

The following paths are relative to the repository root, including when you run
the experiment entry point from `code/`.

| Output | Default location | Override order (highest first) |
| --- | --- | --- |
| Raw simulation logs and databases | `raw-out/_log/learning/D_SARSA/WORKERS_OR_CLOUD/<session>/` | `--raw-root`, `RL_CPN_RAW_ROOT`, `paths.raw_root` in `runtime.local.toml` |
| Exported statistics | `statistics-out/<experiment>/<batch>/<scenario>/` | `--statistics-dir`, `RL_CPN_STATISTICS_ROOT` |

Both default output directories are ignored by Git. For local settings, copy
`environment/runtime.local.example.toml` to `runtime.local.toml` at the repository
root and edit `paths.raw_root`. `RL_CPN_CONFIG` can select another config file.
Relative override values use the same repository root. Existing absolute overrides
still work for archived data.

From `code/`, preview a run with custom output directories:

```bash
python run_experiments.py --dry-run --raw-root raw-out/custom --statistics-dir statistics-out/custom
```

The Gamma-7 sweep evaluator uses the configured raw directory by default.
Use `--log-root` to read an older archive, or pass a database path directly.
For standalone tools, explicit `--log-root`, `--output`, `--reference-output`,
`--results-root` and `--output-root` paths are relative to the current working
directory. From `code/`, for example:

```bash
python evaluate_gamma7.py "../raw-out/_log/learning/D_SARSA/WORKERS_OR_CLOUD/<session>/log.db" --output ../statistics-out/evaluation.json
```

## Docker environment

The Dockerfile follows the upstream environment-only layout. From the repository root:

```bash
docker build -t rl-cpn-env environment
docker run --rm --volume "$PWD":/workspace \
  rl-cpn-env bash run --dry-run
```

Both output directories are written inside the mounted repository. Replace
`--dry-run` with `--workers 1` to execute simulations. The Docker image
has not been built or tested in this session.

## Scope and known limits

Local checks cover dependencies, module syntax, simulation imports, path resolution,
experiment listing and task preview. A complete simulation rerun is not claimed.
Historical figures require statistics and analysis scripts from CNB's separate
`data/experiments` branch; they are not bundled here.

The source retains the existing evaluation definitions. Historical data may
predate battery-recovery and alpha-logging fixes. Service-time evaluation can
overcount when a worker dies more than once after recovery.
