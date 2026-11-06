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

Raw data defaults to `/data/rl-cpn/raw` on Linux and `C:\rl-cpn-data\raw` on
Windows. Override with `RL_CPN_RAW_ROOT`, or copy
`environment/runtime.local.example.toml` to the repository root as
`runtime.local.toml` and set `paths.raw_root`. Environment variables take precedence.
Choose a writable directory outside the repository. Statistics are exported to
`statistics-out/` at the repository root.

Use `python run_experiments.py --help` for filters and options. For individual
databases use `python evaluate_gamma7.py /absolute/path/to/log.db`.

## Docker environment

The Dockerfile follows the upstream environment-only layout. From the repository root:

```bash
docker build -t rl-cpn-env environment
docker run --rm --workdir /code \
  --volume "$PWD/code":/code \
  --volume "$PWD/raw-out":/data/rl-cpn/raw \
  --volume "$PWD/statistics-out":/statistics-out \
  --env RL_CPN_STATISTICS_ROOT=/statistics-out \
  rl-cpn-env bash run --dry-run
```

Replace `--dry-run` with `--workers 1` to execute simulations. The Docker image
has not been built or tested in this session.

## Scope and known limits

Local checks cover dependencies, module syntax, simulation imports, path resolution,
experiment listing and task preview. A complete simulation rerun is not claimed.
Historical figures require statistics and analysis scripts from CNB's separate
`data/experiments` branch; they are not bundled here.

The source retains the existing evaluation definitions. Historical data may
predate battery-recovery and alpha-logging fixes. Service-time evaluation can
overcount when a worker dies more than once after recovery.
