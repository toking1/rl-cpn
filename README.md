# rl-cpn

Reinforcement-learning edge-to-cloud scheduling simulator with dynamic reward
weights, four scenario contracts, Gamma-7 evaluation, and statistics export.

This research extension derives from [Pancio-code/Simulator](https://github.com/Pancio-code/Simulator).
The release reorganizes the [CNB development source](https://cnb.cool/CPN.toking/rl-cpn)
to follow the upstream root-level module layout and filename conventions.
Original copyright headers are retained. Distributed under [GPLv3](LICENSE).

## Layout

```text
rl-cpn/
  run_*.py                 Simulation and experiment entry points
  node.py, cloud.py, job.py Core simulation models
  service_*.py             Simulation services
  simulation_log.py        Logging (upstream log.py, renamed in CNB)
  dynamic_alpha.py         Reactive reward-weight controllers
  scenarios.py             Four scenario contracts
  evaluate_gamma7.py       Metrics and scoring
  statistics_export.py     CSV/JSON statistics export
  runtime_config.py        External raw-data paths
  traffic/                 Fixed simulation inputs
  requirements.txt         Python 3.13 dependency adaptation
  metadata/                Immutable source provenance
  docs/                    Release changes and limitations
```

## Quick start

Use Python 3.13. Run commands from the repository root.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_experiments.py list
python run_experiments.py --dry-run
```

Configure an external raw-data directory before executing simulations:

```bash
# Linux/macOS
export RL_CPN_RAW_ROOT="$HOME/rl-cpn-data/raw"
# Windows PowerShell instead:
# $env:RL_CPN_RAW_ROOT = 'C:\rl-cpn-data\raw'
python run_experiments.py --workers 1
```

Alternatively copy `runtime.local.example.toml` to `runtime.local.toml` and
set `paths.raw_root`. Environment variables take precedence. Raw databases stay
outside Git; standard statistics are generated under `statistics-out/`.
Use `python run_experiments.py --help` for filters and execution options.

## Scope

Supported scenarios: `throughput_shock`, `energy_scarcity`, `faulted_recovery`,
and `cascading`. The default experiment selection contains 87 tasks (63 single
scenario fixed-alpha, 21 cascading fixed-alpha, 3 cascading reactive).
Listing and dry-run commands do not execute the simulation matrix.

Historical analyses and plotting scripts remain on CNB's `data/experiments`
branch. This source release does not bundle raw databases, historical figures,
or a prebuilt Docker image. See [release notes](docs/RELEASE_NOTES.md).

## Attribution

Please cite the upstream authors Andrea Panceri, Gabriele Proietti Mattia and
Roberto Beraldi, and the upstream work described in their
[README](https://github.com/Pancio-code/Simulator#citations):
“Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on
Reinforcement Learning” (2024), and “Leveraging Reinforcement Learning for online
scheduling of real-time tasks in the Edge/Fog-to-Cloud computing continuum”
(2021), DOI: [10.1109/NCA53618.2021.9685413](https://doi.org/10.1109/NCA53618.2021.9685413).

Research extensions: TANG QINGBO. Source revision and original module hashes
are recorded in [release-manifest.json](metadata/release-manifest.json).
