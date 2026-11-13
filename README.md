# rl-cpn

## Introduction

rl-cpn extends [Pancio-code/Simulator](https://github.com/Pancio-code/Simulator)
with dynamic reward weights, four edge-to-cloud simulation scenarios, Gamma-7
evaluation, experiment orchestration and statistics export.

This repository follows the original author's project structure:

```text
code/                  Simulation modules, entry points and traffic inputs
experiments/           Experiment reproduction and plotting entry points
environment/           Dependencies, Dockerfile and post-install script
metadata/metadata.yml  Project description, authors and source provenance
README.md              Introduction and module conventions
REPRODUCING.md         Setup and execution instructions
```

## Citations

Andrea Panceri, Gabriele Proietti Mattia and Roberto Beraldi (2024).
*Real-time and Energy-aware Scheduling for Edge-to-Cloud Continuum based on
Reinforcement Learning.* See the [original repository](https://github.com/Pancio-code/Simulator).

Please cite the original work and redistribute under the same
[GPLv3 license](code/LICENSE). Original copyright headers are retained.
Research extensions are by TANG QINGBO.

## Instructions

Module names below are relative to `code/`, following the author's conventions:

- `run_*.py`: run simulations and experiments.
- `node.py`, `cloud.py`, `job.py`, `models.py`: simulation models.
- `simulation_log.py`: logging utilities.
- `service_*.py`: simulation services.
- `traffic/`: fixed simulation inputs.
- `dynamic_alpha.py`: reactive reward-weight controllers.
- `scenarios.py`: `throughput_shock`, `energy_scarcity`, `faulted_recovery`, `cascading`.
- `evaluate_gamma7.py`: seven metrics and scenario scoring.
- `statistics_export.py`: CSV/JSON statistics export.

Experiment-specific entry points are in `experiments/`.
Setup and commands are in [REPRODUCING.md](REPRODUCING.md).

## Experiment analysis data

The following directories contain the archived analysis tables and frozen
references, grouped by experiment. Repository directories use lowercase
English names with hyphens between words; existing Python module names
retain underscores.

- [Repeated fixed-weight and feedback experiments](experiments/replication/analysis-data/README.md)
- [Ablation experiments](experiments/ablation/analysis-data/README.md)
- [Weight-adjustment and placement comparisons](experiments/comparison/analysis-data/README.md)
- [Wireless extension](experiments/extension/analysis-data/README.md)

Each experiment keeps its archived data in `analysis-data/`, alongside its
reproduction scripts. Within these archives, batch and analysis directories
use the following layout:

```text
experiments/
  replication/analysis-data/
    initial-batch/local-analysis/
    supplementary-batch/
    verification-batch/
    ten-seed-summary/
  ablation/analysis-data/
    local-analysis/
    supplementary-ablation/
  comparison/analysis-data/
    final/
    tune/screen/
    tune/valid/
  extension/analysis-data/
```

These files are analyzed metrics, scores, and related summaries, not the raw
simulation databases. Raw databases are retained outside this repository and
are not publicly available here. Database paths in CSV files are historical
execution-environment identifiers, not links to downloadable files. The
analysis archives preserve their original batch grouping and values; validation
and parameter-selection runs must not be counted as additional formal runs.
