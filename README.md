# rl-cpn

## Introduction

rl-cpn extends [Pancio-code/Simulator](https://github.com/Pancio-code/Simulator)
with dynamic reward weights, four edge-to-cloud simulation scenarios, Gamma-7
evaluation, experiment orchestration and statistics export.

This repository follows the original author's project structure:

```text
code/                  Simulation modules, entry points and traffic inputs
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
- `实验/`: four experiment-specific reproduction entry points.

Setup and commands are in [REPRODUCING.md](REPRODUCING.md).
Raw simulation databases are stored outside Git.
