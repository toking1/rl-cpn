# Reproduction guide

## Environment

Run the following commands from the repository root on Linux:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r environment/requirements-py313.txt
    python -m pip check

The simulation models, traffic inputs, scenarios and Gamma-7 evaluation are in code/. The experiment folders contain the corresponding controllers, orchestration scripts and, where required, frozen scoring references. Output paths in the commands below are relative to the repository root and must be subdirectories of results/.

## Select experiments and run simulations

Each seed defines one replication. Set --seeds to the desired unique seeds and --workers to the desired maximum concurrent processes. Add --dry-run to any command to list the task matrix without running simulations.

Replication covers four scenarios and 21 fixed-alpha strategies plus three reactive strategies per seed:

    python experiments/replication/reproduce.py --seeds 20260715 20260716 20260717 20260718 20260719 --workers 4 --output results/replication

Comparison covers four scenarios and nine methods per seed: PBARWS, UCB, DQN, PPO, LyDROO, DEBO, Greedy, All-Edge and All-Local:

    python experiments/comparison/reproduce.py --seeds 20261110 20261111 --workers 4 --output results/comparison

LyDROO and DEBO are model-based adaptations defined in the comparison baseline source files.

Ablation covers four scenarios and ten base configurations plus four supplementary variants per seed:

    python experiments/ablation/reproduce.py --seeds 20261020 20261021 --workers 4 --output results/ablation

Extension covers cascading and throughput-shock scenarios with three wireless policies per seed:

    python experiments/extension/reproduce.py --seeds 20261020 20261021 --workers 4 --output results/extension

The repetition count equals the number of seeds. Each scenario and strategy uses the corresponding seed. A task is skipped when its completed run record already exists. Run the same command with the same seeds and output directory to resume a matching experiment. A different experiment configuration requires a different output directory.

## Simulation data and results

Each experiment uses its own --output directory, referred to below as OUTPUT. For the examples above, these are results/replication, results/comparison, results/ablation and results/extension.

| Relative path under OUTPUT | Content |
| --- | --- |
| experiment_manifest.json | Experiment name, seeds, scenarios and strategy matrix |
| raw/_log/learning/D_SARSA/WORKERS_OR_CLOUD/ | Per-run SQLite databases, logs and metadata |
| raw/_node_logs/ and raw/_plot/ | Node logs and plot directories created by the simulator |
| models/ | Per-run scheduler models |
| archives/ | Compressed per-run databases and, when available, diagnostics |
| analysis/runs/ | Per-run metrics, scoring, validation and checksums |
| analysis/metrics.csv | Per-run metric and score summary |
| analysis/rankings.csv | Mean scores and standard deviations by scenario and strategy |
| analysis/references/ | Per-scenario Gamma-7 scoring references |
| analysis/trajectories/ | Collected controller trajectories and phase metrics for adaptation experiments |
| traces/ | Controller checks, telemetry and phase diagnostics for adaptation experiments |
| logs/ | Per-task execution logs |

Replication builds each reference from the current fixed-alpha runs. Comparison, ablation and extension use the reference files bundled in their experiment folders. Stored scoring references are required by the plotting scripts.

## Generate figures

After the corresponding experiment completes, run one of the following commands with the same OUTPUT used for the simulation:

    python experiments/replication/plot_results.py --output results/replication
    python experiments/ablation/plot_results.py --output results/ablation
    python experiments/extension/plot_results.py --output results/extension

Replication plotting reads analysis/metrics.csv and analysis/references/, verifies the complete seed, scenario and strategy matrix, recomputes Gamma-7 scores, and generates seven-metric comparisons, rankings, cascading comparisons and sensitivity plots. Intermediate plotting inputs and their validation report are written to OUTPUT/plots/plot_data/ and OUTPUT/plots/data-validation.json.

Ablation plotting reads analysis/runs/ and analysis/trajectories/ for the full, proportional, no-load, no-health and fixed-selected strategies across the four scenarios. Extension plotting reads those directories for the three wireless strategies in its two scenarios. Each plot uses the lowest supplied seed for representative trajectories and all available matching seeds for aggregated phase metrics.

All generated figures are PDF files in OUTPUT/plots/figures/. Comparison has no plotting command.
