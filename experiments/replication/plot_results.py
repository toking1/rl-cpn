from pathlib import Path
import argparse
import os
import subprocess
import sys

repo = Path(__file__).resolve().parents[2]
folder = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Generate PDF figures from replication results")
parser.add_argument("--output", type=Path, default=Path("results/replication"),
                    help="Replication output directory relative to the repository root")
args = parser.parse_args()
if args.output.is_absolute() or ".." in args.output.parts:
    parser.error("--output must be relative to results/")
experiment = (repo / args.output).resolve()
results = (repo / "results").resolve()
if not experiment.is_relative_to(results) or experiment == results:
    parser.error("--output must be a subdirectory of results/")
env = os.environ.copy()
env["RL_CPN_PLOT_ROOT"] = str(experiment / "plots")
for name in ("prepare_plot_data.py", "redraw_final_paper_figures.py"):
    subprocess.run([sys.executable, str(folder / name)], cwd=repo, env=env, check=True)
