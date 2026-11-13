# Repeated experiments: analysis data

The four scenarios and 24 strategies are represented by ten independent seeds
in `ten-seed-summary/e2_metrics_10seeds.csv` (960 records). `ten-seed-summary/analysis.json`
contains derived analysis; the adjacent audit files document figure data.

`initial-batch/` and `supplementary-batch/` contain the original and additional five-seed
metrics, respectively. `verification-batch/` is a verification batch, not five new
independent repetitions. Scenario-specific frozen references are retained in
the batches' `references/` directories. Paths in the files refer to the
original execution environment; raw databases are not included.

`initial-batch/local-analysis/` retains the initial batch's derived summaries
and fault trajectories. Directory names are normalized to English; archived
data contents and historical execution paths are unchanged.

Source: `重复实验/最终分析结果/` in the local experiment archive. Files are
copied without changing their contents.
