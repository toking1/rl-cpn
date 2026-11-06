# 1.0.0 — 2026-11-06

Initial source release in the upstream repository layout.

- Move simulation modules and traffic inputs from `code/` to the root.
- Publish the existing Python 3.13 dependency adaptation as `requirements.txt`.
- Adjust repository-root discovery in runtime_config.py, run_experiments.py and
  evaluate_gamma7.py for the flat layout. Metric definitions, scoring, scenario
  contracts and scheduling algorithms are unchanged.
- Retain original copyright notices and GPLv3 license.
- Exclude CNB/editor configuration, internal research notes and container
  assembly tooling that requires a separate archive snapshot.

## Limits inherited from the development source

Historical results may precede battery-recovery and alpha-trajectory logging
fixes. They should not be assumed to reproduce current code byte for byte.
Service-time evaluation can overcount if a worker dies more than once after
recovery; this release does not change that metric definition. Historical
plotting workflows require the separate CNB data branch and its statistics.

No full 87-task simulation rerun or Docker build is claimed by this release.

## Validation

Local Windows / Python 3.11.2: dependency installation and `pip check`, syntax
checks for all 32 Python modules, simulator entry-point imports, repository-root
resolution, experiment registry listing, and preview of all 87 tasks passed.
Python 3.13 and Linux execution were not tested during this release.
`checks.example.yml` is an optional GitHub Actions configuration example; it
is not an enabled workflow.
