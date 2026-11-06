"""Runtime anomaly scenarios for the canonical experiment set.

Supported scenarios (the four names advertised in README.md):
  throughput_shock : periodic arrival rate x1.5 for 4000-8000s.
  energy_scarcity  : Worker 1 loses one of its three battery modules (1/3 of the
                     total capacity and 1/3 of the residual energy) during
                     4000-7000s; a fully charged module replaces it at 7000s.
  faulted_recovery : Worker 1 is forced unavailable 4000-7000s, then runs at
                     0.45x speed 7000-8000s.
  cascading        : the 30,000s / 3.6x-battery composite benchmark below.

Canonical cascading timeline (8 phases, 30,000s total, battery scaling 3.6):
  P1      0- 4000s : baseline, no anomaly.
  P2   4000- 7000s : Worker 1 forced unavailable.
  P3   7000- 8000s : Worker 1 speed 0.45x.
  P4   8000-14000s : baseline restored.
  P5  14000-17000s : one of Worker 1's three battery modules fails.
  P6  17000-24000s : a fully charged replacement module is installed.
  P7  24000-28000s : periodic arrival rate x1.5.
  P8  28000-30000s : baseline restored, mission ends.

Retired scenarios (cascading_heavy, cascading_v5, throughput_shock_v5,
energy_scarcity_v5, faulted_recovery_v5, cascading_v6) are kept only as
exploration records in EXPERIMENT_LOG.md. They are not implemented here and must
not be used to reason about the current experiments.
"""

from __future__ import annotations

from enum import Enum


CASCADING_BATTERY_SCALING = 3.6
CASCADING_SIMULATION_MAX_TIME = 30000
# Phase boundaries after P1; used by the runner so the timeline has one owner.
CASCADING_PHASE_ENDS = (4000, 7000, 8000, 14000, 17000, 24000, 28000, 30000)


class ScenarioName(str, Enum):
    THROUGHPUT_SHOCK = "throughput_shock"
    ENERGY_SCARCITY = "energy_scarcity"
    FAULTED_RECOVERY = "faulted_recovery"
    CASCADING = "cascading"


def schedule_scenario(env, nodes, scenario: ScenarioName, data_storage=None):
    """Register one scenario process and return it.

    ``nodes`` is ordered as scheduler, worker-1, worker-2, worker-3, matching
    the original cloud runner.
    """
    scheduler, worker_1, worker_2, worker_3 = nodes

    def record(event_type, target, value, details=None):
        if data_storage is not None:
            data_storage.log_scenario_event(
                env.now, event_type, target, value, details or {}
            )

    def apply():
        # Phase 1 (0-4000s) is the shared baseline for every scenario.
        yield env.timeout(4000)

        if scenario is ScenarioName.THROUGHPUT_SHOCK:
            scheduler.set_arrival_rate_multiplier(1.5)
            record("arrival_rate", "scheduler", 1.5, {"scenario": scenario.value})
            yield env.timeout(4000)
            scheduler.set_arrival_rate_multiplier(1.0)
            record("arrival_rate", "scheduler", 1.0, {"scenario": scenario.value})

        elif scenario is ScenarioName.ENERGY_SCARCITY:
            original_capacity_wh = worker_1.get_battery_total_capacity_wh()
            module_capacity_wh = original_capacity_wh / 3.0
            lost_energy_wh = worker_1.get_battery_residual_wh() / 3.0
            restricted_capacity_wh = original_capacity_wh - module_capacity_wh
            worker_1.adjust_battery_capacity(-lost_energy_wh)
            worker_1.set_battery_total_capacity_wh(restricted_capacity_wh)
            record(
                "battery_module_failed",
                "worker_1",
                {"module_capacity_wh": module_capacity_wh,
                 "lost_energy_wh": lost_energy_wh,
                 "total_capacity_wh": restricted_capacity_wh},
                {"scenario": scenario.value, "modules_available": 2, "modules_total": 3},
            )
            yield env.timeout(3000)
            worker_1.set_battery_total_capacity_wh(original_capacity_wh)
            worker_1.adjust_battery_capacity(module_capacity_wh)
            record(
                "battery_module_replaced",
                "worker_1",
                {"module_capacity_wh": module_capacity_wh,
                 "added_energy_wh": module_capacity_wh,
                 "total_capacity_wh": original_capacity_wh},
                {"scenario": scenario.value, "replacement_charge": "full",
                 "modules_available": 3, "modules_total": 3},
            )

        elif scenario is ScenarioName.FAULTED_RECOVERY:
            worker_1.set_forced_unavailable(True)
            record("forced_unavailable", "worker_1", True, {"scenario": scenario.value})
            yield env.timeout(3000)
            worker_1.set_forced_unavailable(False)
            record("forced_unavailable", "worker_1", False, {"scenario": scenario.value})
            worker_1.set_runtime_speed_factor(0.45)
            record("speed_factor", "worker_1", 0.45, {"scenario": scenario.value})
            yield env.timeout(1000)
            worker_1.set_runtime_speed_factor(1.0)
            record("speed_factor", "worker_1", 1.0, {"scenario": scenario.value})

        elif scenario is ScenarioName.CASCADING:
            # ---- P2 (4000-7000s): Worker 1 forced unavailable ----
            worker_1.set_forced_unavailable(True)
            record("forced_unavailable", "worker_1", True, {"scenario": scenario.value})
            yield env.timeout(3000)

            # ---- P3 (7000-8000s): Worker 1 speed 0.45x ----
            worker_1.set_forced_unavailable(False)
            record("forced_unavailable", "worker_1", False, {"scenario": scenario.value})
            worker_1.set_runtime_speed_factor(0.45)
            record("speed_factor", "worker_1", 0.45, {"scenario": scenario.value})
            yield env.timeout(1000)

            # ---- P4 (8000-14000s): restore baseline ----
            worker_1.set_runtime_speed_factor(1.0)
            record("speed_factor", "worker_1", 1.0, {"scenario": scenario.value})
            yield env.timeout(6000)

            # ---- P5 (14000-17000s): one battery module of Worker 1 fails ----
            original_capacity_wh = worker_1.get_battery_total_capacity_wh()
            module_capacity_wh = original_capacity_wh / 3.0
            lost_energy_wh = worker_1.get_battery_residual_wh() / 3.0
            restricted_capacity_wh = original_capacity_wh - module_capacity_wh
            worker_1.adjust_battery_capacity(-lost_energy_wh)
            worker_1.set_battery_total_capacity_wh(restricted_capacity_wh)
            record(
                "battery_module_failed",
                "worker_1",
                {"module_capacity_wh": module_capacity_wh,
                 "lost_energy_wh": lost_energy_wh,
                 "total_capacity_wh": restricted_capacity_wh},
                {"scenario": scenario.value, "modules_available": 2, "modules_total": 3},
            )
            yield env.timeout(3000)

            # ---- P6 (17000-24000s): install one fully charged module ----
            worker_1.set_battery_total_capacity_wh(original_capacity_wh)
            worker_1.adjust_battery_capacity(module_capacity_wh)
            record(
                "battery_module_replaced",
                "worker_1",
                {"module_capacity_wh": module_capacity_wh,
                 "added_energy_wh": module_capacity_wh,
                 "total_capacity_wh": original_capacity_wh},
                {"scenario": scenario.value, "replacement_charge": "full",
                 "modules_available": 3, "modules_total": 3},
            )
            yield env.timeout(7000)

            # ---- P7 (24000-28000s): periodic arrival rate x1.5 ----
            scheduler.set_arrival_rate_multiplier(1.5)
            record("arrival_rate", "scheduler", 1.5, {"scenario": scenario.value})
            yield env.timeout(4000)

            # ---- P8 (28000-30000s): restore baseline and end ----
            scheduler.set_arrival_rate_multiplier(1.0)
            record("arrival_rate", "scheduler", 1.0, {"scenario": scenario.value})
            yield env.timeout(2000)

        else:
            raise ValueError(f"unsupported scenario: {scenario}")

    return env.process(apply())
