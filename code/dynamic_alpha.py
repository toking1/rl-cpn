"""
Three reactive dynamic alpha policies for the current V6 cascading scenario.

Reactive policies operate in [0.0, 1.0] and adapt their polling interval to
worker health.
"""

from __future__ import annotations

from enum import Enum

try:
    from scenarios import CASCADING_MAX_TIME
except ImportError:  # Compatibility with the current CNB V6 scenario module.
    from scenarios import CASCADING_SIMULATION_MAX_TIME as CASCADING_MAX_TIME


class DynamicAlphaPolicy(str, Enum):
    REACTIVE_REJECTION = "reactive_rejection"
    REACTIVE_BATTERY = "reactive_battery"
    REACTIVE_HYBRID = "reactive_hybrid"


# ---------------------------------------------------------------------------
# Shared starting alpha: the known-best value for energy_scarcity from the
# V2 multi-seed full sweep analysis (2026-07-27).
# ---------------------------------------------------------------------------
BEST_ENERGY = 0.35       # energy_scarcity

T_END = CASCADING_MAX_TIME

# Default reactive policy hyper-parameters
DEFAULT_MONITOR_PERIOD = 50    # seconds between alpha re-evaluations (normal state)
DEFAULT_WINDOW_S = 200         # sliding window for rejection rate
HIGH_REJECTION_THRESHOLD = 0.15
LOW_REJECTION_THRESHOLD = 0.05
BATTERY_CRITICAL_PCT = 0.30    # below this fraction → protect batteries
BATTERY_WARNING_PCT = 0.50     # below this fraction → shorten poll interval
ALPHA_MIN = 0.0
ALPHA_MAX = 1.0
ALPHA_MID = BEST_ENERGY        # 0.35

# Adaptive polling periods
PERIOD_NORMAL = 50    # seconds, min_battery >= BATTERY_WARNING_PCT
PERIOD_WARNING = 20   # seconds, BATTERY_CRITICAL_PCT <= min_battery < BATTERY_WARNING_PCT
PERIOD_CRITICAL = 10  # seconds, min_battery < BATTERY_CRITICAL_PCT or any_dead


# ===================================================================
# Shared helpers
# ===================================================================

def _get_worker_stats(nodes):
    """Return (min_battery_pct, any_unavailable, min_lifespan) across workers."""
    workers = nodes[1:]  # skip scheduler
    pcts = []
    lifespans = []
    any_unavailable = False
    for w in workers:
        pcts.append(w.get_battery_residual_percentage())
        lifespans.append(w.get_lifespan())
        if w.is_died():
            any_unavailable = True
    return min(pcts), any_unavailable, min(lifespans)


def _clamp_alpha(val: float) -> float:
    """Clamp alpha to [ALPHA_MIN, ALPHA_MAX] and round."""
    return round(max(ALPHA_MIN, min(ALPHA_MAX, val)), 4)


def _adaptive_period(min_pct: float, any_dead: bool) -> float:
    """Return the polling period based on current node health.

    Critical (min_battery < 30% or any node dead): 10s
    Warning  (min_battery in [30%, 50%))          : 20s
    Normal   (min_battery >= 50%)                 : 50s
    """
    if any_dead or min_pct < BATTERY_CRITICAL_PCT:
        return PERIOD_CRITICAL
    elif min_pct < BATTERY_WARNING_PCT:
        return PERIOD_WARNING
    else:
        return PERIOD_NORMAL


def _apply_alpha(scheduler, policy, previous, candidate, trigger,
                 rate=None, min_pct=None, any_dead=False, min_lifespan=None):
    """
    Set the reward weight and persist the transition when it actually moves.
    Returns the weight now in force so callers can chain updates.
    """
    new = _clamp_alpha(candidate)
    if new == previous:
        return previous
    scheduler.set_reward_alpha(new)
    logger = getattr(scheduler, "log_dynamic_alpha_change", None)
    if logger is not None:
        logger(
            getattr(policy, "value", policy),
            previous,
            new,
            trigger,
            rate,
            min_pct,
            {"any_dead": bool(any_dead), "min_lifespan": min_lifespan},
        )
    return new


# ===================================================================
# Policy implementations (each returns a SimPy process)
# ===================================================================

def _reactive_rejection(env, scheduler, nodes,
                        window_s=DEFAULT_WINDOW_S,
                        high_thr=HIGH_REJECTION_THRESHOLD,
                        low_thr=LOW_REJECTION_THRESHOLD,
                        step=0.01, end_time=T_END, **_):
    """Adjust alpha by rejection-rate feedback with adaptive polling.

    Polling period adapts to node health:
      - Critical (battery<30% or any dead): 10s
      - Warning  (battery 30-50%): 20s
      - Normal   (battery>=50%): 50s
    """
    policy = DynamicAlphaPolicy.REACTIVE_REJECTION
    _apply_alpha(scheduler, policy, scheduler.get_reward_alpha(), ALPHA_MID, "initial")

    while env.now < end_time:
        min_pct, any_dead, min_lifespan = _get_worker_stats(nodes)
        period = _adaptive_period(min_pct, any_dead)
        yield env.timeout(period)
        rate = scheduler.get_rejection_rate(window_s)
        current = scheduler.get_reward_alpha()

        if rate > high_thr:
            current = _apply_alpha(scheduler, policy, current, current + step,
                                   "high_rejection", rate, min_pct, any_dead, min_lifespan)
        elif rate < low_thr:
            current = _apply_alpha(scheduler, policy, current, current - step,
                                   "low_rejection", rate, min_pct, any_dead, min_lifespan)


def _reactive_battery(env, scheduler, nodes,
                      critical_pct=BATTERY_CRITICAL_PCT,
                      healthy_pct=0.60,
                      step=0.01, end_time=T_END, **_):
    """Adjust alpha by battery/worker-health feedback with adaptive polling.

    Polling period adapts to node health:
      - Critical (battery<30% or any dead): 10s
      - Warning  (battery 30-50%): 20s
      - Normal   (battery>=50%): 50s
    """
    policy = DynamicAlphaPolicy.REACTIVE_BATTERY
    _apply_alpha(scheduler, policy, scheduler.get_reward_alpha(), ALPHA_MID, "initial")

    while env.now < end_time:
        min_pct, any_dead, _ = _get_worker_stats(nodes)
        period = _adaptive_period(min_pct, any_dead)
        yield env.timeout(period)
        min_pct, any_dead, min_lifespan = _get_worker_stats(nodes)  # re-sample after wait
        current = scheduler.get_reward_alpha()

        if any_dead:
            current = _apply_alpha(scheduler, policy, current, current - 2 * step,
                                   "worker_dead", None, min_pct, any_dead, min_lifespan)
        elif min_pct < critical_pct:
            current = _apply_alpha(scheduler, policy, current, current - step,
                                   "battery_critical", None, min_pct, any_dead, min_lifespan)
        elif min_pct > healthy_pct:
            current = _apply_alpha(scheduler, policy, current, current + step,
                                   "battery_healthy", None, min_pct, any_dead, min_lifespan)


def _reactive_hybrid(env, scheduler, nodes,
                     window_s=DEFAULT_WINDOW_S,
                     high_thr=HIGH_REJECTION_THRESHOLD,
                     low_thr=LOW_REJECTION_THRESHOLD,
                     critical_pct=BATTERY_CRITICAL_PCT,
                     healthy_pct=0.60,
                     step=0.01, end_time=T_END, **_):
    """Hybrid feedback (rejection + battery) with adaptive polling.

    Polling period adapts to node health:
      - Critical (battery<30% or any dead): 10s
      - Warning  (battery 30-50%): 20s
      - Normal   (battery>=50%): 50s
    """
    policy = DynamicAlphaPolicy.REACTIVE_HYBRID
    _apply_alpha(scheduler, policy, scheduler.get_reward_alpha(), ALPHA_MID, "initial")

    while env.now < end_time:
        min_pct, any_dead, _ = _get_worker_stats(nodes)
        period = _adaptive_period(min_pct, any_dead)
        yield env.timeout(period)
        rate = scheduler.get_rejection_rate(window_s)
        min_pct, any_dead, min_lifespan = _get_worker_stats(nodes)  # re-sample after wait
        current = scheduler.get_reward_alpha()

        delta = 0.0
        reasons = []
        if rate > high_thr:
            delta += step
            reasons.append("high_rejection")
        elif rate < low_thr:
            delta -= step
            reasons.append("low_rejection")

        if any_dead:
            delta -= step * 1.5
            reasons.append("worker_dead")
        elif min_pct < critical_pct:
            delta -= step
            reasons.append("battery_critical")
        elif min_pct > healthy_pct and rate < low_thr:
            delta += step
            reasons.append("battery_healthy")

        if abs(delta) > 1e-9:
            current = _apply_alpha(scheduler, policy, current, current + delta,
                                   "+".join(reasons), rate, min_pct, any_dead, min_lifespan)


POLICY_REGISTRY = {
    DynamicAlphaPolicy.REACTIVE_REJECTION: _reactive_rejection,
    DynamicAlphaPolicy.REACTIVE_BATTERY: _reactive_battery,
    DynamicAlphaPolicy.REACTIVE_HYBRID: _reactive_hybrid,
}


def create_dynamic_alpha_process(env, scheduler, nodes,
                                 policy: DynamicAlphaPolicy,
                                 **kwargs):
    impl = POLICY_REGISTRY.get(policy)
    if impl is None:
        raise ValueError(f"Unknown dynamic alpha policy: {policy}")
    proc = impl(env, scheduler, nodes, **kwargs)
    return env.process(proc)
