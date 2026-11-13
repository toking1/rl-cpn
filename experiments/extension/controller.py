"""Isolated experimental controls; original simulator files remain unchanged."""
import copy
import hashlib
import json
import math
import random
import time
from collections import deque
from pathlib import Path

CONFIGS = ['full', 'no_load', 'no_health', 'period10', 'period20', 'period50',
           'symmetric', 'proportional', 'fixed_initial', 'fixed_selected']
BASE_NEW = ['no_reject_gate', 'serial_priority', 'single_scale_adaptive',
            'symmetric_same_range']
ENVIRONMENTS = {'w3_wired': (3, False), 'w3_wireless': (3, True)}
CONFIGS += [f'{b}__{e}' for b in BASE_NEW for e in ENVIRONMENTS]
CONFIGS += [f'{b}__w3_wireless'
            for b in ['full', 'proportional', 'fixed_selected']]

BEST = {'cascading': .50, 'throughput_shock': .55,
        'energy_scarcity': .40, 'faulted_recovery': .25}


def clip(x, low, high):
    return max(low, min(high, x))


def increments(rate, minimum, dead, variant):
    load = .01 if rate > .15 else (-.01 if rate < .05 else 0.)
    health = -.015 if dead else (-.01 if minimum < .30 else
             (.01 if minimum > .60 and rate < .05 else 0.))
    raw_load, raw_health = load, health

    if variant == 'no_load':
        load = 0.
    elif variant == 'no_health':
        health = 0.
    elif variant == 'no_reject_gate':
        load = 0.
        health = -.015 if dead else (-.01 if minimum < .30 else
                 (.01 if minimum > .60 else 0.))
    elif variant == 'serial_priority':
        if load != 0.:
            health = 0.
    elif variant == 'proportional':
        # Prespecified, untuned P baseline.
        load = .01 * clip((rate - .10) / .10, -1., 1.)
        health = -.015 if dead else .01 * clip(
            (minimum - .45) / .15, -1., 1.)

    # single_scale_adaptive retains full's increment logic.
    raw = load + health
    delta = raw
    if variant == 'symmetric':
        delta = clip(raw, -.01, .01)
    elif variant == 'symmetric_same_range':
        delta = clip(raw, -.0175, .0175)
    elif variant == 'proportional':
        delta = clip(raw, -.025, .01)
    return raw_load, raw_health, load, health, raw, delta


def readonly_rate(scheduler, now, window=200):
    entries = [(t, r) for t, r in scheduler._rejection_history
               if t > now - window]
    return (sum(bool(r) for _, r in entries) / len(entries)
            if entries else 0.), len(entries)


def state(nodes):
    return [{
        'id': w.get_uid(),
        'battery_fraction': w.get_battery_residual_percentage(),
        'battery_wh': w.get_battery_residual_wh(),
        'capacity_wh': w.get_battery_total_capacity_wh(),
        'battery_dead': bool(w.is_died()),
        'available': bool(w.is_available()),
        'speed_factor': getattr(w, '_runtime_speed_factor', 1.),
    } for w in nodes[1:]]


class WirelessTelemetry:
    """10-second sample history, latest arrived sample, and hold-last.

    Bootstrap is delivered immediately. Subsequent snapshots independently
    experience loss and a lognormal RTT, expressed in simulation seconds.
    Expired history cannot replace the held sample; out-of-order arrivals
    cannot move the held sample backwards.
    """

    def __init__(self, seed, scenario, p_loss=.05, history_seconds=10.):
        if seed is None:
            raise ValueError('Wireless telemetry requires seed')
        material = f'wireless-telemetry-v1:{int(seed)}:{scenario}'.encode()
        derived_seed = int.from_bytes(
            hashlib.sha256(material).digest(), 'big')
        self.rng = random.Random(derived_seed)
        self.p_loss = p_loss
        self.history_seconds = history_seconds

        mean_seconds, sd_seconds = .030, .015
        variance_log = math.log1p((sd_seconds / mean_seconds) ** 2)
        self.sigma = math.sqrt(variance_log)
        self.mu = math.log(mean_seconds) - variance_log / 2.

        self.history = deque()
        self.held = None
        self.sequence = 0
        self.dropped = 0

    def _prune(self, now):
        while (self.history and
               self.history[0]['sample_time'] < now - self.history_seconds):
            self.history.popleft()

    def sample(self, now, snapshot):
        if self.held is None and now != 0:
            raise ValueError('Wireless telemetry must bootstrap at t=0')

        bootstrap = self.held is None
        # Draw both values on every non-bootstrap sample, including losses.
        dropped = False if bootstrap else self.rng.random() < self.p_loss
        rtt = 0. if bootstrap else self.rng.lognormvariate(self.mu, self.sigma)
        packet = {
            'sequence': self.sequence,
            'sample_time': float(now),
            'delivery_time': None if dropped else float(now) + rtt,
            'rtt_seconds': rtt,
            'dropped': dropped,
            'bootstrap': bootstrap,
            'snapshot': copy.deepcopy(snapshot),
        }
        self.sequence += 1
        self.dropped += int(dropped)
        self._prune(now)
        self.history.append(packet)
        if bootstrap:
            self.held = packet
        return {
            key: value for key, value in packet.items() if key != 'snapshot'
        }

    def read(self, now):
        if self.held is None:
            raise RuntimeError('Wireless telemetry has not bootstrapped')
        self._prune(now)
        for packet in self.history:
            if (not packet['dropped']
                    and packet['delivery_time'] <= now
                    and packet['sequence'] > self.held['sequence']):
                self.held = packet

        delayed = sum(
            not packet['dropped'] and packet['delivery_time'] > now
            for packet in self.history
        )
        delivered = copy.deepcopy(self.held['snapshot'])
        delivered.update({
            'sample_time': self.held['sample_time'],
            'delivery_time': self.held['delivery_time'],
            'sample_age_seconds': float(now) - self.held['sample_time'],
            'sequence': self.held['sequence'],
        })
        return delivered, {
            'wireless': True,
            'dropped': self.dropped,
            'delayed': delayed,
        }


def install(sim, variant, folder, scenario, *, seed=None,
            scale_injector=None):
    import dynamic_alpha as native

    parts = variant.split('__')
    if len(parts) > 2:
        raise ValueError(f'Invalid strategy: {variant}')
    base = parts[0]
    environment = parts[1] if len(parts) == 2 else 'w3_wired'
    if environment.startswith('w12'):
        raise NotImplementedError(
            'w12 is deferred; scale_injector is reserved for a later phase')
    if environment not in ENVIRONMENTS:
        raise ValueError(f'Unknown environment: {environment}')
    if variant not in CONFIGS:
        raise ValueError(f'Unknown strategy: {variant}')
    if scenario not in BEST:
        raise ValueError(f'Unknown scenario: {scenario}')

    expected_workers, wireless = ENVIRONMENTS[environment]
    if wireless and seed is None:
        raise ValueError('w3_wireless requires seed')
    # Reserved API parameter. No scaling is applied in either w3 environment.
    _ = scale_injector

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    streams = {
        name: (folder / (name + '.jsonl')).open('w', buffering=1)
        for name in ['checks', 'telemetry']
    }
    stats = {
        'checks': 0,
        'changes': 0,
        'controller_seconds': 0.,
        'observer_seconds': 0.,
    }
    original_factory = sim.create_dynamic_alpha_process
    closed = False

    def emit(name, row):
        streams[name].write(json.dumps(row, separators=(',', ':')) + '\n')

    def factory(env, scheduler, nodes, policy, **kwargs):
        assert len(nodes[1:]) == expected_workers, (
            f'{variant}: expected {expected_workers} workers, '
            f'got {len(nodes[1:])}'
        )
        if env.now != 0:
            raise ValueError('Controller installation requires t=0')
        end = kwargs['end_time']
        fixed = base.startswith('fixed_')
        initial = BEST[scenario] if base == 'fixed_selected' else .35
        channel = WirelessTelemetry(seed, scenario) if wireless else None

        def snapshot():
            minimum, dead, lifespan = native._get_worker_stats(nodes)
            rate, outcomes = readonly_rate(scheduler, env.now)
            return {
                'rejection_rate': rate,
                'window_outcomes': outcomes,
                'min_battery_fraction': minimum,
                'any_dead': bool(dead),
                'lifespan': lifespan,
                'workers': state(nodes),
            }

        def observe():
            started = time.perf_counter()
            raw = snapshot()
            if channel is not None:
                packet = channel.sample(env.now, raw)
                delivered, transport = channel.read(env.now)
            else:
                packet = {
                    'sample_time': float(env.now),
                    'delivery_time': float(env.now),
                    'rtt_seconds': 0.,
                    'dropped': False,
                    'bootstrap': env.now == 0,
                }
                delivered = dict(
                    raw, sample_time=float(env.now),
                    delivery_time=float(env.now), sample_age_seconds=0.)
                transport = {'wireless': False, 'dropped': 0, 'delayed': 0}

            # Preserve original top-level fields for collect/extract.
            emit('telemetry', {
                'time': env.now,
                'alpha': scheduler.get_reward_alpha(),
                **raw,
                'raw': raw,
                'delivered': delivered,
                'packet': packet,
                **transport,
            })
            stats['observer_seconds'] += time.perf_counter() - started

        def observer():
            # process() performs the only t=0 observation.
            while env.now < end:
                yield env.timeout(10)
                if env.now > end:
                    return
                observe()

        def process():
            native._apply_alpha(
                scheduler, policy, scheduler.get_reward_alpha(),
                initial, 'initial')
            observe()
            env.process(observer())
            if fixed:
                return

            while env.now < end:
                started = time.perf_counter()
                if channel is not None:
                    delivered, _ = channel.read(env.now)
                    minimum = delivered['min_battery_fraction']
                    dead = delivered['any_dead']
                else:
                    minimum, dead, _ = native._get_worker_stats(nodes)

                would_be_period = native._adaptive_period(minimum, dead)
                if base == 'single_scale_adaptive':
                    period = 20
                elif base.startswith('period'):
                    period = int(base[6:])
                else:
                    period = would_be_period

                before_wait = env.now
                stats['controller_seconds'] += time.perf_counter() - started
                yield env.timeout(period)
                if env.now > end:
                    return

                started = time.perf_counter()
                if channel is not None:
                    delivered, transport = channel.read(env.now)
                    rate = delivered['rejection_rate']
                    minimum = delivered['min_battery_fraction']
                    dead = delivered['any_dead']
                    lifespan = delivered['lifespan']
                    sample_time = delivered['sample_time']
                    delivery_time = delivered['delivery_time']
                else:
                    rate = scheduler.get_rejection_rate(200)
                    minimum, dead, lifespan = native._get_worker_stats(nodes)
                    transport = {
                        'wireless': False, 'dropped': 0, 'delayed': 0}
                    sample_time = delivery_time = float(env.now)

                previous = scheduler.get_reward_alpha()
                rl, rh, load, health, raw, delta = increments(
                    rate, minimum, dead, base)
                new = previous
                if abs(delta) > 1e-9:
                    new = native._apply_alpha(
                        scheduler, policy, previous, previous + delta,
                        f'{variant}:load={load:.6g},health={health:.6g}',
                        rate, minimum, dead, lifespan)

                calculation_seconds = time.perf_counter() - started
                stats['controller_seconds'] += calculation_seconds
                stats['checks'] += 1
                stats['changes'] += new != previous
                emit('checks', {
                    'time': env.now,
                    'wait_started': before_wait,
                    'period': period,
                    'would_be_period': would_be_period,
                    'rejection_rate': rate,
                    'min_battery_fraction': minimum,
                    'any_dead': bool(dead),
                    'previous_alpha': previous,
                    'new_alpha': new,
                    'native_load_delta': rl,
                    'native_health_delta': rh,
                    'load_delta': load,
                    'health_delta': health,
                    'raw_delta': raw,
                    'requested_delta': delta,
                    'applied_delta': new - previous,
                    'at_boundary': new in (0., 1.),
                    'calculation_seconds': calculation_seconds,
                    'sample_time': sample_time,
                    'delivery_time': delivery_time,
                    'sample_age_seconds': float(env.now) - sample_time,
                    **transport,
                    'workers': state(nodes),
                })
        return env.process(process())

    sim.create_dynamic_alpha_process = factory

    def close():
        nonlocal closed
        sim.create_dynamic_alpha_process = original_factory
        if not closed:
            for stream in streams.values():
                stream.close()
            closed = True
        (folder / 'control-summary.json').write_text(
            json.dumps(stats, indent=2))
        return stats

    return close
