"""Placement baselines for the PBARWS experiment harness.

Only simulator instances are patched. No simulator source changes.

Names:
    lydroo: model-based Lyapunov re-instantiation, NOT original LyDROO.
    debo: delayed-feedback Batched UCB, NOT a complete DEBO reproduction.
    all_local, all_edge, greedy_place: non-learning references.

Observation boundary:
    Current job descriptors; scheduler's outstanding-load counters and
    transmission backlog; current worker availability/speed; configured
    service times, link rates and CPU powers.

Feedback boundary:
    DataStorage.done_job(), after Job.a_done(). No unfinished-job feedback.

Energy:
    Worker CPU execution energy only, in Wh. Excludes idle, networking,
    cloud energy, interrupted partial work and battery/solar shocks.

Lyapunov:
    Virtual queues constrain PREDICTED normalized per-decision costs.
    They are not physical queues, per-second budgets, or measured-energy
    guarantees.

The installer writes diagnostics when subsequently executed by the harness.
Importing this module does not write files or run simulations.
"""

import hashlib
import json
import math
import os
import time
from pathlib import Path


VERSION = 'placement-baselines-v1'
VARIANTS = [
    'lydroo', 'debo', 'all_local', 'all_edge', 'greedy_place',
]
PARAM_ENV = 'PBARWS_PLACEMENT_PARAMS'

DEFAULTS = {
    'lydroo': {'V': 10.},
    'debo': {'c': math.sqrt(2.)},
    'all_local': {},
    'all_edge': {},
    'greedy_place': {},
}

# Ordered preregistered candidates; exact ties use this order.
CANDIDATES = {
    'lydroo': [{'V': v} for v in [0.1, 1., 10., 100., 1000.]],
    'debo': [{'c': c} for c in [.25, .5, 1., math.sqrt(2.), 2.]],
}

# Fixed adaptation assumptions; do not tune on test seeds.
BATCH_SIZE = 32
ENERGY_REFERENCE_WH = 1.e-4
DELAY_WEIGHT = .25
ENERGY_WEIGHT = .25
ENERGY_BUDGET = .25
DELAY_BUDGET = .5
TELEMETRY_PERIOD = 10.


def settings(variant, params=None):
    base = variant.split('__', 1)[0]
    if base not in DEFAULTS:
        raise ValueError(f'Unknown placement baseline: {variant}')

    groups = json.loads(os.environ.get(PARAM_ENV, '{}'))
    if not isinstance(groups, dict):
        raise ValueError(f'{PARAM_ENV} must contain a JSON object')
    if set(groups) - set(VARIANTS):
        raise ValueError('Unknown placement parameter group')
    for name, group in groups.items():
        if not isinstance(group, dict):
            raise ValueError(f'Parameters for {name} must be an object')
        if set(group) - set(DEFAULTS[name]):
            raise ValueError(f'Unknown parameters for {name}')

    supplied = dict(groups.get(base, {}))
    if params is not None:
        if not isinstance(params, dict):
            raise ValueError('params must be a dictionary')
        supplied.update(params)
    if set(supplied) - set(DEFAULTS[base]):
        raise ValueError(f'Unknown parameters for {base}')

    result = dict(DEFAULTS[base])
    result.update(supplied)
    for key, value in result.items():
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0):
            raise ValueError(f'{key} must be finite and positive')
        result[key] = float(value)
    return result


def configuration(base, parameters):
    return {
        'version': VERSION,
        'base': base,
        'parameters': dict(parameters),
        'batch_size': BATCH_SIZE,
        'energy_reference_wh': ENERGY_REFERENCE_WH,
        'delay_weight': DELAY_WEIGHT,
        'energy_weight': ENERGY_WEIGHT,
        'energy_budget': ENERGY_BUDGET,
        'delay_budget': DELAY_BUDGET,
        'energy_scope': 'worker_cpu_execution_only',
        'lyapunov_update': 'predicted_cost_per_placement',
        'environment': 'w3_wired',
    }


def signature(config):
    payload = json.dumps(
        config, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def config_tag(variant, params=None):
    base = variant.split('__', 1)[0]
    return signature(configuration(base, settings(variant, params)))


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def nonnegative(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def costs(latency, deadline, energy_wh, timely):
    latency = nonnegative(latency, 'latency')
    deadline = positive(deadline, 'deadline')
    energy_wh = nonnegative(energy_wh, 'energy_wh')
    delay = min(latency / deadline, 2.) / 2.
    energy = min(energy_wh / ENERGY_REFERENCE_WH, 1.)
    utility = float(bool(timely)) - DELAY_WEIGHT * delay
    utility -= ENERGY_WEIGHT * energy
    return {
        'latency_s': latency,
        'deadline_s': deadline,
        'cpu_energy_wh': energy_wh,
        'timely': bool(timely),
        'delay_cost': delay,
        'energy_cost': energy,
        'utility': utility,
    }


class PlacementPolicy:
    """One fresh policy per simulation; deterministic tie breaking."""

    def __init__(self, base, parameters, env, scheduler, workers, cloud):
        self.base = base
        self.parameters = parameters
        self.env = env
        self.scheduler = scheduler
        self.workers = workers
        self.cloud = cloud
        self.arms = list(range(1, len(workers) + 2))

        self.counts = {a: 0 for a in self.arms}
        self.sums = {a: 0. for a in self.arms}
        self.inflight = {a: 0 for a in self.arms}
        self.buffer_counts = {a: 0 for a in self.arms}
        self.buffer = []
        self.updates = 0

        self.energy_queue = 0.
        self.delay_queue = 0.
        self.pending = {}

        self.nominal_durations = (
            list(scheduler._job_periodic_durations)
            + list(scheduler._job_exponential_durations)
        )
        for value in self.nominal_durations:
            positive(value, 'nominal job duration')

    @staticmethod
    def key(job):
        return (job.get_originator_node_uid(), job.get_uid())

    def cpu_power(self, action):
        if action < 2:
            return 0.
        return nonnegative(
            self.workers[action - 2]._power_max_cpu_w, 'CPU power',
        )

    def estimate(self, job, action):
        """Deterministic approximation; never samples a future duration.

        Outstanding jobs include work in transit and work returning to
        the client. Charging nominal service for all of them is a
        conservative workload proxy, not an exact FCFS waiting time.

        Network backlog uses the arriving job's payload for each queued
        transmission. It ignores the residual time of a busy transfer.
        Link arithmetic intentionally matches node.py's existing units.
        """
        scheduler = self.scheduler
        payload = nonnegative(job.get_payload_size(), 'payload')
        duration = positive(job.get_job_duration(), 'job duration')
        deadline = positive(job.get_deadline(), 'deadline')
        age = nonnegative(
            self.env.now - job.get_generated_at(), 'job age',
        )

        client_rate = positive(
            scheduler._net_speed_client_scheduler_mbits, 'client rate',
        )
        worker_rate = positive(
            scheduler._net_speed_scheduler_worker_mbits, 'worker rate',
        )
        cloud_rate = positive(
            scheduler._net_speed_scheduler_cloud_mbits, 'cloud rate',
        )
        return_to_client = payload * 8. / client_rate
        network_backlog = (
            len(scheduler._jobs_transmission_list)
            * payload * 8. / min(client_rate, worker_rate, cloud_rate)
        )

        if action == 1:
            speed = positive(self.cloud._machine_speed, 'cloud speed')
            service = duration / speed
            latency = (
                age + network_backlog + return_to_client
                + 2. * payload * 8. / cloud_rate
                + nonnegative(
                    self.cloud._latency_roundtrip_ms, 'cloud RTT',
                ) / 1000.
                + service
            )
            energy_wh = 0.
        else:
            index = action - 2
            worker = self.workers[index]
            speed = positive(
                worker._machine_speed * worker.get_runtime_speed_factor(),
                'worker speed',
            )
            loads = scheduler._loads_cluster[index]
            if len(loads) != len(self.nominal_durations):
                raise RuntimeError('Unexpected job-type/load layout')
            workload = sum(
                nonnegative(count, 'outstanding load') * nominal
                for count, nominal in zip(loads, self.nominal_durations)
            )
            service = duration / speed
            latency = (
                age + network_backlog + return_to_client
                + 2. * payload * 8. / worker_rate
                + workload / speed + service
            )
            energy_wh = self.cpu_power(action) * service / 3600.

        return costs(latency, deadline, energy_wh, latency <= deadline)

    def drift_score(self, predicted):
        next_energy = max(
            0., self.energy_queue
            + predicted['energy_cost'] - ENERGY_BUDGET,
        )
        next_delay = max(
            0., self.delay_queue
            + predicted['delay_cost'] - DELAY_BUDGET,
        )
        drift = .5 * (
            next_energy ** 2 - self.energy_queue ** 2
            + next_delay ** 2 - self.delay_queue ** 2
        )
        return drift - self.parameters['V'] * predicted['utility']

    def ucb_action(self, feasible):
        # Outstanding/buffered samples balance cold-start assignments only.
        # They are NEVER treated as observed rewards or UCB sample counts.
        unseen = [a for a in feasible if self.counts[a] == 0]
        if unseen:
            return min(
                unseen,
                key=lambda a: (
                    self.inflight[a] + self.buffer_counts[a], a,
                ),
            )

        total = sum(self.counts.values())
        coefficient = self.parameters['c']

        def score(action):
            n = self.counts[action]
            return (
                self.sums[action] / n
                + coefficient * math.sqrt(math.log(max(2, total)) / n)
            )

        return max(feasible, key=lambda a: (score(a), -a))

    def choose(self, state, job):
        key = self.key(job)
        if key in self.pending:
            raise RuntimeError(f'Duplicate placement: {key}')

        # Preserve native capacity and availability admission rules.
        feasible = [
            int(a)
            for a in self.scheduler._get_possible_actions(job, state)
            if a != 0
        ]
        if not feasible or any(a not in self.arms for a in feasible):
            raise RuntimeError('Unexpected WORKERS_OR_CLOUD action set')

        predicted = {}
        before = [self.energy_queue, self.delay_queue]

        if self.base == 'all_edge':
            if 1 not in feasible:
                raise RuntimeError('Cloud action unexpectedly unavailable')
            action = 1
        elif self.base == 'all_local':
            local = [a for a in feasible if a >= 2]
            action = min(
                local,
                key=lambda a: (
                    sum(self.scheduler._loads_cluster[a - 2]), a,
                ),
            ) if local else 0
        elif self.base == 'debo':
            action = self.ucb_action(feasible)
        else:
            predicted = {a: self.estimate(job, a) for a in feasible}
            if self.base == 'greedy_place':
                action = max(
                    feasible,
                    key=lambda a: (predicted[a]['utility'], -a),
                )
            elif self.base == 'lydroo':
                action = min(
                    feasible,
                    key=lambda a: (self.drift_score(predicted[a]), a),
                )
                selected = predicted[action]
                self.energy_queue = max(
                    0., self.energy_queue
                    + selected['energy_cost'] - ENERGY_BUDGET,
                )
                self.delay_queue = max(
                    0., self.delay_queue
                    + selected['delay_cost'] - DELAY_BUDGET,
                )
            else:
                raise RuntimeError(f'Unhandled baseline: {self.base}')

        self.pending[key] = {
            'action': action,
            'cpu_power_w': self.cpu_power(action),
            'deadline_s': positive(job.get_deadline(), 'deadline'),
        }
        if self.base == 'debo':
            self.inflight[action] += 1

        return action, {
            'job': list(key),
            'action': action,
            'feasible_actions': feasible,
            'predictions': predicted,
            'virtual_queues_before': before,
            'virtual_queues_after': [
                self.energy_queue, self.delay_queue,
            ],
        }

    def completed(self, job):
        key = self.key(job)
        record = self.pending.pop(key, None)
        if record is None:
            raise RuntimeError(f'Completion without placement: {key}')
        if not job.is_done():
            raise RuntimeError('Feedback arrived before job completion')

        action = record['action']
        execution = (
            nonnegative(job.get_total_time_execution(), 'execution time')
            if job.is_executed() else 0.
        )
        energy_wh = record['cpu_power_w'] * execution / 3600.
        latency = nonnegative(job.get_total_time(), 'completed latency')
        timely = (
            job.is_executed()
            and not job.is_rejected()
            and latency <= record['deadline_s']
        )
        observed = costs(
            latency, record['deadline_s'], energy_wh, timely,
        )

        batch_committed = False
        if self.base == 'debo':
            self.inflight[action] -= 1
            # U lies in [-0.5, 1]; map to [0, 1] for UCB.
            reward = (observed['utility'] + .5) / 1.5
            self.buffer.append((action, reward))
            self.buffer_counts[action] += 1

            if len(self.buffer) == BATCH_SIZE:
                for arm, value in self.buffer:
                    self.counts[arm] += 1
                    self.sums[arm] += value
                self.buffer.clear()
                for arm in self.arms:
                    self.buffer_counts[arm] = 0
                self.updates += 1
                batch_committed = True

        return {
            'job': list(key),
            'action': action,
            'executed': bool(job.is_executed()),
            'rejected': bool(job.is_rejected()),
            **observed,
            'batch_committed': batch_committed,
        }

    def summary(self):
        return {
            'pending_at_horizon': len(self.pending),
            'uncommitted_completed_feedback': len(self.buffer),
            'batch_updates': self.updates,
            'committed_counts': dict(self.counts),
            'committed_reward_sums': dict(self.sums),
            'energy_virtual_queue': self.energy_queue,
            'delay_virtual_queue': self.delay_queue,
        }


def install(sim, variant, folder, scenario, *, seed=None,
            scale_injector=None, params=None):
    """Return an idempotent close() callback, matching controller.install."""
    import controller as harness

    parts = variant.split('__')
    if len(parts) > 2:
        raise ValueError(f'Invalid strategy: {variant}')
    base = parts[0]
    environment = parts[1] if len(parts) == 2 else 'w3_wired'
    if environment != 'w3_wired':
        raise NotImplementedError(
            'Placement v1 supports w3_wired only. Wireless placement '
            'requires an explicitly delayed observation adapter.'
        )
    if scenario not in harness.BEST:
        raise ValueError(f'Unknown scenario: {scenario}')
    if seed is None:
        raise ValueError('A shared experiment seed is required')
    if scale_injector is not None:
        raise NotImplementedError('Worker scaling is not implemented')

    parameters = settings(variant, params)
    config = configuration(base, parameters)
    tag = signature(config)
    original_factory = sim.create_dynamic_alpha_process
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    streams = {
        name: (folder / f'{name}.jsonl').open('w', buffering=1)
        for name in ['placements', 'feedback', 'telemetry']
    }
    stats = {
        'checks': 0,
        'changes': 0,
        'controller_seconds': 0.,
        'observer_seconds': 0.,
        'completed_feedback': 0,
        'version': VERSION,
        'config_tag': tag,
        'seed': int(seed),
        'scenario': scenario,
        'configuration': config,
    }
    patches = []
    processes = []
    learner = None
    closed = False

    def emit(name, row):
        streams[name].write(
            json.dumps(
                row, separators=(',', ':'), allow_nan=False,
            ) + '\n'
        )

    def patch(obj, name, replacement):
        # Restore the original instance/class lookup behavior on close.
        had_instance_value = name in vars(obj)
        previous = vars(obj).get(name)
        patches.append((obj, name, had_instance_value, previous))
        setattr(obj, name, replacement)

    def factory(env, scheduler, nodes, policy, **kwargs):
        nonlocal learner
        from node import Node

        if closed or learner is not None:
            raise RuntimeError('Install a fresh baseline for each run')
        if env.now != 0:
            raise ValueError('Placement installation requires t=0')
        if scheduler._actions_space != Node.ActionsSpace.WORKERS_OR_CLOUD:
            raise NotImplementedError('Requires WORKERS_OR_CLOUD')
        if scheduler.get_learning_type() != Node.LearningType.D_SARSA:
            raise NotImplementedError('Expected the D-SARSA harness')

        discovery = scheduler._service_discovery
        if discovery.get_clusters_count() != 1:
            raise NotImplementedError('Only the single-cluster harness')
        workers = discovery.get_workers_in_cluster(
            scheduler.get_cluster_id(),
        )
        if len(workers) != 3 or len(nodes) != 4:
            raise ValueError('Expected one scheduler and three workers')
        if {id(w) for w in workers} != {id(w) for w in nodes[1:]}:
            raise ValueError('Unexpected scheduler/worker layout')
        end = positive(kwargs['end_time'], 'end_time')

        learner = PlacementPolicy(
            base, parameters, env, scheduler, workers,
            discovery.get_node_cloud(),
        )
        storage = scheduler._service_data_storage
        original_done_job = storage.done_job

        def act(state, job):
            started = time.perf_counter()
            action, row = learner.choose(state, job)
            elapsed = time.perf_counter() - started
            stats['checks'] += 1
            stats['controller_seconds'] += elapsed
            emit('placements', {
                'time': float(env.now),
                'calculation_seconds': elapsed,
                **row,
            })
            return action

        def no_sarsa_learning(alpha=1.):
            # Keep native episode draining/logging, but never update Q.
            # Retaining LearningType avoids changing storage directories.
            scheduler._no_learning_log_episode()

        def done_job(job, reward):
            # Native completion, rejection history, load release and DB
            # recording retain their existing order and implementation.
            result = original_done_job(job, reward)
            started = time.perf_counter()
            row = learner.completed(job)
            stats['controller_seconds'] += time.perf_counter() - started
            stats['completed_feedback'] += 1
            emit('feedback', {'time': float(env.now), **row})
            return result

        patch(scheduler, '_act', act)
        patch(scheduler, '_d_sarsa_learn_episode', no_sarsa_learning)
        patch(storage, 'done_job', done_job)

        # The alpha policy argument is metadata only for these variants.
        # No alpha process, alpha adaptation, or SARSA learning is started.
        def observe():
            started = time.perf_counter()
            rate, outcomes = harness.readonly_rate(scheduler, env.now)
            worker_state = harness.state(nodes)
            emit('telemetry', {
                'time': float(env.now),
                'alpha': scheduler.get_reward_alpha(),
                'rejection_rate': rate,
                'window_outcomes': outcomes,
                'min_battery_fraction': min(
                    row['battery_fraction'] for row in worker_state
                ),
                'any_dead': any(
                    row['battery_dead'] for row in worker_state
                ),
                'workers': worker_state,
                'virtual_queues': [
                    learner.energy_queue, learner.delay_queue,
                ],
            })
            stats['observer_seconds'] += time.perf_counter() - started

        def observer():
            import simpy
            try:
                while not closed and env.now < end:
                    observe()
                    yield env.timeout(
                        min(TELEMETRY_PERIOD, end - env.now)
                    )
            except simpy.Interrupt:
                return

        process = env.process(observer())
        processes.append(process)
        return process

    sim.create_dynamic_alpha_process = factory

    def close():
        nonlocal closed
        if closed:
            return stats
        closed = True

        for process in processes:
            if process.is_alive:
                process.interrupt('placement controller closed')

        for obj, name, had_instance_value, previous in reversed(patches):
            if had_instance_value:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)

        sim.create_dynamic_alpha_process = original_factory
        if learner is not None:
            stats.update(learner.summary())

        # Do not fabricate outcomes for horizon-censored jobs or commit
        # a short final batch after the decision horizon.
        for stream in streams.values():
            stream.close()
        (folder / 'control-summary.json').write_text(
            json.dumps(stats, indent=2, allow_nan=False),
            encoding='utf-8',
        )
        return stats

    return close
