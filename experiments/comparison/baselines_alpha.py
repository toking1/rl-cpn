"""Online alpha-controller re-instantiations for the PBARWS harness."""

import copy
import hashlib
import json
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np


VERSION = 'alpha-baselines-v1'
VARIANTS = ['ucb_mab', 'ppo_alpha', 'dqn_alpha']
GRID = np.arange(21, dtype=float) / 20.
STATE_SIZE = 5
PARAM_ENV = 'PBARWS_ALPHA_PARAMS'

DEFAULTS = {
    'ucb_mab': {
        'c': math.sqrt(2.),
    },
    'dqn_alpha': {
        'hidden': 32,
        'lr': .001,
        'gamma': .98,
        'batch': 32,
        'warmup': 64,
        'capacity': 4096,
        'target_every': 50,
        'eps_start': 1.,
        'eps_end': .05,
        'eps_decay': 500,
        'grad_clip': 1.,
    },
    'ppo_alpha': {
        'hidden': 32,
        'lr': .0003,
        'value_lr': .001,
        'gamma': .98,
        'gae_lambda': .95,
        'rollout': 32,
        'batch': 16,
        'epochs': 4,
        'clip': .2,
        'target_kl': .03,
        'grad_clip': 1.,
    },
}

# Ordered, preregistered candidates; ties use this order.
CANDIDATES = {
    'ucb_mab': [
        {'c': .25},
        {'c': .5},
        {'c': 1.},
        {'c': math.sqrt(2.)},
        {'c': 2.},
    ],
    'dqn_alpha': [
        {'lr': lr, 'eps_decay': decay}
        for lr in [.0003, .001, .003]
        for decay in [250, 750]
    ],
    'ppo_alpha': [
        {'lr': lr, 'rollout': rollout}
        for lr in [.0001, .0003, .001]
        for rollout in [32, 64]
    ],
}
# Make the default DQN configuration a registered candidate.
DEFAULTS['dqn_alpha']['eps_decay'] = 750


def settings(variant, params=None):
    base = variant.split('__', 1)[0]
    if base not in DEFAULTS:
        raise ValueError(f'Unknown alpha baseline: {variant}')

    overrides = json.loads(os.environ.get(PARAM_ENV, '{}'))
    if not isinstance(overrides, dict):
        raise ValueError(f'{PARAM_ENV} must contain a JSON object')
    unknown = set(overrides) - set(VARIANTS)
    if unknown:
        raise ValueError(f'Unknown parameter groups: {sorted(unknown)}')

    supplied = overrides.get(base, {})
    if not isinstance(supplied, dict):
        raise ValueError(f'Parameters for {base} must be an object')
    supplied = dict(supplied)
    if params is not None:
        if not isinstance(params, dict):
            raise ValueError('params must be a dictionary')
        supplied.update(params)

    unknown = set(supplied) - set(DEFAULTS[base])
    if unknown:
        raise ValueError(f'Unknown {base} parameters: {sorted(unknown)}')

    result = dict(DEFAULTS[base])
    result.update(supplied)

    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{key} must be numeric')
        if not math.isfinite(value):
            raise ValueError(f'{key} must be finite')
        if isinstance(DEFAULTS[base][key], int):
            if int(value) != value or value < 1:
                raise ValueError(f'{key} must be a positive integer')
            result[key] = int(value)

    if base == 'ucb_mab':
        if not .25 <= result['c'] <= 2.:
            raise ValueError('c must be in [0.25, 2.0]')
        return result

    for key in ['lr', 'grad_clip']:
        if result[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not 0 < result['gamma'] < 1:
        raise ValueError('gamma must be in (0, 1)')

    if base == 'dqn_alpha':
        if not 0 <= result['eps_end'] <= result['eps_start'] <= 1:
            raise ValueError('Require 0 <= eps_end <= eps_start <= 1')
        if not result['batch'] <= result['warmup'] <= result['capacity']:
            raise ValueError('Require batch <= warmup <= capacity')
    else:
        if result['value_lr'] <= 0:
            raise ValueError('value_lr must be positive')
        if not 0 <= result['gae_lambda'] <= 1:
            raise ValueError('gae_lambda must be in [0, 1]')
        if not 0 < result['clip'] < 1:
            raise ValueError('clip must be in (0, 1)')
        if result['target_kl'] <= 0:
            raise ValueError('target_kl must be positive')
        if result['batch'] > result['rollout']:
            raise ValueError('batch must not exceed rollout')
    return result


def config_tag(variant, params=None):
    payload = {
        'version': VERSION,
        'variant': variant,
        'params': settings(variant, params),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def make_rng(seed, base):
    if seed is None:
        raise ValueError('An explicit run seed is required')
    material = f'{VERSION}:{int(seed)}:{base}'.encode()
    derived = int.from_bytes(hashlib.sha256(material).digest()[:16], 'big')
    return np.random.default_rng(derived)


def features(observation, alpha):
    # No worker identities, future events, lifespan or scenario features.
    return np.array([
        np.clip(observation['rejection_rate'], 0., 1.),
        np.clip(observation['min_battery_fraction'], 0., 1.),
        float(bool(observation['any_dead'])),
        np.clip(alpha, 0., 1.),
        float(observation['window_outcomes'] > 0),
    ], dtype=float)


def utility(observation):
    # h = 1 if dead, otherwise clip((0.30 - b) / 0.30, 0, 1).
    # q = 1 - (r_eff + h) / 2, in [0, 1].
    # r_eff = r for a nonempty 200-second window, otherwise 0.5.
    # Empty windows receive neutral service credit, not perfect service.
    # This is an endpoint proxy; overlapping windows mix action histories.
    rate = float(np.clip(observation['rejection_rate'], 0., 1.))
    if observation['window_outcomes'] == 0:
        rate = .5
    battery = float(observation['min_battery_fraction'])
    risk = (1. if observation['any_dead']
            else float(np.clip((.30 - battery) / .30, 0., 1.)))
    return 1. - .5 * (rate + risk)


class MLP:
    """One tanh hidden layer, Adam and global gradient clipping."""

    def __init__(self, inputs, hidden, outputs, rng, output_scale=1.):
        self.p = [
            rng.normal(0., 1. / math.sqrt(inputs), (inputs, hidden)),
            np.zeros(hidden),
            rng.normal(
                0., output_scale / math.sqrt(hidden), (hidden, outputs)),
            np.zeros(outputs),
        ]
        self.m = [np.zeros_like(p) for p in self.p]
        self.v = [np.zeros_like(p) for p in self.p]
        self.steps = 0

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        hidden = np.tanh(x @ self.p[0] + self.p[1])
        return hidden @ self.p[2] + self.p[3]

    def update(self, x, output_gradient, lr, grad_clip):
        x = np.asarray(x, dtype=float)
        hidden = np.tanh(x @ self.p[0] + self.p[1])
        hidden_gradient = (
            output_gradient @ self.p[2].T) * (1. - hidden * hidden)
        gradients = [
            x.T @ hidden_gradient,
            hidden_gradient.sum(axis=0),
            hidden.T @ output_gradient,
            output_gradient.sum(axis=0),
        ]
        if not all(np.isfinite(g).all() for g in gradients):
            raise FloatingPointError('Non-finite network gradient')

        norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients))
        scale = min(1., grad_clip / max(norm, 1e-12))
        self.steps += 1
        for i, gradient in enumerate(gradients):
            gradient = gradient * scale
            self.m[i] = .9 * self.m[i] + .1 * gradient
            self.v[i] = .999 * self.v[i] + .001 * gradient * gradient
            mean = self.m[i] / (1. - .9 ** self.steps)
            variance = self.v[i] / (1. - .999 ** self.steps)
            self.p[i] -= lr * mean / (np.sqrt(variance) + 1e-8)

        if not all(np.isfinite(p).all() for p in self.p):
            raise FloatingPointError('Non-finite network parameter')


class UCBMAB:
    def __init__(self, params, rng):
        self.p = params
        self.rng = rng
        self.counts = np.zeros(21, dtype=np.int64)
        self.means = np.zeros(21, dtype=float)
        self.transitions = 0

    def initial(self, state):
        return 7

    def learn(self, state, action, reward, next_state):
        # Each count is one completed decision window, not one job.
        self.counts[action] += 1
        self.means[action] += (
            reward - self.means[action]) / self.counts[action]
        self.transitions += 1

    def choose(self, state):
        unseen = np.flatnonzero(self.counts == 0)
        if len(unseen):
            action = int(self.rng.choice(unseen))
        else:
            index = self.means + self.p['c'] * np.sqrt(
                math.log(max(2, self.transitions)) / self.counts)
            tied = np.flatnonzero(index == index.max())
            action = int(self.rng.choice(tied))
        return float(GRID[action]), action

    def finish(self):
        pass

    def diagnostics(self):
        return {
            'transitions': self.transitions,
            'counts': self.counts.tolist(),
            'mean_rewards': self.means.tolist(),
        }


class DQNAlpha:
    def __init__(self, params, rng):
        self.p = params
        self.rng = rng
        self.online = MLP(STATE_SIZE, params['hidden'], 21, rng, .1)
        self.target = copy.deepcopy(self.online)
        self.replay = deque(maxlen=params['capacity'])
        self.transitions = 0
        self.updates = 0
        self.last_loss = None

    def initial(self, state):
        return 7

    def epsilon(self):
        fraction = min(1., self.transitions / self.p['eps_decay'])
        return (self.p['eps_start']
                + fraction * (self.p['eps_end'] - self.p['eps_start']))

    def choose(self, state):
        if self.rng.random() < self.epsilon():
            action = int(self.rng.integers(21))
        else:
            values = self.online.predict(state)
            tied = np.flatnonzero(values == values.max())
            action = int(self.rng.choice(tied))
        return float(GRID[action]), action

    def learn(self, state, action, reward, next_state):
        self.replay.append((
            state.copy(), int(action), float(reward), next_state.copy()))
        self.transitions += 1
        if len(self.replay) < self.p['warmup']:
            return

        indices = self.rng.choice(
            len(self.replay), self.p['batch'], replace=False)
        samples = [self.replay[int(i)] for i in indices]
        states = np.stack([row[0] for row in samples])
        actions = np.array([row[1] for row in samples], dtype=int)
        rewards = np.array([row[2] for row in samples])
        next_states = np.stack([row[3] for row in samples])

        # Mission end is a sampling time limit, not an observed absorbing state.
        targets = rewards + self.p['gamma'] * np.max(
            self.target.predict(next_states), axis=1)
        predicted = self.online.predict(states)
        rows = np.arange(len(samples))
        error = predicted[rows, actions] - targets

        # Huber loss with threshold 1.
        magnitude = np.abs(error)
        self.last_loss = float(np.mean(np.where(
            magnitude <= 1., .5 * error * error, magnitude - .5)))
        gradient = np.zeros_like(predicted)
        gradient[rows, actions] = np.clip(error, -1., 1.) / len(samples)
        self.online.update(
            states, gradient, self.p['lr'], self.p['grad_clip'])

        self.updates += 1
        if self.updates % self.p['target_every'] == 0:
            self.target.p = [p.copy() for p in self.online.p]

    def finish(self):
        pass

    def diagnostics(self):
        return {
            'transitions': self.transitions,
            'updates': self.updates,
            'epsilon': self.epsilon(),
            'replay_size': len(self.replay),
            'loss': self.last_loss,
        }


def sigmoid(x):
    return .5 * (1. + np.tanh(.5 * x))


def digamma(x):
    # Positive-argument recurrence plus asymptotic expansion.
    x = np.array(x, dtype=float, copy=True)
    result = np.zeros_like(x)
    while np.any(x < 8.):
        mask = x < 8.
        result = np.where(mask, result - 1. / x, result)
        x = np.where(mask, x + 1., x)
    inv = 1. / x
    inv2 = inv * inv
    return (result + np.log(x) - .5 * inv
            - inv2 / 12. + inv2 ** 2 / 120.
            - inv2 ** 3 / 252. + inv2 ** 4 / 240.)


def log_gamma(x):
    x = np.asarray(x, dtype=float)
    return np.array(
        [math.lgamma(float(v)) for v in x.ravel()]).reshape(x.shape)


def beta_parameters(logits):
    # Both shape parameters stay in (1, 20).
    # The finite range prevents numerically singular boundary policies.
    probability = sigmoid(logits)
    shape = 1. + 19. * probability
    derivative = 19. * probability * (1. - probability)
    return shape, derivative


def beta_log_prob(actions, shape):
    a, b = shape[:, 0], shape[:, 1]
    return ((a - 1.) * np.log(actions)
            + (b - 1.) * np.log1p(-actions)
            - log_gamma(a) - log_gamma(b) + log_gamma(a + b))


class PPOAlpha:
    def __init__(self, params, rng):
        self.p = params
        self.rng = rng
        self.actor = MLP(STATE_SIZE, params['hidden'], 2, rng, .01)
        self.critic = MLP(STATE_SIZE, params['hidden'], 1, rng, .1)

        # Initial Beta mean approximately 0.35; still sampled online.
        shapes = np.array([2.8, 5.2])
        probability = (shapes - 1.) / 19.
        self.actor.p[3] = np.log(probability / (1. - probability))
        self.buffer = []
        self.transitions = 0
        self.updates = 0
        self.skipped_initial = 0
        self.last_kl = None
        self.last_value_loss = None

    def initial(self, state):
        # Forced alpha=0.35 was not drawn from the behavior policy.
        return None

    def choose(self, state):
        shape, _ = beta_parameters(self.actor.predict(state[None, :]))
        action = float(self.rng.beta(shape[0, 0], shape[0, 1]))
        while not 0. < action < 1.:
            action = float(self.rng.beta(shape[0, 0], shape[0, 1]))
        log_prob = float(beta_log_prob(np.array([action]), shape)[0])
        value = float(self.critic.predict(state)[0])

        # Store the unrounded latent action and its own density.
        # The simulator's deterministic rounding is part of the environment.
        token = (action, log_prob, value)
        return action, token

    def learn(self, state, action, reward, next_state):
        if action is None:
            self.skipped_initial += 1
            return
        latent, old_log_prob, value = action
        next_value = float(self.critic.predict(next_state)[0])
        self.buffer.append((
            state.copy(), latent, old_log_prob, value,
            float(reward), next_value))
        self.transitions += 1
        if len(self.buffer) >= self.p['rollout']:
            self.update()

    def update(self):
        if not self.buffer:
            return

        states = np.stack([row[0] for row in self.buffer])
        actions = np.array([row[1] for row in self.buffer])
        old_log_probs = np.array([row[2] for row in self.buffer])
        values = np.array([row[3] for row in self.buffer])
        rewards = np.array([row[4] for row in self.buffer])
        next_values = np.array([row[5] for row in self.buffer])

        # Discount and GAE are per decision, shared with DQN's gamma.
        # Buffer boundaries bootstrap; no invented terminal reward.
        advantages = np.zeros(len(self.buffer))
        carry = 0.
        for i in reversed(range(len(self.buffer))):
            delta = (rewards[i] + self.p['gamma'] * next_values[i]
                     - values[i])
            carry = delta + self.p['gamma'] * self.p['gae_lambda'] * carry
            advantages[i] = carry
        returns = advantages + values
        if len(advantages) > 1 and advantages.std() > 1e-8:
            advantages = (
                advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stop_actor = False
        for epoch in range(self.p['epochs']):
            order = self.rng.permutation(len(states))
            for start in range(0, len(order), self.p['batch']):
                indices = order[start:start + self.p['batch']]
                x = states[indices]
                advantage = advantages[indices]
                latent = actions[indices]

                if not stop_actor:
                    shape, derivative = beta_parameters(self.actor.predict(x))
                    log_prob = beta_log_prob(latent, shape)
                    log_ratio = log_prob - old_log_probs[indices]
                    ratio = np.exp(np.clip(log_ratio, -60., 60.))
                    kl = float(np.mean(ratio - 1. - log_ratio))
                    self.last_kl = kl

                    if kl > self.p['target_kl']:
                        stop_actor = True
                    else:
                        # Gradient of -min(ratio*A, clip(ratio)*A).
                        active = np.where(
                            advantage >= 0.,
                            ratio <= 1. + self.p['clip'],
                            ratio >= 1. - self.p['clip'])
                        coefficient = (
                            -advantage * ratio * active / len(indices))
                        a, b = shape[:, 0], shape[:, 1]
                        common = digamma(a + b)
                        score = np.column_stack([
                            np.log(latent) - digamma(a) + common,
                            np.log1p(-latent) - digamma(b) + common,
                        ])
                        gradient = (
                            coefficient[:, None] * score * derivative)
                        self.actor.update(
                            x, gradient, self.p['lr'], self.p['grad_clip'])

                predicted = self.critic.predict(x)[:, 0]
                error = predicted - returns[indices]
                self.last_value_loss = float(.5 * np.mean(error * error))
                self.critic.update(
                    x, error[:, None] / len(indices),
                    self.p['value_lr'], self.p['grad_clip'])

        self.updates += 1
        self.buffer.clear()

    def finish(self):
        # Only already-observed complete transitions; no terminal sampling.
        self.update()

    def diagnostics(self):
        return {
            'transitions': self.transitions,
            'updates': self.updates,
            'buffer_size': len(self.buffer),
            'skipped_forced_initial': self.skipped_initial,
            'approx_kl': self.last_kl,
            'value_loss': self.last_value_loss,
        }


CLASSES = {
    'ucb_mab': UCBMAB,
    'dqn_alpha': DQNAlpha,
    'ppo_alpha': PPOAlpha,
}


def install(sim, variant, folder, scenario, *, seed=None,
            scale_injector=None, params=None):
    import dynamic_alpha as native
    import controller as harness

    parts = variant.split('__')
    if len(parts) > 2:
        raise ValueError(f'Invalid strategy: {variant}')
    base = parts[0]
    environment = parts[1] if len(parts) == 2 else 'w3_wired'
    if base not in CLASSES:
        raise ValueError(f'Unknown strategy: {variant}')
    if environment.startswith('w12'):
        raise NotImplementedError('w12 is deferred')
    if environment not in harness.ENVIRONMENTS:
        raise ValueError(f'Unknown environment: {environment}')
    if scenario not in harness.BEST:
        raise ValueError(f'Unknown scenario: {scenario}')
    if seed is None:
        raise ValueError('Alpha baselines require seed')

    expected_workers, wireless = harness.ENVIRONMENTS[environment]
    _ = scale_injector
    parameters = settings(variant, params)
    signature = config_tag(variant, params)
    original_factory = sim.create_dynamic_alpha_process
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
    learner = None
    last_check = 0.
    closed = False

    def emit(name, row):
        streams[name].write(
            json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')

    def factory(env, scheduler, nodes, policy, **kwargs):
        nonlocal learner
        if learner is not None:
            raise RuntimeError('Install a fresh controller for every run')
        if env.now != 0:
            raise ValueError('Controller installation requires t=0')
        if len(nodes[1:]) != expected_workers:
            raise ValueError(
                f'{variant}: expected {expected_workers} workers')
        end = float(kwargs['end_time'])
        if not math.isfinite(end) or end <= 0:
            raise ValueError('end_time must be positive and finite')

        learner = CLASSES[base](parameters, make_rng(seed, base))
        channel = (harness.WirelessTelemetry(seed, scenario)
                   if wireless else None)

        def snapshot():
            # Same raw telemetry boundary as controller.py.
            minimum, dead, lifespan = native._get_worker_stats(nodes)
            rate, outcomes = harness.readonly_rate(scheduler, env.now)
            return {
                'rejection_rate': rate,
                'window_outcomes': outcomes,
                'min_battery_fraction': minimum,
                'any_dead': bool(dead),
                'lifespan': lifespan,
                'workers': harness.state(nodes),
            }

        def read():
            if channel is not None:
                return channel.read(env.now)
            raw = snapshot()
            return dict(
                raw, sample_time=float(env.now),
                delivery_time=float(env.now), sample_age_seconds=0.), {
                    'wireless': False, 'dropped': 0, 'delayed': 0}

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
                transport = {
                    'wireless': False, 'dropped': 0, 'delayed': 0}
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
            while env.now < end:
                yield env.timeout(10)
                if env.now > end:
                    return
                observe()

        def process():
            nonlocal last_check
            native._apply_alpha(
                scheduler, policy, scheduler.get_reward_alpha(),
                .35, 'initial')
            observe()
            env.process(observer())
            observation, _ = read()
            previous_state = features(
                observation, scheduler.get_reward_alpha())
            previous_action = learner.initial(previous_state)

            while env.now < end:
                started = time.perf_counter()
                observation, _ = read()
                period = native._adaptive_period(
                    observation['min_battery_fraction'],
                    observation['any_dead'])
                before_wait = env.now
                stats['controller_seconds'] += time.perf_counter() - started
                yield env.timeout(period)
                if env.now > end:
                    return

                started = time.perf_counter()
                observation, transport = read()
                previous = scheduler.get_reward_alpha()
                current_state = features(observation, previous)
                reward = utility(observation)

                learner.learn(
                    previous_state, previous_action, reward, current_state)
                candidate, action = learner.choose(current_state)
                if not math.isfinite(candidate) or not 0 <= candidate <= 1:
                    raise FloatingPointError('Invalid alpha action')

                rate = observation['rejection_rate']
                minimum = observation['min_battery_fraction']
                dead = observation['any_dead']
                new = native._apply_alpha(
                    scheduler, policy, previous, candidate, base,
                    rate, minimum, dead, observation['lifespan'])

                # Diagnostics only; these increments never control the action.
                rl, rh, _, _, _, _ = harness.increments(
                    rate, minimum, dead, 'full')
                calculation_seconds = time.perf_counter() - started
                stats['controller_seconds'] += calculation_seconds
                stats['checks'] += 1
                stats['changes'] += int(new != previous)
                last_check = float(env.now)

                emit('checks', {
                    'time': env.now,
                    'wait_started': before_wait,
                    'period': period,
                    'would_be_period': period,
                    'rejection_rate': rate,
                    'window_outcomes': observation['window_outcomes'],
                    'min_battery_fraction': minimum,
                    'any_dead': bool(dead),
                    'previous_alpha': previous,
                    'new_alpha': new,
                    'native_load_delta': rl,
                    'native_health_delta': rh,
                    'load_delta': None,
                    'health_delta': None,
                    'raw_delta': candidate - previous,
                    'requested_delta': candidate - previous,
                    'applied_delta': new - previous,
                    'at_boundary': new in (0., 1.),
                    'calculation_seconds': calculation_seconds,
                    'sample_time': observation['sample_time'],
                    'delivery_time': observation['delivery_time'],
                    'sample_age_seconds': (
                        float(env.now) - observation['sample_time']),
                    **transport,
                    'workers': observation['workers'],
                    'reward': reward,
                    'reward_for_alpha': previous,
                    'reward_window_start': before_wait,
                    'reward_window_end': float(env.now),
                    'reward_observation_window_seconds': 200,
                    'state': current_state.tolist(),
                    'policy_action': candidate,
                    'learner': learner.diagnostics(),
                })
                previous_state = current_state
                previous_action = action

        return env.process(process())

    sim.create_dynamic_alpha_process = factory

    def close():
        nonlocal closed
        if closed:
            return stats
        closed = True
        sim.create_dynamic_alpha_process = original_factory
        try:
            if learner is not None:
                started = time.perf_counter()
                learner.finish()
                stats['controller_seconds'] += time.perf_counter() - started
            # Keep summary keys numeric for existing aggregate CSV handling.
            (folder / 'control-summary.json').write_text(
                json.dumps(stats, indent=2, allow_nan=False))
            (folder / 'alpha-baseline.json').write_text(json.dumps({
                'version': VERSION,
                'variant': variant,
                'seed': int(seed),
                'parameters': parameters,
                'config_tag': signature,
                'state_features': [
                    'rejection_rate', 'min_battery_fraction', 'any_dead',
                    'current_alpha', 'nonempty_window'],
                'reward': '1 - (r_eff + h) / 2',
                'discount_unit': 'decision',
                'last_completed_check': last_check,
                'terminal_handling': (
                    'time-limit bootstrap; no additional tail observation'),
                'learner': learner.diagnostics() if learner else None,
            }, indent=2, allow_nan=False))
        finally:
            for stream in streams.values():
                stream.close()
        return stats

    return close
