"""The batched trainer env must agree with the PettingZoo reference env.

Two implementations of the same physics invite silent drift, so this pins them
together. If they diverge, the training results stop meaning what the reference
env says they mean, and that must fail loudly rather than quietly.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegis.envs import EnvConfig, FlutterSuppressionEnv
from aegis.envs.batched import BatchedFlutterEnv

FIXED = dict(
    speed_ratio_range=(1.1, 1.1),
    tip_plunge_range=(0.05, 0.05),
    tip_twist_range=(0.0, 0.0),
    episode_duration=0.4,
)


def _reference_rollout(config: EnvConfig, actions: np.ndarray):
    """Roll the PettingZoo env with a fixed action sequence."""
    env = FlutterSuppressionEnv(config)
    env.reset(seed=0)
    energies, deflections = [], []
    for step in range(actions.shape[0]):
        env.step({a: actions[step, i : i + 1] for i, a in enumerate(env.agents)})
        if not env.agents:
            break
        energies.append(env.sim.structural_energy)
        deflections.append(env.sim.deflection.copy())
    return np.asarray(energies), np.asarray(deflections)


def _batched_rollout(config: EnvConfig, actions: np.ndarray):
    env = BatchedFlutterEnv(config, n_envs=1, seed=0)
    energies, deflections = [], []
    for step in range(actions.shape[0]):
        _, _, dones, step_info = env.step(actions[step][None, :])
        if dones[0]:
            break
        energies.append(step_info["energy"][0])
        deflections.append(env.deflection[0].copy())
    return np.asarray(energies), np.asarray(deflections)


def test_batched_matches_reference_trajectory() -> None:
    """Same initial condition and same actions must give the same physics."""
    config = EnvConfig(**FIXED)
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1.0, 1.0, size=(60, 3))

    ref_energy, ref_deflection = _reference_rollout(config, actions)
    batch_energy, batch_deflection = _batched_rollout(config, actions)

    length = min(ref_energy.size, batch_energy.size)
    assert length > 40
    assert ref_energy[:length] == pytest.approx(batch_energy[:length], rel=1e-9)
    assert ref_deflection[:length] == pytest.approx(
        batch_deflection[:length], rel=1e-9, abs=1e-12
    )


def test_batched_matches_reference_control_power() -> None:
    """The Proposition 1 credit signal must be identical in both implementations."""
    config = EnvConfig(**FIXED)
    reference = FlutterSuppressionEnv(config)
    reference.reset(seed=0)
    batched = BatchedFlutterEnv(config, n_envs=1, seed=0)

    action = np.asarray([0.4, -0.7, 0.9])
    for _ in range(25):
        reference.step({a: action[i : i + 1] for i, a in enumerate(reference.agents)})
        batched.step(action[None, :])
    assert reference.sim.control_power() == pytest.approx(
        batched.control_power()[0], rel=1e-9
    )


def test_batched_shapes_and_autoreset() -> None:
    env = BatchedFlutterEnv(EnvConfig(episode_duration=0.05), n_envs=8, seed=1)
    observations = env.reset()
    assert observations.shape == (8, 3, env.obs_dim)

    seen_done = False
    for _ in range(40):
        observations, rewards, dones, _ = env.step(np.zeros((8, 3)))
        assert observations.shape == (8, 3, env.obs_dim)
        assert rewards.shape == (8, 3)
        assert dones.shape == (8,)
        assert np.all(np.isfinite(observations))
        seen_done |= bool(dones.any())
    assert seen_done, "episodes never terminated, so auto-reset was never exercised"


def test_environments_run_at_different_airspeeds() -> None:
    env = BatchedFlutterEnv(
        EnvConfig(speed_ratio_range=(1.0, 1.6), episode_duration=0.5), n_envs=32, seed=2
    )
    env.reset()
    assert np.std(env.airspeed) > 1.0


@pytest.mark.parametrize("comm_mode", ["none", "neighbor_local", "modal_oracle"])
def test_batched_observation_width_matches_reference(comm_mode: str) -> None:
    config = EnvConfig(comm_mode=comm_mode, **FIXED)
    reference = FlutterSuppressionEnv(config)
    batched = BatchedFlutterEnv(config, n_envs=2, seed=3)
    expected = reference.observation_space(reference.possible_agents[0]).shape[0]
    assert batched.obs_dim == expected


def test_jam_holds_in_batched_env() -> None:
    """Jammed surfaces never move, checked against the live mask each step.

    Episodes auto-reset, and a reset redraws which surface jams -- so a mask
    captured once goes stale. The invariant has to be checked against the
    current mask, not a snapshot.
    """
    env = BatchedFlutterEnv(
        EnvConfig(jam_probability=1.0, jam_angle_range=(0.1, 0.1), episode_duration=0.3),
        n_envs=4,
        seed=4,
    )
    env.reset()
    for _ in range(40):
        env.step(np.ones((4, 3)))
        mask = env._jam_mask
        assert np.all(np.abs(env.deflection[mask] - env._jam_value[mask]) < 1e-12)
        assert mask.sum(axis=1).max() <= 1
