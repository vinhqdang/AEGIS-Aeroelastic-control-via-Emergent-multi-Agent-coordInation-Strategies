"""PettingZoo API conformance and semantics for the AEGIS environment."""

from __future__ import annotations

import numpy as np
import pytest

from aegis.envs import COMM_MODES, REWARD_MODES, EnvConfig, FlutterSuppressionEnv

FAST = dict(episode_duration=0.2, speed_ratio_range=(1.1, 1.1))


@pytest.fixture(scope="module")
def env() -> FlutterSuppressionEnv:
    return FlutterSuppressionEnv(EnvConfig(**FAST))


def _random_actions(env: FlutterSuppressionEnv, rng) -> dict:
    return {a: rng.uniform(-1.0, 1.0, size=(1,)) for a in env.agents}


def test_passes_pettingzoo_parallel_api_test() -> None:
    from pettingzoo.test import parallel_api_test

    parallel_api_test(FlutterSuppressionEnv(EnvConfig(**FAST)), num_cycles=40)


def test_reset_returns_one_observation_per_agent(env: FlutterSuppressionEnv) -> None:
    observations, infos = env.reset(seed=0)
    assert set(observations) == set(env.possible_agents)
    assert set(infos) == set(env.possible_agents)
    for agent, observation in observations.items():
        assert env.observation_space(agent).shape == observation.shape
        assert np.all(np.isfinite(observation))


def test_episode_truncates_at_the_configured_duration(env: FlutterSuppressionEnv) -> None:
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    steps = 0
    while env.agents:
        _, _, terminations, truncations, _ = env.step(_random_actions(env, rng))
        steps += 1
        assert steps <= 1000
    assert steps == round(FAST["episode_duration"] / env.config.control_dt)
    assert not any(terminations.values())
    assert all(truncations.values())


@pytest.mark.parametrize("comm_mode", COMM_MODES)
def test_observation_size_matches_communication_mode(comm_mode: str) -> None:
    env = FlutterSuppressionEnv(EnvConfig(comm_mode=comm_mode, **FAST))
    observations, _ = env.reset(seed=2)
    expected = env.observation_space(env.possible_agents[0]).shape[0]
    assert all(o.shape == (expected,) for o in observations.values())
    if comm_mode == "none":
        assert expected == 7


@pytest.mark.parametrize("reward_mode", REWARD_MODES)
def test_every_reward_mode_produces_finite_rewards(reward_mode: str) -> None:
    env = FlutterSuppressionEnv(EnvConfig(reward_mode=reward_mode, **FAST))
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    while env.agents:
        _, rewards, _, _, _ = env.step(_random_actions(env, rng))
        assert all(np.isfinite(r) for r in rewards.values())


def test_shared_reward_is_identical_across_agents() -> None:
    """The naive baseline must actually be uninformative about who did what."""
    env = FlutterSuppressionEnv(EnvConfig(reward_mode="shared", effort_penalty=0.0,
                                          rate_penalty=0.0, **FAST))
    env.reset(seed=4)
    _, rewards, _, _, _ = env.step({a: np.asarray([0.5]) for a in env.agents})
    values = list(rewards.values())
    assert values == pytest.approx([values[0]] * len(values))


def test_physics_reward_distinguishes_agents() -> None:
    """The point of Proposition 1: identical actions earn different credit."""
    env = FlutterSuppressionEnv(EnvConfig(reward_mode="physics", effort_penalty=0.0,
                                          rate_penalty=0.0, **FAST))
    env.reset(seed=5)
    for _ in range(6):
        _, rewards, _, _, _ = env.step({a: np.asarray([0.6]) for a in env.agents})
    values = np.asarray(list(rewards.values()))
    assert values.std() > 1e-6, "physics credit collapsed to a shared signal"


def test_observations_stay_local() -> None:
    """With comm off, an agent's observation must not encode the global mode.

    Perturbing a distant station's motion should leave an inboard agent's
    observation untouched only if that agent truly reads local data; here we
    assert the weaker, checkable property that the observation is exactly the
    documented local vector, with no extra channels.
    """
    env = FlutterSuppressionEnv(EnvConfig(comm_mode="none", **FAST))
    observations, _ = env.reset(seed=6)
    assert all(o.shape == (7,) for o in observations.values())


def test_jam_is_reported_in_info() -> None:
    env = FlutterSuppressionEnv(
        EnvConfig(jam_probability=1.0, jam_angle_range=(0.12, 0.12), **FAST)
    )
    _, infos = env.reset(seed=7)
    jammed = next(iter(infos.values()))["jammed"]
    assert len(jammed) == 1


def test_jammed_surface_does_not_move() -> None:
    env = FlutterSuppressionEnv(
        EnvConfig(jam_probability=1.0, jam_angle_range=(0.12, 0.12), **FAST)
    )
    _, infos = env.reset(seed=8)
    jammed = next(iter(infos.values()))["jammed"][0]
    index = env.possible_agents.index(jammed)
    for _ in range(10):
        env.step({a: np.asarray([1.0]) for a in env.agents})
    assert env.sim.deflection[index] == pytest.approx(0.12)


def test_airspeed_is_randomised_across_episodes() -> None:
    env = FlutterSuppressionEnv(EnvConfig(episode_duration=0.05,
                                          speed_ratio_range=(1.0, 1.6)))
    speeds = []
    for seed in range(8):
        _, infos = env.reset(seed=seed)
        speeds.append(next(iter(infos.values()))["speed_ratio"])
    assert np.std(speeds) > 0.05


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="comm_mode must be one of"):
        EnvConfig(comm_mode="telepathy")
    with pytest.raises(ValueError, match="reward_mode must be one of"):
        EnvConfig(reward_mode="vibes")
    with pytest.raises(ValueError, match="physics_weight"):
        EnvConfig(physics_weight=1.5)


def test_step_before_reset_is_an_error() -> None:
    env = FlutterSuppressionEnv(EnvConfig(**FAST))
    with pytest.raises(RuntimeError, match=r"reset\(\) must be called"):
        env.step({})


def test_turbulence_runs_end_to_end() -> None:
    env = FlutterSuppressionEnv(
        EnvConfig(turbulence_intensity=2.0, episode_duration=0.15,
                  speed_ratio_range=(1.1, 1.1))
    )
    env.reset(seed=9)
    rng = np.random.default_rng(9)
    while env.agents:
        _, rewards, _, _, _ = env.step(_random_actions(env, rng))
        assert all(np.isfinite(r) for r in rewards.values())
