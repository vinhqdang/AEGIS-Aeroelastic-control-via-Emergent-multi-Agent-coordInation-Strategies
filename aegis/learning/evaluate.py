"""Reward-independent evaluation: the only way to compare these controllers.

Different reward modes have different scales, so PPO return says nothing across
runs -- a physics-credit policy and a shared-reward policy simply cannot be
ranked by their own reward. Everything is therefore scored on the same physical
quantities:

* **divergence rate** -- fraction of episodes that leave the linear regime. The
  headline safety metric.
* **suppression rate** -- exponential decay rate of the structural energy, 1/s.
  Negative is good; this is the same statistic the classical analysis reports.
* **peak tip plunge** -- how far the wing actually moved.
* **RMS deflection** -- what it cost in control authority.

Every controller is evaluated on *identical* episodes: same airspeeds, same
initial deformation, same turbulence, same injected failure. Sampling conditions
independently per controller would confound the comparison with luck.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from aegis.envs.batched import BatchedFlutterEnv
from aegis.envs.flutter_marl import EnvConfig


class BatchedController(Protocol):
    """Anything that maps a batched environment and observation to actions."""

    def reset(self, n_envs: int) -> None: ...
    def __call__(self, env: BatchedFlutterEnv, observation: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class EvalCondition:
    """One evaluation cell: a flight condition plus an optional failure."""

    speed_ratio: float
    jam_surface: int = -1          # agent index, or -1 for none
    jam_angle: float = 0.0         # rad
    label: str = ""

    def describe(self) -> str:
        if self.label:
            return self.label
        if self.jam_surface < 0:
            return f"U={self.speed_ratio:.2f}Uf healthy"
        return (
            f"U={self.speed_ratio:.2f}Uf jam#{self.jam_surface}"
            f"@{np.rad2deg(self.jam_angle):.0f}deg"
        )


@dataclass(frozen=True)
class EvalResult:
    """Metrics for one controller on one condition."""

    condition: EvalCondition
    divergence_rate: float
    suppression_rate: float
    suppression_rate_std: float
    peak_tip_plunge: float
    rms_deflection: float
    n_episodes: int

    @property
    def stabilised(self) -> bool:
        """No divergences and the energy genuinely decayed."""
        return self.divergence_rate == 0.0 and self.suppression_rate < 0.0


class LearnedController:
    """Wraps a trained policy so it satisfies :class:`BatchedController`."""

    def __init__(self, policy, device: torch.device, deterministic: bool = True):
        self.policy = policy
        self.device = device
        self.deterministic = deterministic
        self._hidden: torch.Tensor | None = None
        self.name = "learned"

    def reset(self, n_envs: int) -> None:
        self._hidden = self.policy.initial_hidden(n_envs, self.device)

    @torch.no_grad()
    def __call__(self, env: BatchedFlutterEnv, observation: np.ndarray) -> np.ndarray:
        if self._hidden is None:
            self.reset(env.n_envs)
        tensor = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
        central = None
        if self.policy.spec.central_state_dim is not None:
            central = _central_state(env, self.device)
        mean, _, _, hidden = self.policy(tensor, self._hidden, central)
        self._hidden = hidden
        action = mean if self.deterministic else self.policy.distribution(mean).sample()
        return action.clamp(-1.0, 1.0).cpu().numpy()


def _central_state(env: BatchedFlutterEnv, device: torch.device) -> torch.Tensor:
    modal = env.plant_state[:, : 2 * env.n_modes]
    extras = np.concatenate(
        [env.deflection, (env.airspeed / env.flutter_speed - 1.0)[:, None]], axis=-1
    )
    return torch.as_tensor(
        np.concatenate([modal, extras], axis=-1), device=device, dtype=torch.float32
    )


def evaluate(
    controller: BatchedController,
    config: EnvConfig,
    conditions: list[EvalCondition],
    n_episodes: int = 64,
    initial_tip_plunge: float = 0.05,
    seed: int = 12345,
) -> list[EvalResult]:
    """Score ``controller`` on every condition, on matched episodes."""
    results = []
    for index, condition in enumerate(conditions):
        env = BatchedFlutterEnv(config, n_envs=n_episodes, seed=seed + index, auto_reset=False)
        speeds = np.full(n_episodes, env.flutter_speed * condition.speed_ratio)
        # Alternate the sign of the initial pluck so the set is symmetric and a
        # policy cannot score well by being biased in one direction.
        signs = np.where(np.arange(n_episodes) % 2 == 0, 1.0, -1.0)
        plunge = signs * initial_tip_plunge
        jam = np.full(n_episodes, condition.jam_surface)
        angle = np.full(n_episodes, condition.jam_angle)

        observation = env.reset_to(
            speeds, plunge, jam_surface=jam, jam_angle=angle, gust_seed=seed + index
        )
        controller.reset(n_episodes)

        n_steps = round(config.episode_duration / config.control_dt)
        energies = np.zeros((n_steps, n_episodes))
        tips = np.zeros((n_steps, n_episodes))
        deflections = np.zeros((n_steps, n_episodes, env.n_agents))
        diverged = np.zeros(n_episodes, dtype=bool)
        alive = np.ones((n_steps, n_episodes), dtype=bool)

        tip_row = env.model.tip_rows()[0]
        for step in range(n_steps):
            action = controller(env, observation)
            observation, _, dones, info = env.step(action)
            energies[step] = info["energy"]
            tips[step] = env.plant_state[:, : env.n_modes] @ tip_row
            deflections[step] = env.deflection
            alive[step] = ~info["finished"] | dones
            diverged |= info["diverged"]

        results.append(
            _summarise(condition, energies, tips, deflections, diverged, alive, config)
        )
    return results


def _summarise(
    condition: EvalCondition,
    energies: np.ndarray,
    tips: np.ndarray,
    deflections: np.ndarray,
    diverged: np.ndarray,
    alive: np.ndarray,
    config: EnvConfig,
) -> EvalResult:
    n_steps, n_episodes = energies.shape
    time = np.arange(n_steps) * config.control_dt

    rates = np.full(n_episodes, np.nan)
    for episode in range(n_episodes):
        valid = alive[:, episode] & (energies[:, episode] > 0.0)
        if valid.sum() < 5:
            continue
        slope = np.polyfit(time[valid], np.log(energies[valid, episode]), 1)[0]
        rates[episode] = 0.5 * slope  # energy ~ amplitude^2

    finite = np.isfinite(rates)
    return EvalResult(
        condition=condition,
        divergence_rate=float(diverged.mean()),
        suppression_rate=float(np.mean(rates[finite])) if finite.any() else float("nan"),
        suppression_rate_std=float(np.std(rates[finite])) if finite.any() else float("nan"),
        peak_tip_plunge=float(np.abs(tips).max()),
        rms_deflection=float(np.sqrt(np.mean(deflections**2))),
        n_episodes=n_episodes,
    )


def standard_conditions(
    speed_ratios: tuple[float, ...] = (1.05, 1.25, 1.45),
    jam_angles_deg: tuple[float, ...] = (0.0, 8.0, 14.0),
    jam_surface: int = 2,
) -> list[EvalCondition]:
    """The evaluation grid the thesis needs: envelope crossed with failures."""
    conditions = []
    for ratio in speed_ratios:
        for angle in jam_angles_deg:
            conditions.append(
                EvalCondition(
                    speed_ratio=ratio,
                    jam_surface=jam_surface if angle > 0.0 else -1,
                    jam_angle=np.deg2rad(angle),
                )
            )
    return conditions


def summary_table(results: dict[str, list[EvalResult]]) -> str:
    """Render a controller-by-condition comparison as text."""
    reference = next(iter(results.values()))
    header = f"{'condition':>30}  " + "  ".join(f"{name:>16}" for name in results)
    lines = [header, "-" * len(header)]
    for index, entry in enumerate(reference):
        cells = []
        for name in results:
            item = results[name][index]
            if item.divergence_rate > 0.0:
                cells.append(f"{'DIV ' + f'{item.divergence_rate * 100:.0f}%':>16}")
            else:
                cells.append(f"{item.suppression_rate:>16.3f}")
        lines.append(f"{entry.condition.describe():>30}  " + "  ".join(cells))
    lines.append("")
    lines.append("cells: energy suppression rate [1/s], negative is better; DIV = diverged")
    return "\n".join(lines)
