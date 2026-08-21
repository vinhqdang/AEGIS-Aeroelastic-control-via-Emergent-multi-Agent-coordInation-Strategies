"""Multi-agent PPO for AEGIS.

Parameters are shared across agents with the agent index as a one-hot input, so
one policy covers any number of surfaces. Every (environment, agent) pair is a
row in the batch.

Two details that matter for correctness rather than performance:

* **Recurrent state and auto-reset.** The phasor encoder is recurrent, so the
  hidden state must be zeroed exactly where an episode ended. The batched env
  auto-resets, so the done flag from step ``t`` applies to the hidden state
  entering step ``t+1``.
* **Advantages are computed per agent.** With the physics credit each agent gets
  a different reward, so a single shared advantage would throw away exactly the
  signal the method is about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from aegis.envs.batched import BatchedFlutterEnv
from aegis.learning.networks import MultiAgentPolicy, PolicySpec


@dataclass
class PPOConfig:
    """Hyperparameters. Defaults are tuned for this plant, not copied blindly."""

    total_steps: int = 1_000_000
    n_envs: int = 128
    rollout_length: int = 128
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 3.0e-4
    gamma: float = 0.995          # ~1 s horizon at 200 Hz, ~11 flutter cycles.
    # Energy decay is a long-horizon behaviour: at 0.985 the horizon covered
    # under four flutter cycles and the policy learned to hold the wing
    # bounded rather than to damp it.
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 2.0e-3
    max_grad_norm: float = 0.5
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingLog:
    """Per-update training diagnostics."""

    steps: list[int] = field(default_factory=list)
    mean_return: list[float] = field(default_factory=list)
    divergence_rate: list[float] = field(default_factory=list)
    mean_energy: list[float] = field(default_factory=list)
    policy_loss: list[float] = field(default_factory=list)
    value_loss: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, list]:
        return {
            "steps": self.steps,
            "mean_return": self.mean_return,
            "divergence_rate": self.divergence_rate,
            "mean_energy": self.mean_energy,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
        }


class PPOTrainer:
    """On-policy trainer over a :class:`BatchedFlutterEnv`."""

    def __init__(
        self,
        env: BatchedFlutterEnv,
        spec: PolicySpec,
        config: PPOConfig | None = None,
    ):
        self.env = env
        self.config = config if config is not None else PPOConfig()
        torch.manual_seed(self.config.seed)

        self.device = torch.device(self.config.device)
        self.policy = MultiAgentPolicy(spec).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate, eps=1e-5
        )
        self.spec = spec
        self.log = TrainingLog()
        self._recent_returns: list[float] = []

    # ------------------------------------------------------------------ train
    def train(self, progress: bool = True) -> TrainingLog:
        config = self.config
        n_envs, n_agents = self.env.n_envs, self.env.n_agents
        steps_per_update = config.rollout_length * n_envs
        n_updates = max(1, config.total_steps // steps_per_update)

        observation = torch.as_tensor(self.env.reset(), device=self.device)
        hidden = self.policy.initial_hidden(n_envs, self.device)

        for update in range(n_updates):
            batch, observation, hidden = self._collect(observation, hidden)
            losses = self._optimise(batch)

            steps_done = (update + 1) * steps_per_update
            self.log.steps.append(steps_done)
            self.log.mean_return.append(
                float(np.mean(self._recent_returns[-200:])) if self._recent_returns else 0.0
            )
            self.log.divergence_rate.append(batch["divergence_rate"])
            self.log.mean_energy.append(batch["mean_energy"])
            self.log.policy_loss.append(losses[0])
            self.log.value_loss.append(losses[1])
            self.log.entropy.append(losses[2])

            if progress and (update % max(1, n_updates // 10) == 0 or update == n_updates - 1):
                print(
                    f"    update {update + 1:4d}/{n_updates}  steps {steps_done:>9,}"
                    f"  return {self.log.mean_return[-1]:+9.3f}"
                    f"  divergence {batch['divergence_rate'] * 100:5.1f}%"
                    f"  energy {batch['mean_energy']:.3e}",
                    flush=True,
                )
        del n_agents
        return self.log

    # ---------------------------------------------------------------- rollout
    @torch.no_grad()
    def _collect(
        self, observation: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        config = self.config
        n_envs, n_agents = self.env.n_envs, self.env.n_agents
        length = config.rollout_length

        observations = torch.zeros(length, n_envs, n_agents, self.spec.obs_dim, device=self.device)
        hiddens = torch.zeros(length, *hidden.shape, device=self.device)
        actions = torch.zeros(length, n_envs, n_agents, device=self.device)
        log_probs = torch.zeros(length, n_envs, n_agents, device=self.device)
        values = torch.zeros(length, n_envs, n_agents, device=self.device)
        rewards = torch.zeros(length, n_envs, n_agents, device=self.device)
        dones = torch.zeros(length, n_envs, device=self.device)
        central = None
        if self.spec.central_state_dim is not None:
            central = torch.zeros(length, n_envs, self.spec.central_state_dim, device=self.device)

        diverged_count, energy_sum = 0, 0.0
        for step in range(length):
            central_state = self._central_state() if central is not None else None
            mean, _, value, next_hidden = self.policy(observation, hidden, central_state)
            distribution = self.policy.distribution(mean)
            action = distribution.sample()

            observations[step] = observation
            hiddens[step] = hidden
            actions[step] = action
            log_probs[step] = distribution.log_prob(action)
            values[step] = value
            if central is not None:
                central[step] = central_state

            next_observation, reward, done, info = self.env.step(
                action.clamp(-1.0, 1.0).cpu().numpy()
            )
            rewards[step] = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
            dones[step] = torch.as_tensor(done, device=self.device, dtype=torch.float32)

            diverged_count += int(info["diverged"].sum())
            energy_sum += float(info["energy"].mean())
            if np.any(done):
                self._recent_returns.extend(info["episode_return"][done].tolist())

            observation = torch.as_tensor(next_observation, device=self.device)
            # The env auto-resets, so a finished episode must not carry its
            # recurrent state into the next one.
            hidden = next_hidden * (1.0 - dones[step]).view(-1, 1, 1)

        with torch.no_grad():
            central_state = self._central_state() if central is not None else None
            _, _, last_value, _ = self.policy(observation, hidden, central_state)

        advantages, returns = self._gae(rewards, values, dones, last_value)
        batch = {
            "observations": observations.reshape(-1, n_agents, self.spec.obs_dim),
            "hiddens": hiddens.reshape(-1, *hidden.shape[1:]),
            "actions": actions.reshape(-1, n_agents),
            "log_probs": log_probs.reshape(-1, n_agents),
            "advantages": advantages.reshape(-1, n_agents),
            "returns": returns.reshape(-1, n_agents),
            "central": central.reshape(-1, self.spec.central_state_dim)
            if central is not None
            else None,
            "divergence_rate": diverged_count / (length * n_envs),
            "mean_energy": energy_sum / length,
        }
        return batch, observation, hidden

    def _central_state(self) -> torch.Tensor:
        """Privileged training-only state for the centralised-critic baseline."""
        modal = self.env.plant_state[:, : 2 * self.env.n_modes]
        extras = np.concatenate(
            [self.env.deflection, (self.env.airspeed / self.env.flutter_speed - 1.0)[:, None]],
            axis=-1,
        )
        return torch.as_tensor(
            np.concatenate([modal, extras], axis=-1), device=self.device, dtype=torch.float32
        )

    def _gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        advantages = torch.zeros_like(rewards)
        running = torch.zeros_like(last_value)
        for step in reversed(range(rewards.shape[0])):
            not_done = (1.0 - dones[step]).unsqueeze(-1)
            next_value = last_value if step == rewards.shape[0] - 1 else values[step + 1]
            delta = rewards[step] + config.gamma * next_value * not_done - values[step]
            running = delta + config.gamma * config.gae_lambda * not_done * running
            advantages[step] = running
        return advantages, advantages + values

    # --------------------------------------------------------------- optimise
    def _optimise(self, batch: dict) -> tuple[float, float, float]:
        config = self.config
        n_samples = batch["observations"].shape[0]
        indices = np.arange(n_samples)
        minibatch_size = max(1, n_samples // config.minibatches)

        advantages = batch["advantages"]
        normalised = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses, value_losses, entropies = [], [], []
        for _ in range(config.epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, minibatch_size):
                slice_indices = torch.as_tensor(
                    indices[start : start + minibatch_size], device=self.device
                )
                central = (
                    batch["central"][slice_indices] if batch["central"] is not None else None
                )
                mean, _, value, _ = self.policy(
                    batch["observations"][slice_indices],
                    batch["hiddens"][slice_indices],
                    central,
                )
                distribution = self.policy.distribution(mean)
                log_prob = distribution.log_prob(batch["actions"][slice_indices])
                ratio = (log_prob - batch["log_probs"][slice_indices]).exp()

                minibatch_advantage = normalised[slice_indices]
                unclipped = ratio * minibatch_advantage
                clipped = (
                    ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range)
                    * minibatch_advantage
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (value - batch["returns"][slice_indices]).pow(2).mean()
                entropy = distribution.entropy().mean()

                loss = (
                    policy_loss
                    + config.value_coefficient * value_loss
                    - config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), config.max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(float(policy_loss))
                value_losses.append(float(value_loss))
                entropies.append(float(entropy))

        return (
            float(np.mean(policy_losses)),
            float(np.mean(value_losses)),
            float(np.mean(entropies)),
        )

    # ------------------------------------------------------------------ act
    @torch.no_grad()
    def act(
        self, observation: np.ndarray, hidden: torch.Tensor, deterministic: bool = True
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Greedy (or sampled) action for evaluation."""
        tensor = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
        central = self._central_state() if self.spec.central_state_dim is not None else None
        mean, _, _, next_hidden = self.policy(tensor, hidden, central)
        action = mean if deterministic else self.policy.distribution(mean).sample()
        return action.clamp(-1.0, 1.0).cpu().numpy(), next_hidden
