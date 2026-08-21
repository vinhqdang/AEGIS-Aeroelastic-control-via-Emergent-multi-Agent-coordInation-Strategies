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
* **Updates run on sequences, not on shuffled timesteps.** The first version
  stored the hidden state per step and re-ran the policy on individual samples.
  That trains the recurrence as a one-step map: no gradient flows through time,
  so the phasor encoder cannot learn to integrate local accelerations into a
  modal estimate, which is the entire purpose of it. Minibatches are therefore
  segments of ``bptt_length`` consecutive steps, unrolled with gradient from a
  detached stored hidden state, with the hidden state zeroed inside the segment
  wherever an episode ended.
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
    # Truncated BPTT segment length. The recurrent encoder is trained by
    # unrolling this many steps with gradient from the stored hidden state.
    # Sampling independent timesteps instead gives the recurrence no temporal
    # gradient at all, so it can only ever learn a one-step map.
    bptt_length: int = 32
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
    # Weight on the auxiliary phasor-regression loss. Zero disables it. The
    # encoder is asked to reproduce the true modal phasor from local history,
    # which is the observer the policy otherwise has to discover by trial and
    # error through the policy gradient alone.
    aux_coefficient: float = 0.0
    # Initial exploration standard deviation, in normalised action units.
    # Scale matters here more than usual: a controller that achieves -5.5 1/s on
    # this plant uses about 0.7% of available travel, i.e. 0.1 degrees RMS, while
    # sigma = 0.30 corresponds to 4.5 degrees. The exploration noise is then some
    # 40x the useful control amplitude, so the policy never experiences the fine
    # phased actuation that works, whatever its credit signal or architecture.
    initial_log_std: float = -1.2
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Fraction of training over which the task ramps from easy to full. 0
    # disables the curriculum.
    curriculum_fraction: float = 0.4


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
        with torch.no_grad():
            self.policy.log_std.fill_(self.config.initial_log_std)
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
            if config.curriculum_fraction > 0.0:
                progress_fraction = (update + 1) / max(
                    1, int(config.curriculum_fraction * n_updates)
                )
                self.env.set_task_difficulty(progress_fraction)
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

        observations = torch.zeros(
            length, n_envs, n_agents, self.spec.obs_dim, device=self.device
        )
        hiddens = torch.zeros(length, *hidden.shape, device=self.device)
        actions = torch.zeros(length, n_envs, n_agents, device=self.device)
        log_probs = torch.zeros(length, n_envs, n_agents, device=self.device)
        values = torch.zeros(length, n_envs, n_agents, device=self.device)
        rewards = torch.zeros(length, n_envs, n_agents, device=self.device)
        dones = torch.zeros(length, n_envs, device=self.device)
        central = None
        if self.spec.central_state_dim is not None:
            central = torch.zeros(
                length, n_envs, self.spec.central_state_dim, device=self.device
            )

        phasor_target = None
        if self.config.aux_coefficient > 0.0 and self.spec.use_phasor_consensus:
            phasor_target = torch.zeros(
                length, n_envs, 2 * self.spec.n_retained_modes, device=self.device
            )

        diverged_count, energy_sum = 0, 0.0
        for step in range(length):
            central_state = self._central_state() if central is not None else None
            mean, _, value, next_hidden = self.policy(observation, hidden, central_state)
            distribution = self.policy.distribution(mean)
            action = distribution.sample()

            observations[step] = observation
            hiddens[step] = hidden
            if phasor_target is not None:
                phasor_target[step] = torch.as_tensor(
                    self.env.modal_phasor_target(),
                    device=self.device,
                    dtype=torch.float32,
                )
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
        # Time-major throughout: the optimiser slices consecutive segments, so
        # flattening here would destroy the sequence structure it needs.
        batch = {
            "observations": observations,
            "hiddens": hiddens,
            "actions": actions,
            "log_probs": log_probs,
            "advantages": advantages,
            "returns": returns,
            "dones": dones,
            "central": central,
            "phasor_target": phasor_target,
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
        """PPO update over segments of consecutive steps, with BPTT."""
        config = self.config
        length, n_envs = batch["actions"].shape[0], batch["actions"].shape[1]
        segment = min(config.bptt_length, length)
        n_segments = length // segment

        advantages = batch["advantages"]
        normalised = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pairs = np.array(
            [(seg, env) for seg in range(n_segments) for env in range(n_envs)], dtype=int
        )
        batch_size = max(1, pairs.shape[0] // config.minibatches)

        policy_losses, value_losses, entropies = [], [], []
        for _ in range(config.epochs):
            np.random.shuffle(pairs)
            for offset in range(0, pairs.shape[0], batch_size):
                losses = self._segment_update(
                    batch, normalised, pairs[offset : offset + batch_size], segment
                )
                policy_losses.append(losses[0])
                value_losses.append(losses[1])
                entropies.append(losses[2])

        if not policy_losses:
            return 0.0, 0.0, 0.0
        return (
            float(np.mean(policy_losses)),
            float(np.mean(value_losses)),
            float(np.mean(entropies)),
        )

    def _segment_update(
        self, batch: dict, normalised: torch.Tensor, chunk: np.ndarray, segment: int
    ) -> tuple[float, float, float]:
        """One PPO step on a minibatch of (segment, environment) sequences."""
        config = self.config
        starts = torch.as_tensor(chunk[:, 0] * segment, device=self.device)
        envs = torch.as_tensor(chunk[:, 1], device=self.device)

        # Detached hidden state at the segment start: truncated BPTT.
        hidden = batch["hiddens"][starts, envs].detach()

        log_probs, values, entropy_terms, aux_terms = [], [], [], []
        for offset in range(segment):
            time_index = starts + offset
            observation = batch["observations"][time_index, envs]
            central = (
                batch["central"][time_index, envs]
                if batch["central"] is not None
                else None
            )
            mean, _, value, next_hidden = self.policy(observation, hidden, central)
            distribution = self.policy.distribution(mean)
            log_probs.append(distribution.log_prob(batch["actions"][time_index, envs]))
            values.append(value)
            entropy_terms.append(distribution.entropy())
            if batch["phasor_target"] is not None:
                estimate = self.policy.encode(observation, hidden)
                target = batch["phasor_target"][time_index, envs].unsqueeze(1)
                aux_terms.append((estimate - target).pow(2).mean())
            # Zero the recurrent state where an episode ended inside the segment.
            alive = (1.0 - batch["dones"][time_index, envs]).view(-1, 1, 1)
            hidden = next_hidden * alive

        log_prob = torch.stack(log_probs)
        value = torch.stack(values)
        entropy = torch.stack(entropy_terms).mean()

        steps = torch.arange(segment, device=self.device).unsqueeze(1)
        time_index = starts.unsqueeze(0) + steps
        env_index = envs.unsqueeze(0).expand(segment, -1)
        old_log_prob = batch["log_probs"][time_index, env_index]
        advantage = normalised[time_index, env_index]
        target = batch["returns"][time_index, env_index]

        ratio = (log_prob - old_log_prob).exp()
        unclipped = ratio * advantage
        clipped = (
            ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range) * advantage
        )
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = (value - target).pow(2).mean()

        aux_loss = (
            torch.stack(aux_terms).mean()
            if aux_terms
            else torch.zeros((), device=self.device)
        )
        loss = (
            policy_loss
            + config.value_coefficient * value_loss
            - config.entropy_coefficient * entropy
            + config.aux_coefficient * aux_loss
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config.max_grad_norm)
        self.optimizer.step()
        return float(policy_loss), float(value_loss), float(entropy)

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
