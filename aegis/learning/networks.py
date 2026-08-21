"""Policy and value networks, including the three MoCCA mechanisms.

The mechanisms from ``docs/ALGORITHM.md`` live here:

* :class:`PhasorEncoder` -- a recurrent estimator that turns one agent's local
  accelerometer history into a phasor for each retained critical mode. It has to
  be recurrent: a single instantaneous local reading cannot distinguish a mode
  moving up from the same displacement moving down.
* :class:`PhasorConsensus` -- ``T`` rounds of averaging over the span line graph.
  This is the communication channel, and its whole point is that the message is
  ``2R`` numbers regardless of how many surfaces there are.
* :class:`PhaseLockedHead` -- emits a gain and a phase *relative to the consensus
  phasor* rather than a raw deflection, plus a broadband residual channel.

The phasor is carried as a ``(cos, sin)`` pair rather than an angle. Regressing
an angle directly puts a branch cut in the middle of the output space, which
makes the wrap-around a discontinuity the network has to fight; a unit-vector
pair has no such seam.

Actions are Gaussian for PPO: whatever head is in use produces the mean, and a
state-independent log-standard-deviation is learned alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

HIDDEN = 128
ENCODER_HIDDEN = 64


@dataclass(frozen=True)
class PolicySpec:
    """Shape and architecture of a policy."""

    obs_dim: int
    n_agents: int
    n_retained_modes: int = 2
    consensus_rounds: int = 2
    consensus_weight: float = 0.5
    use_phasor_consensus: bool = False
    use_phase_locked_head: bool = False
    # Append an actuator-health channel to the consensus message. Phase alone
    # cannot tell an agent that a neighbour has stopped responding, which is the
    # information a residual policy needs in order to know when NOT to correct.
    use_health_channel: bool = False
    saturation_index: int = 5  # observation channel holding own saturation
    central_state_dim: int | None = None  # None selects a decentralised critic


#: Output-layer gain for action heads. Small enough that a fresh policy is
#: effectively a no-op, which is what makes residual control well-posed.
ACTION_OUTPUT_GAIN = 0.01


def _mlp(sizes: list[int], activation=nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def _shrink_output(module: nn.Module, gain: float = ACTION_OUTPUT_GAIN) -> nn.Module:
    """Scale down the last linear layer so the head starts near zero output."""
    last = [layer for layer in module.modules() if isinstance(layer, nn.Linear)][-1]
    with torch.no_grad():
        last.weight.mul_(gain)
        last.bias.zero_()
    return module


class PhasorEncoder(nn.Module):
    """Local recurrent estimator of the critical-mode phasors.

    Input is one agent's local observation; output is ``2R`` numbers read as
    ``R`` ``(cos, sin)`` pairs scaled by an amplitude. The recurrent state is
    what lets a single station infer phase from a history of accelerations.
    """

    def __init__(self, obs_dim: int, n_retained_modes: int):
        super().__init__()
        self.cell = nn.GRUCell(obs_dim, ENCODER_HIDDEN)
        self.readout = nn.Linear(ENCODER_HIDDEN, 2 * n_retained_modes)
        self.n_retained_modes = n_retained_modes

    def forward(
        self, observation: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``observation`` is ``(batch, obs_dim)``; ``hidden`` is ``(batch, H)``.

        The readout is squashed. An unbounded phasor multiplied by a gain inside
        the phase-locked head drives its output ``tanh`` straight into saturation
        at initialisation, which kills the gradient and is why the first
        implementation of this policy failed to learn at all.
        """
        next_hidden = self.cell(observation, hidden)
        return torch.tanh(self.readout(next_hidden)), next_hidden

    @property
    def hidden_size(self) -> int:
        return ENCODER_HIDDEN


class PhasorConsensus(nn.Module):
    """Averaging consensus over the spanwise line graph.

    Nearest-neighbour only, because that is what a distributed avionics bus
    physically is. The operator is a fixed doubly-stochastic matrix, so this adds
    no parameters -- the bandwidth cost is the message itself, and it does not
    grow with the number of agents.
    """

    def __init__(self, n_agents: int, rounds: int, weight: float):
        super().__init__()
        if not 0.0 < weight <= 1.0:
            raise ValueError(f"consensus weight must lie in (0, 1], got {weight}")
        self.rounds = rounds
        adjacency = torch.zeros(n_agents, n_agents)
        for i in range(n_agents):
            neighbours = [j for j in (i - 1, i + 1) if 0 <= j < n_agents]
            for j in neighbours:
                adjacency[i, j] = 1.0 / len(neighbours)
        operator = (1.0 - weight) * torch.eye(n_agents) + weight * adjacency
        self.register_buffer("operator", operator)

    def forward(self, messages: torch.Tensor) -> torch.Tensor:
        """``messages`` is ``(batch, n_agents, message_dim)``."""
        for _ in range(self.rounds):
            messages = torch.einsum("ij,bjd->bid", self.operator, messages)
        return messages


class PhaseLockedHead(nn.Module):
    """Emit a gain and phase relative to the consensus phasor, plus a residual.

    For a mode whose phasor is ``(a, b)``, acting with gain ``A`` at phase offset
    ``psi`` means ``A * (a cos psi - b sin psi)`` -- a scaled rotation of the
    phasor. Parameterising the action this way bakes in the resonant structure of
    flutter suppression, which is the point; the additive broadband channel is
    kept because turbulence response is not narrowband and a purely resonant
    action space would fail under it.
    """

    def __init__(self, feature_dim: int, n_retained_modes: int):
        super().__init__()
        self.n_retained_modes = n_retained_modes
        # Per mode: gain, cos(psi), sin(psi). Plus a general (broadband) channel.
        self.head = _mlp([feature_dim, HIDDEN, 3 * n_retained_modes + 1])

    def forward(self, features: torch.Tensor, phasor: torch.Tensor) -> torch.Tensor:
        """``features`` is ``(batch, F)``, ``phasor`` is ``(batch, 2R)``.

        The resonant term is added to a general action channel rather than
        replacing it, which makes this head a strict generalisation of the plain
        one: setting the gains to zero recovers an unconstrained policy. The
        first version *replaced* the general action, so the parameterisation was
        a restriction, and the resulting policy scored worse than its own
        ablation. A resonant prior should be an inductive bias, not a cage.
        """
        raw = self.head(features)
        modes = self.n_retained_modes
        gain = torch.tanh(raw[:, :modes])
        rotation = raw[:, modes : 3 * modes].reshape(-1, modes, 2)
        rotation = rotation / (rotation.norm(dim=-1, keepdim=True) + 1e-6)
        general = raw[:, -1]

        components = phasor.reshape(-1, modes, 2)
        rotated = (
            components[..., 0] * rotation[..., 0] - components[..., 1] * rotation[..., 1]
        )
        return torch.tanh(general + (gain * rotated).sum(dim=-1))


class MultiAgentPolicy(nn.Module):
    """Shared-parameter actor-critic over agents, with an agent identity input.

    Parameters are shared and the agent index is supplied as a one-hot, so the
    policy can specialise per station (it must -- the modal residue differs
    sharply along the span) without the parameter count growing with the number
    of surfaces. That is what makes the scaling study possible.
    """

    def __init__(self, spec: PolicySpec):
        super().__init__()
        self.spec = spec
        identity = spec.n_agents
        self.encoder: PhasorEncoder | None = None
        self.consensus: PhasorConsensus | None = None

        message_dim = 0
        if spec.use_phasor_consensus:
            self.encoder = PhasorEncoder(spec.obs_dim, spec.n_retained_modes)
            self.consensus = PhasorConsensus(
                spec.n_agents, spec.consensus_rounds, spec.consensus_weight
            )
            message_dim = 2 * spec.n_retained_modes
            if spec.use_health_channel:
                message_dim += 2

        feature_dim = spec.obs_dim + identity + message_dim
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh()
        )

        if spec.use_phase_locked_head:
            if not spec.use_phasor_consensus:
                raise ValueError(
                    "the phase-locked head needs a phasor; enable use_phasor_consensus"
                )
            self.action_head: nn.Module = PhaseLockedHead(HIDDEN, spec.n_retained_modes)
        else:
            self.action_head = _mlp([HIDDEN, 1])
        _shrink_output(self.action_head)

        critic_input = (
            spec.central_state_dim + identity
            if spec.central_state_dim is not None
            else feature_dim
        )
        self.critic = _mlp([critic_input, HIDDEN, HIDDEN, 1])
        # sigma = 0.30. The previous 0.50 is a very large perturbation on an
        # action space normalised to [-1, 1], and it swamps a residual channel.
        self.log_std = nn.Parameter(torch.full((1,), -1.2))
        self.register_buffer("identity", torch.eye(spec.n_agents))

    # ---------------------------------------------------------------- helpers
    def initial_hidden(self, batch: int, device: torch.device) -> torch.Tensor:
        size = self.encoder.hidden_size if self.encoder is not None else 1
        return torch.zeros(batch, self.spec.n_agents, size, device=device)

    def _features(
        self, observation: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(features, phasor, next_hidden)``, all flattened over agents."""
        batch, n_agents, _ = observation.shape
        identity = self.identity.expand(batch, n_agents, n_agents)

        phasor = torch.zeros(batch, n_agents, 0, device=observation.device)
        next_hidden = hidden
        if self.encoder is not None and self.consensus is not None:
            flat_obs = observation.reshape(batch * n_agents, -1)
            flat_hidden = hidden.reshape(batch * n_agents, -1)
            message, new_hidden = self.encoder(flat_obs, flat_hidden)
            next_hidden = new_hidden.reshape(batch, n_agents, -1)
            message = message.reshape(batch, n_agents, -1)
            if self.spec.use_health_channel:
                # Own saturation, and its magnitude, are broadcast alongside the
                # phasor so the consensus carries actuator health as well as phase.
                start = self.spec.saturation_index
                own = observation[..., start : start + 1]
                message = torch.cat([message, own, own.abs()], dim=-1)
            phasor = self.consensus(message)

        features = torch.cat([observation, identity, phasor], dim=-1)
        return (
            features.reshape(batch * n_agents, -1),
            phasor.reshape(batch * n_agents, -1),
            next_hidden,
        )

    def forward(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        central_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(mean, log_std, value, next_hidden)``.

        ``mean`` and ``value`` are shaped ``(batch, n_agents)``.
        """
        batch, n_agents, _ = observation.shape
        features, phasor, next_hidden = self._features(observation, hidden)
        latent = self.trunk(features)

        if isinstance(self.action_head, PhaseLockedHead):
            phase_part = phasor[:, : 2 * self.spec.n_retained_modes]
            mean = self.action_head(latent, phase_part)
        else:
            mean = torch.tanh(self.action_head(latent).squeeze(-1))

        if self.spec.central_state_dim is not None:
            if central_state is None:
                raise ValueError("central critic requires central_state")
            identity = self.identity.expand(batch, n_agents, n_agents)
            critic_input = torch.cat(
                [central_state.unsqueeze(1).expand(batch, n_agents, -1), identity], dim=-1
            ).reshape(batch * n_agents, -1)
        else:
            critic_input = features

        value = self.critic(critic_input).reshape(batch, n_agents)
        return mean.reshape(batch, n_agents), self.log_std, value, next_hidden

    def distribution(self, mean: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(mean, self.log_std.exp())
