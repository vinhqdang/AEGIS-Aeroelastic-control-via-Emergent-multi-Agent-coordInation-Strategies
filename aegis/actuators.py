"""Control-surface actuator dynamics.

Every surface is a first-order servo with a rate limit and a deflection limit.
These three effects are what make distributed flutter suppression hard in
practice: the lag introduces phase loss right where the flutter mode lives, and
the limits mean an agent that commands aggressively simply saturates instead of
acting. Agents that ignore them learn policies that do not transfer.

State is carried as a plain array so the whole bank integrates alongside the
plant inside one RK4 step.
"""

from __future__ import annotations

import numpy as np

from aegis.physics.wing import WingProperties


class ActuatorBank:
    """First-order servos with rate and deflection saturation, one per surface."""

    def __init__(self, wing: WingProperties):
        self.n_surfaces = wing.n_surfaces
        self.time_constants = np.asarray([s.actuator_tau for s in wing.surfaces])
        self.max_deflection = np.asarray([s.max_deflection for s in wing.surfaces])
        self.max_rate = np.asarray([s.max_rate for s in wing.surfaces])
        self.names = tuple(s.name for s in wing.surfaces)
        self._jam_mask = np.zeros(self.n_surfaces, dtype=bool)
        self.jam_deflection = np.zeros(self.n_surfaces)

    def jam(self, jammed: dict[str, float]) -> None:
        """Freeze named surfaces at fixed deflections, in rad.

        This is what an actuator failure actually looks like: the surface holds
        wherever it was when the servo died, it does not helpfully return to
        neutral. Modelling it as "commands are ignored" makes a failure look
        almost free, because the surface then contributes nothing instead of
        contributing a steady wrong load the others must trim out.
        """
        unknown = set(jammed) - set(self.names)
        if unknown:
            raise ValueError(f"unknown surface names: {sorted(unknown)}")
        self._jam_mask = np.zeros(self.n_surfaces, dtype=bool)
        self.jam_deflection = np.zeros(self.n_surfaces)
        for index, name in enumerate(self.names):
            if name in jammed:
                held = jammed[name]
                if abs(held) > self.max_deflection[index]:
                    raise ValueError(
                        f"{name}: jam deflection {held} exceeds travel limit "
                        f"{self.max_deflection[index]}"
                    )
                self._jam_mask[index] = True
                self.jam_deflection[index] = held

    @property
    def jam_mask(self) -> np.ndarray:
        """Boolean mask of surfaces frozen by a failure."""
        return self._jam_mask

    def initial_deflection(self) -> np.ndarray:
        """Deflection at reset: zero, except jammed surfaces which start held."""
        return np.where(self._jam_mask, self.jam_deflection, 0.0)

    def derivative(self, deflection: np.ndarray, command: np.ndarray) -> np.ndarray:
        """Deflection rate, rate-limited and clamped at the travel stops.

        Clamping the rate at the stops (rather than clipping position after the
        fact) keeps the derivative consistent with the integrated state, so RK4
        does not chatter against the limit.
        """
        target = np.clip(command, -self.max_deflection, self.max_deflection)
        rate = np.clip(
            (target - deflection) / self.time_constants, -self.max_rate, self.max_rate
        )
        at_upper = (deflection >= self.max_deflection) & (rate > 0.0)
        at_lower = (deflection <= -self.max_deflection) & (rate < 0.0)
        return np.where(at_upper | at_lower | self._jam_mask, 0.0, rate)

    def normalized_command(self, action: np.ndarray) -> np.ndarray:
        """Map actions in [-1, 1] onto physical commanded deflections in rad."""
        return np.clip(action, -1.0, 1.0) * self.max_deflection

    def saturation_fraction(self, deflection: np.ndarray) -> np.ndarray:
        """Deflection as a fraction of travel, in [-1, 1] -- useful as an observation."""
        return deflection / self.max_deflection
