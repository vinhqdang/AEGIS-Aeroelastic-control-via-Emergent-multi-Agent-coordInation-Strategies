"""Atmospheric disturbance models used to excite the wing.

Two families, both returning a vertical gust velocity in m/s, positive up:

* :class:`DiscreteGust` -- the 1-cosine gust from the certification
  specifications. Deterministic, so it is the right choice for comparing
  controllers on identical inputs.
* :class:`DrydenTurbulence` -- continuous turbulence from the Dryden vertical
  velocity spectrum, realised as a second-order filter driven by white noise.
  Stochastic, so it is what agents should train against.

Both are pure functions of time given a seed, which keeps episodes reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import bilinear, lfilter

_BURN_IN_TIME_CONSTANTS = 12.0


class GustModel:
    """Interface: vertical gust velocity in m/s at a given time."""

    def velocity(self, time: float) -> float:
        raise NotImplementedError

    def reset(self, rng: np.random.Generator) -> None:
        """Redraw any random parameters at the start of an episode."""


@dataclass
class NoGust:
    """Still air -- used for initial-condition-only flutter studies."""

    def velocity(self, time: float) -> float:
        return 0.0

    def reset(self, rng: np.random.Generator) -> None:
        return None


class DiscreteGust(GustModel):
    """1-cosine gust: a single smooth pulse of chosen amplitude and length."""

    def __init__(
        self,
        amplitude: float = 5.0,
        gust_length: float = 25.0,
        airspeed: float = 150.0,
        start_time: float = 0.05,
        randomize: bool = True,
    ):
        if amplitude <= 0 or gust_length <= 0 or airspeed <= 0:
            raise ValueError("amplitude, gust_length and airspeed must be positive")
        self.nominal_amplitude = amplitude
        self.gust_length = gust_length
        self.airspeed = airspeed
        self.nominal_start = start_time
        self.randomize = randomize
        self.amplitude = amplitude
        self.start_time = start_time

    @property
    def duration(self) -> float:
        """Time for the gust to sweep past a fixed point, s."""
        return self.gust_length / self.airspeed

    def reset(self, rng: np.random.Generator) -> None:
        if not self.randomize:
            self.amplitude, self.start_time = self.nominal_amplitude, self.nominal_start
            return
        # Sign and magnitude both vary so a policy cannot memorise one response.
        self.amplitude = (
            self.nominal_amplitude
            * rng.uniform(0.5, 1.5)
            * rng.choice([-1.0, 1.0])
        )
        self.start_time = self.nominal_start * rng.uniform(0.5, 2.0)

    def velocity(self, time: float) -> float:
        elapsed = time - self.start_time
        if not 0.0 <= elapsed <= self.duration:
            return 0.0
        return 0.5 * self.amplitude * (1.0 - np.cos(2.0 * np.pi * elapsed / self.duration))


class DrydenTurbulence(GustModel):
    """Dryden vertical-gust spectrum realised as a second-order shaping filter.

    The transfer function is

        H(s) = sigma * sqrt(L / (pi U)) * (1 + sqrt(3) L s / U) / (1 + L s / U)^2

    driven by unit white noise. It is integrated on a fixed grid at construction
    time so :meth:`velocity` is a cheap lookup during a rollout -- the plant
    integrator takes sub-steps, and re-drawing noise inside an RK4 stage would
    break the integrator's error estimate.
    """

    def __init__(
        self,
        intensity: float = 2.0,
        scale_length: float = 533.4,
        airspeed: float = 150.0,
        duration: float = 6.0,
        sample_rate: float = 2000.0,
        seed: int | None = None,
    ):
        if intensity <= 0 or scale_length <= 0 or airspeed <= 0:
            raise ValueError("intensity, scale_length and airspeed must be positive")
        self.intensity = intensity
        self.scale_length = scale_length
        self.airspeed = airspeed
        self.duration = duration
        self.sample_rate = sample_rate
        self._samples = np.zeros(int(duration * sample_rate) + 2)
        self.reset(np.random.default_rng(seed))

    def reset(self, rng: np.random.Generator) -> None:
        """Draw a fresh turbulence realisation.

        The analog shaping filter is discretised by the bilinear transform and
        applied with ``lfilter`` rather than integrated by hand: the earlier
        hand-rolled companion form silently dropped a ``1/tau^2`` on the input
        and mis-scaled the noise variance, which inflated the realised RMS by
        almost a factor of six. Using a tested filter implementation removes
        that whole class of error, and :func:`realised_rms` checks the result.
        """
        tau = self.scale_length / self.airspeed
        gain = self.intensity * np.sqrt(self.scale_length / (np.pi * self.airspeed))
        numerator = gain * np.asarray([np.sqrt(3.0) * tau, 1.0])
        denominator = np.asarray([tau**2, 2.0 * tau, 1.0])
        b, a = bilinear(numerator, denominator, fs=self.sample_rate)

        # Unit one-sided PSD in rad/s corresponds to a discrete variance of
        # pi * sample_rate. The filter correlation time L/U is several seconds at
        # cruise -- longer than an episode -- so it is warmed up before recording,
        # otherwise every episode would inherit the same ramp-up transient
        # instead of stationary turbulence.
        burn_in = int(_BURN_IN_TIME_CONSTANTS * tau * self.sample_rate)
        n = self._samples.size
        noise = rng.normal(0.0, np.sqrt(np.pi * self.sample_rate), size=n + burn_in)
        self._samples = lfilter(b, a, noise)[burn_in:]

    def realised_rms(self) -> float:
        """RMS of the current realisation -- should approach ``intensity``."""
        return float(np.std(self._samples))

    def velocity(self, time: float) -> float:
        index = int(np.clip(time * self.sample_rate, 0, self._samples.size - 1))
        return float(self._samples[index])
