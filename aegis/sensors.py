"""Distributed sensing: what each agent is actually allowed to see.

Real distributed control on a wing runs off accelerometers bonded near each
actuator, not off a modal state estimate. AEGIS models that literally: a pair of
accelerometers per station, one forward and one aft of the elastic axis, from
which bending and torsional acceleration are recovered by sum and difference.

Because each agent sees only its own station pair, the observation is genuinely
partial -- the coupled bending-torsion flutter mode is a global quantity that no
single agent can measure. That is the constraint the coordination has to work
around, so it is enforced here rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aegis.physics.aeroelastic import AeroelasticModel


@dataclass(frozen=True)
class SensorStation:
    """One accelerometer pair at a spanwise station."""

    name: str
    y_frac: float
    forward_offset: float  # chordwise distance ahead of the EA, m (positive)
    aft_offset: float      # chordwise distance aft of the EA, m (positive)
    plunge_row: np.ndarray
    twist_row: np.ndarray

    def accelerations(self, modal_acceleration: np.ndarray) -> tuple[float, float]:
        """Forward and aft accelerometer readings, m/s^2, positive **down**.

        A point at chordwise distance ``x`` aft of the elastic axis moves down by
        ``h + x * alpha``, so its downward acceleration is
        ``hddot + x * alphaddot``.
        """
        plunge_accel = float(self.plunge_row @ modal_acceleration)
        twist_accel = float(self.twist_row @ modal_acceleration)
        return (
            plunge_accel - self.forward_offset * twist_accel,
            plunge_accel + self.aft_offset * twist_accel,
        )


class SensorArray:
    """One :class:`SensorStation` co-located with each control surface."""

    def __init__(
        self,
        model: AeroelasticModel,
        forward_offset_frac: float = 0.25,
        aft_offset_frac: float = 0.30,
    ):
        wing = model.wing
        chord = wing.chord
        self.stations: tuple[SensorStation, ...] = tuple(
            SensorStation(
                name=surface.name,
                y_frac=_band_midpoint(surface.y_start_frac, surface.y_end_frac),
                forward_offset=forward_offset_frac * chord,
                aft_offset=aft_offset_frac * chord,
                plunge_row=model.station_rows(
                    _band_midpoint(surface.y_start_frac, surface.y_end_frac)
                )[0],
                twist_row=model.station_rows(
                    _band_midpoint(surface.y_start_frac, surface.y_end_frac)
                )[1],
            )
            for surface in wing.surfaces
        )

    def __len__(self) -> int:
        return len(self.stations)

    def read(self, modal_acceleration: np.ndarray) -> np.ndarray:
        """All accelerometer readings, shape ``(n_stations, 2)``."""
        return np.asarray(
            [station.accelerations(modal_acceleration) for station in self.stations]
        )

    def local_states(
        self, modal_position: np.ndarray, modal_rate: np.ndarray
    ) -> np.ndarray:
        """Local plunge rate and twist angle/rate per station, shape ``(n, 3)``.

        These stand in for the strain-gauge and rate-gyro channels that sit
        alongside the accelerometers on a real distributed installation.
        """
        readings = np.empty((len(self.stations), 3))
        for index, station in enumerate(self.stations):
            readings[index] = (
                station.plunge_row @ modal_rate,
                station.twist_row @ modal_position,
                station.twist_row @ modal_rate,
            )
        return readings


def _band_midpoint(start_frac: float, end_frac: float) -> float:
    return 0.5 * (start_frac + end_frac)
