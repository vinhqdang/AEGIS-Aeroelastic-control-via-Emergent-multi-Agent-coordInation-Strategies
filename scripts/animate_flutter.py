"""Render the AEGIS explainer animations.

Produces, at the same supercritical flight condition:

1. ``open_loop``    -- surfaces locked, flutter diverges.
2. ``lqr``          -- centralised full-state LQR, flutter suppressed.
3. ``local``        -- decentralised local rate feedback, no communication.
4. ``lqr_jam``      -- centralised LQR with the outboard tab jammed off-neutral,
   and the controller unaware of it.

Watched in that order they show the physics, the ceiling, the naive distributed
floor, and the weakness the AEGIS thesis targets.

Usage::

    python scripts/animate_flutter.py --outdir media --format mp4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aegis.control.lqr import CentralizedLQR, LocalRateFeedback
from aegis.physics.aeroelastic import AeroelasticModel
from aegis.physics.wing import GOLAND_WING
from aegis.simulator import WingSimulation
from aegis.viz.animation import AnimationSpec, render_animation

INITIAL_TIP_PLUNGE = 0.05   # 5 cm pluck of the first bending mode
DURATION = 2.4              # s
DIVERGENCE_LIMIT = 0.9      # m of tip travel -- past this the linear model is void
JAM_ANGLE = np.deg2rad(8.0)
GROWTH_TOLERANCE = 0.05     # 1/s, below which we call the response neutral


@dataclass(frozen=True)
class Case:
    """One animation to render."""

    name: str
    title: str
    subtitle: str
    controller: object | None
    jam: dict[str, float]


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)

    model = AeroelasticModel(GOLAND_WING)
    flutter = model.flutter_point()
    airspeed = args.airspeed or flutter.airspeed * 1.10
    print(f"flutter point: {flutter}")
    ratio = airspeed / flutter.airspeed
    print(f"animating at U = {airspeed:.1f} m/s  (U/U_f = {ratio:.3f})\n")

    for case in _build_cases(model, airspeed):
        sim = WingSimulation(model, airspeed, divergence_limit=DIVERGENCE_LIMIT)
        if case.jam:
            sim.actuators.jam(case.jam)

        trajectory = sim.rollout(
            case.controller,
            duration=DURATION,
            initial_tip_plunge=INITIAL_TIP_PLUNGE,
            stop_on_divergence=True,
        )
        print(f"  {case.name:12s} {_verdict(trajectory)}")

        path = outdir / f"{case.name}.{args.format}"
        render_animation(
            model,
            trajectory,
            AnimationSpec(
                title=case.title,
                subtitle=case.subtitle,
                output_path=path,
                stride=args.stride,
                fps=args.fps,
                flutter_speed=flutter.airspeed,
            ),
        )
        print(f"  {'':12s} wrote {path}\n")


def _verdict(trajectory) -> str:
    growth = trajectory.energy_growth_rate()
    if trajectory.diverged:
        label = "DIVERGED (left the linear regime)"
    elif growth > GROWTH_TOLERANCE:
        label = "growing"
    elif growth < -GROWTH_TOLERANCE:
        label = "suppressed"
    else:
        label = "neutral"
    return (
        f"peak tip = {trajectory.peak_tip_plunge * 100:7.2f} cm"
        f"   growth = {growth:+7.3f} 1/s"
        f"   duration = {trajectory.time[-1]:.2f} s   -> {label}"
    )


def _build_cases(model: AeroelasticModel, airspeed: float) -> list[Case]:
    # The jammed case reuses the nominal gain on purpose: a real failure is not
    # announced to the controller, and re-synthesising for the failure would
    # measure a different (easier) problem.
    nominal = CentralizedLQR(model, airspeed)
    return [
        Case(
            name="open_loop",
            title="Open loop — flutter",
            subtitle=(
                "All surfaces locked at zero. Bending and torsion coalesce into one "
                "unstable mode and the response grows without bound."
            ),
            controller=None,
            jam={},
        ),
        Case(
            name="lqr",
            title="Centralised LQR — the ceiling",
            subtitle=(
                "Full plant state including wake states, actuator lag inside the "
                "design plant. This is the performance a distributed policy must match."
            ),
            controller=nominal,
            jam={},
        ),
        Case(
            name="local",
            title="Local rate feedback — the distributed floor",
            subtitle=(
                "Each surface sees only its own station and never communicates. "
                "It damps local motion while blind to the coupled flutter mode."
            ),
            controller=LocalRateFeedback(model),
            jam={},
        ),
        Case(
            name="lqr_jam",
            title="Centralised LQR — outboard tab jammed at 8°",
            subtitle=(
                "The gain was synthesised for three healthy surfaces and is never "
                "told about the failure. This is the weakness AEGIS targets."
            ),
            controller=nominal,
            jam={"tab_outboard": JAM_ANGLE},
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="media", help="output directory")
    parser.add_argument("--format", default="mp4", choices=["mp4", "gif"])
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--stride", type=int, default=2, help="trajectory steps per frame")
    parser.add_argument(
        "--airspeed", type=float, default=None,
        help="flight speed in m/s (default: 1.10 x flutter speed)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
