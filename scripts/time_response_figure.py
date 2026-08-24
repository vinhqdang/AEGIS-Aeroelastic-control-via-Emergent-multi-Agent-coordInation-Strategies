"""Time-domain response of the classical ladder to the same disturbance.

Every result in the paper is an aggregate: a decay rate, a mean over
conditions, a phase gradient. None of it shows a reader what the *signal*
looks like -- tip plunge settling or diverging over time -- which is the first
plot a controls reviewer reaches for. This produces that plot: one flight
condition, one initial condition, four controllers, sharing axes so the
comparison is direct.

Usage::

    python scripts/time_response_figure.py --out paper/figures/fig_timedomain.pdf
"""

from __future__ import annotations

import argparse

import numpy as np

from aegis.control.lqr import CentralizedLQR, LocalRateFeedback
from aegis.physics.aeroelastic import AeroelasticModel
from aegis.physics.wing import GOLAND_WING
from aegis.simulator import WingSimulation

INITIAL_TIP_PLUNGE = 0.05
DURATION = 2.4
DIVERGENCE_LIMIT = 0.9


def main() -> None:
    args = _parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aegis.viz.paper import INK_MUTED, controller_color, use_paper_style

    use_paper_style()

    model = AeroelasticModel(GOLAND_WING)
    flutter = model.flutter_point()
    airspeed = flutter.airspeed * 1.10

    controllers = [
        ("open loop", None),
        ("local rate feedback", LocalRateFeedback(model)),
        ("centralised LQR", CentralizedLQR(model, airspeed)),
    ]
    trajectories = [
        (name, _rollout(model, airspeed, controller))
        for name, controller in controllers
    ]

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.4, 5.0), sharex=True,
        gridspec_kw=dict(height_ratios=[1.35, 1.0], hspace=0.12, left=0.13,
                          right=0.97, top=0.94, bottom=0.11),
    )

    for index, (name, trajectory) in enumerate(trajectories):
        color = controller_color(index)
        top.plot(
            trajectory.time, trajectory.tip_plunge * 100.0,
            color=color, lw=1.4, label=name,
        )
        bottom.plot(
            trajectory.time, np.maximum(trajectory.energy, 1e-9),
            color=color, lw=1.4,
        )

    top.axhline(0.0, color=INK_MUTED, lw=0.6)
    top.set_ylabel("tip plunge   [cm]")
    top.legend(loc="upper left", fontsize=8)
    top.set_title(
        "Same 5 cm pluck at $U/U_\\mathrm{f}=1.10$",
        fontsize=9.5,
    )

    bottom.set_yscale("log")
    bottom.set_ylabel("structural energy   [J]")
    bottom.set_xlabel("time   [s]")
    bottom.set_xlim(0.0, DURATION)

    figure.savefig(args.out)
    print(f"wrote {args.out}")
    for name, trajectory in trajectories:
        growth = trajectory.energy_growth_rate()
        print(f"  {name:22s} growth={growth:+.3f} 1/s  "
              f"peak={float(np.abs(trajectory.tip_plunge).max())*100:.1f} cm  "
              f"diverged={trajectory.diverged}")


def _rollout(model, airspeed, controller):
    sim = WingSimulation(model, airspeed, divergence_limit=DIVERGENCE_LIMIT)
    return sim.rollout(
        controller,
        duration=DURATION,
        initial_tip_plunge=INITIAL_TIP_PLUNGE,
        stop_on_divergence=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="paper/figures/fig_timedomain.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    main()
