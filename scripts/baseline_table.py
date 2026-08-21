"""Score the classical baseline ladder on the standard evaluation grid.

These are the numbers AEGIS has to beat. Run before any training so the target
is fixed in advance rather than chosen after seeing the learned results.
"""

from __future__ import annotations

import numpy as np

from aegis.control.batched_baselines import (
    BatchedLocalFeedback,
    BatchedLQG,
    BatchedLQR,
    BatchedScheduledLQR,
)
from aegis.envs.flutter_marl import EnvConfig
from aegis.learning.evaluate import evaluate, standard_conditions, summary_table
from aegis.physics.aeroelastic import AeroelasticModel
from aegis.physics.wing import GOLAND_WING


def main() -> None:
    config = EnvConfig(episode_duration=2.0, speed_ratio_range=(1.0, 1.5))
    model = AeroelasticModel(GOLAND_WING)
    flutter = model.flutter_point().airspeed
    speed_range = (flutter * 1.0, flutter * 1.5)
    print(f"flutter speed {flutter:.1f} m/s; evaluating over {speed_range[0]:.0f}-{speed_range[1]:.0f} m/s\n")

    controllers = {
        "local_fb": BatchedLocalFeedback(),
        "lqr_fixed": BatchedLQR(model, flutter * 1.1, config.control_dt),
        "lqr_sched": BatchedScheduledLQR(model, speed_range, config.control_dt),
        "lqg_local": BatchedLQG(model, speed_range, config.control_dt),
    }
    conditions = standard_conditions()
    results = {}
    for name, controller in controllers.items():
        results[name] = evaluate(controller, config, conditions, n_episodes=32)
        worst = max(r.divergence_rate for r in results[name])
        print(f"  {name:10s} evaluated; worst-case divergence {worst * 100:.0f}%")
    print()
    print(summary_table(results))


if __name__ == "__main__":
    main()
