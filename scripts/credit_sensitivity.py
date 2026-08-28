"""How much control-influence identification error does the credit signal tolerate?

The closed-form credit's *sign* -- whether a surface's action is scored as
damping or exciting a mode -- depends on ``v_r^T b_k`` (``modal_residues`` in
``aegis.physics.aeroelastic``), which is a simulation quantity: on real
hardware the control-influence column ``b_k`` a designer would use to compute
it is itself an estimate, subject to identification error. This script answers
two questions:

1. (Analytic, no training.) Aeroelastic "control reversal" -- elastic twist
   redistributing lift until a surface's net effect on a mode changes sign --
   is a real, well documented phenomenon, not a numerical curiosity, so identi-
   fication error is modelled as independent per (mode, surface) noise on the
   control-influence entries, scaled relative to the natural magnitude of that
   mode's row (since entries within a row share the same physical origin --
   the same modal projection -- but each surface's own aerodynamic derivative
   is independently estimated). For each entry, the relative noise magnitude at
   which its sign would flip is ``|entry| / row_scale``: small entries relative
   to their row are fragile, large ones are robust, which is the physically
   correct qualitative picture and is not visible under a naive uniform-scale
   perturbation (which flips every entry simultaneously at -100% regardless of
   magnitude, and is reported here too, for the plant's own already-measured
   1.9x UVLM control-authority discrepancy as a scale reference).

2. (Empirical, light training sweep.) With the *true* dynamics held fixed and
   only the credit signal's belief about control influence perturbed
   (``EnvConfig.credit_identification_error``, see ``aegis/envs/batched.py``'s
   ``_credit_force_matrix``), does training the best credit variant degrade
   gracefully or catastrophically as that belief error grows?

Usage::

    python scripts/credit_sensitivity.py --outdir runs/credit_sensitivity
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from aegis.envs.batched import BatchedFlutterEnv
from aegis.envs.flutter_marl import EnvConfig
from aegis.learning.evaluate import LearnedController, evaluate, standard_conditions
from aegis.learning.networks import PolicySpec
from aegis.learning.ppo import PPOConfig, PPOTrainer
from aegis.physics.aeroelastic import AeroelasticModel
from aegis.physics.wing import GOLAND_WING

# Matches scripts/train_agents.py's TRAIN_TASK/CENTRAL_STATE_DIM exactly, so
# this probe trains on the same task distribution as the main ablations.
_TRAIN_TASK = dict(
    speed_ratio_range=(1.00, 1.50),
    episode_duration=2.0,
    jam_probability=0.5,
    jam_angle_range=(-0.245, 0.245),
    tip_plunge_range=(0.03, 0.06),
)
_CENTRAL_STATE_DIM = 12


def fragility_report(model: AeroelasticModel, speed_ratios: list[float]) -> dict:
    """Per (speed, mode, surface): the entry, and the relative per-entry noise
    magnitude that would flip its sign, scaled to that mode's row."""
    flutter = model.flutter_point().airspeed
    report: dict[str, list[dict]] = {}
    for ratio in speed_ratios:
        speed = ratio * flutter
        residues = model.modal_residues(speed)
        n_modes, n_surfaces = residues.shape
        rows = []
        for mode in range(n_modes):
            row = residues[mode]
            row_scale = np.abs(row).max()
            for surface in range(n_surfaces):
                entry = row[surface]
                fragility = abs(entry) / row_scale if row_scale > 0 else float("nan")
                rows.append(
                    {
                        "mode": mode,
                        "surface": surface,
                        "residue": float(entry),
                        # Smallest independent per-entry relative noise (as a
                        # fraction of the row's own scale) that flips this
                        # entry's sign. Smaller = more fragile.
                        "relative_noise_at_sign_flip": float(fragility),
                    }
                )
        report[f"U={ratio:.2f}Uf"] = rows
    return report


def _train_one(
    error: float, seed: int, steps: int, n_envs: int, eval_episodes: int
) -> dict:
    """Train the exact-plus-residual credit variant with a misinformed credit
    signal (true dynamics unaffected) and evaluate it, mirroring
    scripts/train_agents.py's _run_one for the "ippo_blended" variant."""
    train_config = EnvConfig(
        reward_mode="blended", credit_identification_error=error, **_TRAIN_TASK
    )
    env = BatchedFlutterEnv(train_config, n_envs=n_envs, seed=seed)
    spec = PolicySpec(obs_dim=env.obs_dim, n_agents=env.n_agents)
    trainer = PPOTrainer(
        env, spec, PPOConfig(total_steps=steps, n_envs=n_envs, seed=seed)
    )
    started = time.perf_counter()
    trainer.train(progress=False)
    elapsed = time.perf_counter() - started

    eval_config = EnvConfig(
        reward_mode="blended",
        credit_identification_error=error,
        speed_ratio_range=(1.0, 1.5),
        episode_duration=2.0,
    )
    controller = LearnedController(trainer.policy, trainer.device)
    results = evaluate(
        controller, eval_config, standard_conditions(), n_episodes=eval_episodes
    )
    healthy = [r for r in results if r.condition.jam_surface < 0 and r.divergence_rate == 0.0]
    decay = float(np.mean([r.suppression_rate for r in healthy])) if healthy else float("nan")
    return {
        "credit_identification_error": error,
        "seed": seed,
        "train_seconds": elapsed,
        "worst_divergence": max(r.divergence_rate for r in results),
        "healthy_decay_rate": decay,
    }


def identification_error_sweep(
    levels: list[float], seeds: int, steps: int, n_envs: int, eval_episodes: int
) -> list[dict]:
    rows = []
    for error in levels:
        for seed in range(seeds):
            print(f"  training error={error:+.2f} seed={seed} ...", flush=True)
            row = _train_one(error, seed, steps, n_envs, eval_episodes)
            print(
                f"    decay {row['healthy_decay_rate']:+.3f} 1/s, "
                f"worst divergence {row['worst_divergence']*100:.0f}%"
            )
            rows.append(row)
    return rows


def main() -> None:
    args = _parse_args()
    model = AeroelasticModel(GOLAND_WING)
    speed_ratios = [1.05, 1.25, 1.45]

    report = fragility_report(model, speed_ratios)
    most_fragile = min(
        (row for rows in report.values() for row in rows),
        key=lambda r: r["relative_noise_at_sign_flip"],
    )
    print(
        "Per-entry independent identification noise, scaled to each mode's own "
        "row: the most fragile (surface, mode) pair flips sign at "
        f"{most_fragile['relative_noise_at_sign_flip']*100:.1f}% relative "
        f"noise (surface {most_fragile['surface']}, mode {most_fragile['mode']}, "
        f"residue {most_fragile['residue']:.1f}).\n"
        "Under a uniform per-surface scale error instead (every entry moves by "
        "the same fraction), every sign flips simultaneously at exactly -100% "
        "relative error, regardless of magnitude -- the plant's own reported "
        "UVLM discrepancy (Section 3.5) is +90% (an overestimate, the safe "
        "direction, moving assumed values further from zero), a factor of ~2 "
        "away from that boundary.\n"
    )
    print(json.dumps(report, indent=2))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.train_sweep:
        levels = [float(x) for x in args.levels.split(",")]
        print(f"\ntraining sweep: levels={levels}, {args.seeds} seeds each\n")
        rows = identification_error_sweep(
            levels, args.seeds, args.steps, args.n_envs, args.eval_episodes
        )
        (outdir / "identification_error_sweep.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {outdir / 'identification_error_sweep.json'}")

    (outdir / "sign_flip_analysis.json").write_text(
        json.dumps(
            {
                "speed_ratios": speed_ratios,
                "per_entry_fragility": report,
                "most_fragile": most_fragile,
                "uniform_scale_sign_flip_relative_error": -1.0,
                "uvlm_reference_relative_error": 0.9,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {outdir / 'sign_flip_analysis.json'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="runs/credit_sensitivity")
    parser.add_argument("--train-sweep", action="store_true")
    parser.add_argument("--levels", default="0.0,0.3,0.6,0.9,-0.3,-0.6")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=48)
    return parser.parse_args()


if __name__ == "__main__":
    main()
