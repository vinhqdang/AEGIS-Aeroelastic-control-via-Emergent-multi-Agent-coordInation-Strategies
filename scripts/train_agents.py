"""Train one AEGIS variant and score it on the standard evaluation grid.

Variants are named presets so the ablation set is defined in one place rather
than in shell arguments. Every variant trains on the *same* task distribution --
flight condition and actuator failures randomised over the envelope where the
classical baselines fail -- so the only differences are the ones being ablated:
the reward decomposition, the communication channel, and the action
parameterisation.

Usage::

    python scripts/train_agents.py --preset mocca --steps 1000000
    python scripts/train_agents.py --preset all --steps 1000000 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from aegis.envs.batched import BatchedFlutterEnv
from aegis.envs.flutter_marl import EnvConfig
from aegis.learning.evaluate import (
    LearnedController,
    evaluate,
    standard_conditions,
)
from aegis.learning.networks import PolicySpec
from aegis.learning.ppo import PPOConfig, PPOTrainer

# Training task: the envelope and failure set where every classical baseline
# either diverges or barely holds. Chosen from scripts/baseline_table.py before
# any learning, so the target is not moved after seeing results.
TRAIN_TASK = dict(
    speed_ratio_range=(1.00, 1.50),
    episode_duration=2.0,
    jam_probability=0.5,
    jam_angle_range=(-0.245, 0.245),
    tip_plunge_range=(0.03, 0.06),
)

CENTRAL_STATE_DIM = 12  # 2*n_modes modal states + 3 deflections + speed


@dataclass(frozen=True)
class Variant:
    """One row of the ablation table."""

    name: str
    reward_mode: str
    comm_mode: str
    use_phasor_consensus: bool
    use_phase_locked_head: bool
    central_critic: bool
    note: str
    consensus_rounds: int = 2
    base_controller: str = "none"
    residual_authority: float = 1.0
    use_health_channel: bool = False
    aux_coefficient: float = 0.0


VARIANTS: dict[str, Variant] = {
    "ippo_shared": Variant(
        "ippo_shared", "shared", "none", False, False, False,
        "naive baseline: one global reward, no communication",
    ),
    "mappo_shared": Variant(
        "mappo_shared", "shared", "none", False, False, True,
        "learned credit: centralised critic on privileged state",
    ),
    "shaped_shared": Variant(
        "shaped_shared", "shaped", "none", False, False, False,
        "aligned shared shaping, NO per-agent credit: isolates the credit term",
    ),
    "ippo_physics": Variant(
        "ippo_physics", "physics", "none", False, False, False,
        "Proposition 1 credit alone, no communication",
    ),
    "ippo_blended": Variant(
        "ippo_blended", "blended", "none", False, False, False,
        "exact-plus-residual credit, no communication",
    ),
    "comm_raw": Variant(
        "comm_raw", "blended", "neighbor_local", False, False, False,
        "raw neighbour observations: 14 floats of bandwidth",
    ),
    "comm_oracle": Variant(
        "comm_oracle", "blended", "modal_oracle", False, False, False,
        "oracle modal state: upper bound on any encoder",
    ),
    "recurrent_only": Variant(
        "recurrent_only", "blended", "none", True, False, False,
        "recurrent encoder, NO message passing: isolates memory from comms",
        consensus_rounds=0,
    ),
    "mocca_nohead": Variant(
        "mocca_nohead", "blended", "none", True, False, False,
        "phasor consensus, raw action head",
    ),
    "aux_encoder": Variant(
        "aux_encoder", "blended", "none", True, False, False,
        "phasor encoder supervised on the true modal phasor, raw action head",
        aux_coefficient=1.0,
    ),
    "aux_residual": Variant(
        "aux_residual", "blended", "none", True, False, False,
        "supervised phasor encoder correcting a decentralised LQG base law",
        base_controller="dlqg", residual_authority=0.3, aux_coefficient=1.0,
    ),
    "mocca": Variant(
        "mocca", "blended", "none", True, True, False,
        "MoCCA: phasor consensus + phase-locked actions + blended credit",
    ),
    "mocca_residual": Variant(
        "mocca_residual", "blended", "none", True, True, False,
        "MoCCA correcting a decentralised LQG base law",
        base_controller="dlqg", residual_authority=0.3,
    ),
    "residual_health": Variant(
        "residual_health", "blended", "none", True, False, False,
        "residual on decentralised LQG with a health-carrying consensus message",
        base_controller="dlqg", residual_authority=0.3, use_health_channel=True,
    ),
    "residual_comm": Variant(
        "residual_comm", "blended", "neighbor_local", False, False, False,
        "residual on decentralised LQG, neighbours share raw local observations",
        base_controller="dlqg", residual_authority=0.3,
    ),
    "residual_only": Variant(
        "residual_only", "blended", "none", False, False, False,
        "residual on decentralised LQG, no comms and no phasor: isolates residual",
        base_controller="dlqg", residual_authority=0.3,
    ),
}


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    names = list(VARIANTS) if args.preset == "all" else args.preset.split(",")
    for name in names:
        if name not in VARIANTS:
            raise SystemExit(f"unknown preset {name!r}; choose from {list(VARIANTS)} or 'all'")

    for name in names:
        for seed in range(args.seeds):
            _run_one(VARIANTS[name], seed, args, outdir)


def _run_one(variant: Variant, seed: int, args, outdir: Path) -> None:
    tag = f"{variant.name}_s{seed}"
    if args.surfaces:
        tag = f"{variant.name}_n{args.surfaces}_s{seed}"
    print(f"\n=== {tag} :: {variant.note}")

    task = dict(TRAIN_TASK)
    if args.easy:
        # Capability probe: can the architecture reach classical performance on
        # the SINGLE easiest condition at all? Every variant so far plateaus in
        # the same narrow band regardless of credit signal, communication, action
        # parameterisation, memory or supervision, which points at a systemic
        # cause rather than an algorithmic one. If a policy trained on one
        # condition still cannot approach the classical -5.5 1/s there, the
        # limit is not breadth of the task distribution.
        task["speed_ratio_range"] = (1.05, 1.05)
        task["jam_probability"] = 0.0
        task["tip_plunge_range"] = (0.05, 0.05)
    if args.surfaces:
        from aegis.physics.wing import goland_with_surfaces

        task["wing"] = goland_with_surfaces(args.surfaces)
    train_config = EnvConfig(
        reward_mode=variant.reward_mode,
        comm_mode=variant.comm_mode,
        base_controller=variant.base_controller,
        residual_authority=variant.residual_authority,
        **task,
    )
    env = BatchedFlutterEnv(train_config, n_envs=args.n_envs, seed=seed)
    spec = PolicySpec(
        obs_dim=env.obs_dim,
        n_agents=env.n_agents,
        use_phasor_consensus=variant.use_phasor_consensus,
        use_phase_locked_head=variant.use_phase_locked_head,
        consensus_rounds=variant.consensus_rounds,
        use_health_channel=variant.use_health_channel,
        central_state_dim=CENTRAL_STATE_DIM if variant.central_critic else None,
    )
    trainer = PPOTrainer(
        env,
        spec,
        PPOConfig(
            total_steps=args.steps,
            n_envs=args.n_envs,
            rollout_length=args.rollout,
            aux_coefficient=variant.aux_coefficient,
            initial_log_std=args.log_std,
            entropy_coefficient=args.entropy,
            seed=seed,
        ),
    )

    started = time.perf_counter()
    log = trainer.train()
    elapsed = time.perf_counter() - started
    print(f"    trained in {elapsed:.0f}s")

    # Evaluation uses a fixed grid with no random failures: conditions are
    # specified explicitly so every variant and baseline faces the same episodes.
    eval_config = EnvConfig(
        reward_mode=variant.reward_mode,
        comm_mode=variant.comm_mode,
        base_controller=variant.base_controller,
        residual_authority=variant.residual_authority,
        speed_ratio_range=(1.0, 1.5),
        episode_duration=2.0,
        **({"wing": task["wing"]} if args.surfaces else {}),
    )
    controller = LearnedController(trainer.policy, trainer.device)
    results = evaluate(
        controller,
        eval_config,
        standard_conditions(),
        n_episodes=args.eval_episodes,
    )

    worst = max(r.divergence_rate for r in results)
    healthy = [r for r in results if r.condition.jam_surface < 0]
    print(
        f"    worst divergence {worst * 100:5.1f}%   "
        f"healthy suppression {np.mean([r.suppression_rate for r in healthy]):+.3f} 1/s"
    )

    payload = {
        "variant": asdict(variant),
        "seed": seed,
        "steps": args.steps,
        "train_seconds": elapsed,
        "log": log.as_dict(),
        "results": [
            {
                "condition": {
                    "speed_ratio": r.condition.speed_ratio,
                    "jam_surface": r.condition.jam_surface,
                    "jam_angle": r.condition.jam_angle,
                    "label": r.condition.describe(),
                },
                "divergence_rate": r.divergence_rate,
                "suppression_rate": r.suppression_rate,
                "suppression_rate_std": r.suppression_rate_std,
                "peak_tip_plunge": r.peak_tip_plunge,
                "rms_deflection": r.rms_deflection,
            }
            for r in results
        ],
    }
    (outdir / f"{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save(trainer.policy.state_dict(), outdir / f"{tag}.pt")
    print(f"    wrote {outdir / f'{tag}.json'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="mocca", help="variant name, comma list, or 'all'")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument(
        "--surfaces", type=int, default=0,
        help="number of control surfaces (0 keeps the default 3-surface layout)",
    )
    parser.add_argument(
        "--log-std", type=float, default=-1.2,
        help="initial exploration log sigma in action units",
    )
    parser.add_argument(
        "--entropy", type=float, default=2.0e-3,
        help="entropy bonus; lower lets PPO shrink sigma for fine control",
    )
    parser.add_argument(
        "--easy", action="store_true",
        help="train on the single easiest condition (capability probe)",
    )
    parser.add_argument("--outdir", default="runs")
    return parser.parse_args()


if __name__ == "__main__":
    main()
