"""Compare the spanwise actuation phase of classical and learned controllers.

This is the figure behind the word "emergent". It measures, for each controller,
the phase at which every surface acts relative to the critical modal motion, and
plots it against spanwise station. The measurement is identical for a learned
policy and for an LQG, so the learned pattern can be held against what optimal
centralised synthesis produces instead of being admired in isolation.

Usage::

    python scripts/phase_figure.py --runs runs/final --variants mocca_residual,mocca
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aegis.analysis.phase import measure_phase_profile, plot_phase_profiles
from aegis.control.batched_baselines import (
    BatchedDecentralizedLQG,
    BatchedLocalFeedback,
    BatchedLQG,
)
from aegis.envs.flutter_marl import EnvConfig
from aegis.learning.evaluate import LearnedController
from aegis.learning.networks import MultiAgentPolicy, PolicySpec
from aegis.physics.aeroelastic import AeroelasticModel
from aegis.physics.wing import GOLAND_WING

CENTRAL_STATE_DIM = 12


def main() -> None:
    args = _parse_args()
    model = AeroelasticModel(GOLAND_WING)
    flutter = model.flutter_point().airspeed
    speed_range = (flutter, flutter * 1.5)

    base_config = EnvConfig(episode_duration=2.0, speed_ratio_range=(1.0, 1.5))
    profiles = []

    classical = {
        "local feedback": BatchedLocalFeedback(),
        "decentralised LQG": BatchedDecentralizedLQG(
            model, speed_range, base_config.control_dt
        ),
        "centralised LQG": BatchedLQG(model, speed_range, base_config.control_dt),
    }
    for label, controller in classical.items():
        profile = measure_phase_profile(controller, base_config, label)
        profiles.append(profile)
        _report(profile)

    for name in filter(None, args.variants.split(",")):
        loaded = _load_variant(Path(args.runs), name, model)
        if loaded is None:
            print(f"  (no checkpoint for {name})")
            continue
        controller, config = loaded
        profile = measure_phase_profile(controller, config, name)
        profiles.append(profile)
        _report(profile)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_phase_profiles(profiles, output)
    print(f"\nwrote {output}")


def _report(profile) -> None:
    print(
        f"  {profile.label:22s} f* = {profile.dominant_hz:5.2f} Hz   "
        f"gradient = {profile.phase_gradient():+7.1f} deg/span   "
        f"authority = {100 * profile.authority_fraction:5.1f}%"
    )


def _load_variant(runs: Path, name: str, model):
    """Rebuild a trained policy and its environment configuration."""
    metadata = sorted(runs.glob(f"{name}_s*.json"))
    checkpoints = sorted(runs.glob(f"{name}_s*.pt"))
    if not metadata or not checkpoints:
        return None

    payload = json.loads(metadata[0].read_text(encoding="utf-8"))
    variant = payload["variant"]
    config = EnvConfig(
        reward_mode=variant["reward_mode"],
        comm_mode=variant["comm_mode"],
        base_controller=variant.get("base_controller", "none"),
        residual_authority=variant.get("residual_authority", 1.0),
        speed_ratio_range=(1.0, 1.5),
        episode_duration=2.0,
    )

    from aegis.envs.batched import BatchedFlutterEnv

    probe = BatchedFlutterEnv(config, n_envs=1, seed=0)
    spec = PolicySpec(
        obs_dim=probe.obs_dim,
        n_agents=probe.n_agents,
        use_phasor_consensus=variant["use_phasor_consensus"],
        use_phase_locked_head=variant["use_phase_locked_head"],
        consensus_rounds=variant.get("consensus_rounds", 2),
        central_state_dim=CENTRAL_STATE_DIM if variant["central_critic"] else None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = MultiAgentPolicy(spec).to(device)
    policy.load_state_dict(torch.load(checkpoints[0], map_location=device))
    policy.eval()
    del model
    return LearnedController(policy, device), config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs/final")
    parser.add_argument("--variants", default="mocca_residual,mocca,ippo_physics")
    parser.add_argument("--out", default="paper/figures/fig_phase.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    main()
