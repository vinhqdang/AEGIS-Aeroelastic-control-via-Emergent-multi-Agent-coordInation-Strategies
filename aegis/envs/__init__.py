"""Multi-agent environments exposing the aeroelastic plants to RL algorithms."""

from aegis.envs.flutter_marl import (
    COMM_MODES,
    REWARD_MODES,
    EnvConfig,
    FlutterSuppressionEnv,
)

__all__ = [
    "COMM_MODES",
    "REWARD_MODES",
    "EnvConfig",
    "FlutterSuppressionEnv",
]
