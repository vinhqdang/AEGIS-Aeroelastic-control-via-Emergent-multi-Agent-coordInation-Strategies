"""Measuring the spanwise phase pattern a controller adopts.

"Emergent coordination" is a loaded phrase, so it is defined here as something
measurable: the phase at which each surface acts relative to the critical modal
motion, as a function of spanwise station. A controller that has learned to damp
a travelling structural wave should show a systematic phase progression along the
span; one that simply reacts locally should not.

The measurement is deliberately controller-agnostic. It reads only the recorded
deflection and modal-rate signals, so a learned policy, an LQG and a hand-tuned
damper are all characterised the same way, and the learned pattern can be
compared against what classical synthesis produces rather than admired on its own.

Method: take the cross-spectrum of each surface deflection against the first
modal velocity, evaluated at the frequency where the modal response peaks. The
complex ratio gives a gain and a phase per surface. Working at the dominant
frequency rather than over the whole band matters because flutter suppression is
a narrowband phase problem -- a broadband average would blur exactly the quantity
of interest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aegis.envs.batched import BatchedFlutterEnv
from aegis.envs.flutter_marl import EnvConfig

# Band searched for the dominant structural response. The lower bound excludes
# quasi-static trim drift; the upper bound sits well below Nyquist and above the
# highest retained structural mode (about 57 Hz for the Goland wing). Without it
# the peak search can settle on a bin carrying no real signal -- it reported
# exactly 100 Hz, the Nyquist frequency, for a policy whose response actually
# peaked at 3 Hz.
_SEARCH_BAND_HZ = (1.0, 60.0)


@dataclass(frozen=True)
class PhaseProfile:
    """Per-surface gain and phase relative to the critical modal velocity."""

    span_fraction: np.ndarray   # (n_agents,) mid-band spanwise station
    gain: np.ndarray            # (n_agents,) rad of deflection per unit modal rate
    phase_deg: np.ndarray       # (n_agents,) degrees, wrapped to (-180, 180]
    dominant_hz: float
    authority_fraction: float   # RMS deflection as a fraction of available travel
    label: str

    def phase_gradient(self) -> float:
        """Least-squares slope of unwrapped phase against span, deg per semispan.

        A non-zero slope is the operational meaning of a spanwise phase pattern.
        """
        unwrapped = np.rad2deg(np.unwrap(np.deg2rad(self.phase_deg)))
        if self.span_fraction.size < 2:
            return float("nan")
        return float(np.polyfit(self.span_fraction, unwrapped, 1)[0])


def measure_phase_profile(
    controller,
    config: EnvConfig,
    label: str,
    speed_ratio: float = 1.15,
    n_episodes: int = 24,
    initial_tip_plunge: float = 0.05,
    seed: int = 4242,
) -> PhaseProfile:
    """Run rollouts and extract the per-surface phase relative to mode 1."""
    env = BatchedFlutterEnv(config, n_envs=n_episodes, seed=seed, auto_reset=False)
    speeds = np.full(n_episodes, env.flutter_speed * speed_ratio)
    signs = np.where(np.arange(n_episodes) % 2 == 0, 1.0, -1.0)
    observation = env.reset_to(
        speeds, signs * initial_tip_plunge, gust_seed=seed
    )
    controller.reset(n_episodes)

    n_steps = round(config.episode_duration / config.control_dt)
    deflections = np.zeros((n_steps, n_episodes, env.n_agents))
    modal_rate = np.zeros((n_steps, n_episodes))
    for step in range(n_steps):
        action = controller(env, observation)
        observation, _, _, _ = env.step(action)
        deflections[step] = env.deflection
        modal_rate[step] = env.plant_state[:, env.n_modes]

    gain, phase, dominant = _cross_spectrum_phase(
        deflections, modal_rate, config.control_dt
    )
    span = np.asarray(
        [0.5 * (s.y_start_frac + s.y_end_frac) for s in config.wing.surfaces]
    )
    travel = np.asarray([s.max_deflection for s in config.wing.surfaces])
    authority = float(np.sqrt(np.mean((deflections / travel) ** 2)))
    return PhaseProfile(span, gain, phase, dominant, authority, label)


def _cross_spectrum_phase(
    deflections: np.ndarray, modal_rate: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Gain and phase of each deflection channel against the modal rate."""
    n_steps = deflections.shape[0]
    window = np.hanning(n_steps)[:, None]

    rate_spectrum = np.fft.rfft(modal_rate * window, axis=0)
    power = (np.abs(rate_spectrum) ** 2).sum(axis=1)
    frequencies = np.fft.rfftfreq(n_steps, dt)

    # Search only where a structural response can physically live.
    in_band = (frequencies >= _SEARCH_BAND_HZ[0]) & (frequencies <= _SEARCH_BAND_HZ[1])
    masked = np.where(in_band, power, 0.0)
    peak = int(np.argmax(masked))

    reference = rate_spectrum[peak]  # (n_episodes,)
    gains = np.zeros(deflections.shape[2])
    phases = np.zeros(deflections.shape[2])
    for agent in range(deflections.shape[2]):
        channel = np.fft.rfft(deflections[:, :, agent] * window, axis=0)[peak]
        # Average the complex ratio over episodes, not the phases: averaging
        # angles across a wrap boundary produces meaningless results.
        ratio = np.mean(channel * np.conj(reference)) / (
            np.mean(np.abs(reference) ** 2) + 1e-30
        )
        gains[agent] = float(np.abs(ratio))
        phases[agent] = float(np.rad2deg(np.angle(ratio)))
    return gains, phases, float(frequencies[peak])


def plot_phase_profiles(profiles: list[PhaseProfile], path) -> None:
    """Phase and gain against span, one line per controller."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aegis.viz.paper import INK_MUTED, controller_color, use_paper_style

    use_paper_style()
    # Tall enough to reserve a legend row below both panels: with five long,
    # stat-bearing labels and only three points per line, "best" placement has
    # nowhere clean inside the axes and lands the legend on top of the curves.
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(5.6, 5.7), sharex=True)
    for index, profile in enumerate(profiles):
        color = controller_color(index)
        top.plot(
            profile.span_fraction, profile.phase_deg, "o-", color=color,
            label=(
                f"{profile.label}  ({profile.phase_gradient():+.0f}$^\\circ$/span, "
                f"{100 * profile.authority_fraction:.0f}% authority)"
            ),
        )
        bottom.plot(profile.span_fraction, profile.gain, "o-", color=color,
                    label=profile.label)

    top.axhline(0.0, color=INK_MUTED, lw=0.6, ls=":")
    top.set_ylabel("phase vs mode 1  [deg]")
    top.set_title(
        "Spanwise actuation phase at the dominant response frequency", loc="left"
    )
    bottom.set_ylabel("gain  [rad per unit modal rate]")
    bottom.set_xlabel("spanwise station  [fraction of semispan]")
    bottom.set_title("Actuation gain", loc="left")

    handles, labels = top.get_legend_handles_labels()
    figure.subplots_adjust(bottom=0.30, hspace=0.35)
    figure.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.52, 0.0),
        fontsize=6.6, ncol=1, frameon=False,
    )
    figure.savefig(path)
    plt.close(figure)
