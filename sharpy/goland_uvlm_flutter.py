"""Goland flutter speed from SHARPy's UVLM, for comparison with the AEGIS plant.

Deliberately a minimal deviation from SHARPy's own passing test
(``tests/linear/goland_wing/test_goland_flutter.py``): same solver chain, same
scaling, same ROM settings, same mode handling. Only the air density and the
swept velocity range are changed, so that the answer is directly comparable with
the AEGIS model at sea level.

Earlier attempts to write this case from scratch failed in the linear assembly
with a shape mismatch, because four things their configuration carries were
missing: the ``ScalingDict`` non-dimensionalisation, ``remove_sym_modes`` (a
full-span wing has symmetric and antisymmetric mode pairs), removal of the gust
input, and the Krylov ROM. Starting from the working configuration and changing
one thing is the reliable way to do this.

What the comparison means. Both models use identical structural data. AEGIS
represents the air with 2D strip theory plus a two-state Wagner lag, which has no
tip relief and therefore over-predicts loading near the tip; UVLM resolves the
finite span. Strip theory should give the lower flutter speed, and the size of
the gap is the quantity of interest.

Usage (inside the container)::

    python goland_uvlm_flutter.py --density 1.225 --u-min 120 --u-max 180
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

AEGIS_FLUTTER_SPEED = 137.35
AEGIS_FLUTTER_OMEGA = 69.34


def build(args, route: str, case_name: str):
    import sharpy.cases.templates.flying_wings as wings

    # The case is linearised at unit speed and non-dimensionalised through
    # ScalingDict; the velocity sweep is then done on the linear system.
    reference_speed = 1.0

    rom_settings = {
        "algorithm": "mimo_rational_arnoldi",
        "r": 6,
        "single_side": "observability",
        "frequency": np.array([0.0]),
    }

    wing = wings.GolandControlSurface(
        M=args.chordwise,
        N=args.spanwise,
        Mstar_fact=args.wake_factor,
        u_inf=reference_speed,
        alpha=0.0,
        cs_deflection=[0.0, 0.0],
        n_control_surfaces=2,
        rho=args.density,
        sweep=0,
        physical_time=2,
        n_surfaces=2,
        route=route,
        case_name=case_name,
    )
    wing.gust_intensity = 0.01
    wing.sigma = 1

    wing.clean_test_files()
    wing.update_derived_params()
    wing.set_default_config_dict()
    wing.generate_aero_file()
    wing.generate_fem_file()

    rom_settings["tangent_input_file"] = route + "/" + case_name + ".rom.h5"

    wing.config["SHARPy"] = {
        "flow": [
            "BeamLoader",
            "AerogridLoader",
            "StaticCoupled",
            "Modal",
            "LinearAssembler",
            "AsymptoticStability",
        ],
        "case": case_name,
        "route": route,
        "write_screen": "off",
        "write_log": "on",
        "log_folder": route + "/output/",
        "log_file": case_name + ".log",
    }
    wing.config["BeamLoader"] = {"unsteady": "off", "orientation": wing.quat}
    wing.config["AerogridLoader"] = {
        "unsteady": "off",
        "aligned_grid": "on",
        "mstar": wing.Mstar_fact * wing.M,
        "freestream_dir": wing.u_inf_direction,
        "wake_shape_generator": "StraightWake",
        "wake_shape_generator_input": {
            "u_inf": reference_speed,
            "u_inf_direction": wing.u_inf_direction,
            "dt": wing.dt,
        },
    }
    aero_settings = {
        "rho": wing.rho,
        "print_info": "off",
        "horseshoe": "off",
        "num_cores": args.cores,
        "n_rollup": 0,
        "rollup_dt": wing.dt,
        "rollup_aic_refresh": 1,
        "rollup_tolerance": 1e-4,
        "velocity_field_generator": "SteadyVelocityField",
        "velocity_field_input": {
            "u_inf": reference_speed,
            "u_inf_direction": wing.u_inf_direction,
        },
    }
    wing.config["StaticUvlm"] = dict(aero_settings)
    wing.config["StaticCoupled"] = {
        "print_info": "off",
        "max_iter": 200,
        "n_load_steps": 1,
        "tolerance": 1e-10,
        "relaxation_factor": 0.0,
        "aero_solver": "StaticUvlm",
        "aero_solver_settings": aero_settings,
        "structural_solver": "NonLinearStatic",
        "structural_solver_settings": {
            "print_info": "off",
            "max_iterations": 150,
            "num_load_steps": 4,
            "delta_curved": 1e-1,
            "min_delta": 1e-10,
            "gravity_on": "on",
            "gravity": 9.81,
        },
    }
    wing.config["Modal"] = {
        "NumLambda": 20,
        "rigid_body_modes": "off",
        "print_matrices": "off",
        "save_data": "off",
        "rigid_modes_cg": "off",
        "continuous_eigenvalues": "off",
        "dt": 0,
        "plot_eigenvalues": False,
        "max_rotation_deg": 15.0,
        "max_displacement": 0.15,
        "write_modes_vtk": False,
        "use_undamped_modes": True,
    }
    wing.config["LinearAssembler"] = {
        "linear_system": "LinearAeroelastic",
        "linear_system_settings": {
            "beam_settings": {
                "modal_projection": "on",
                "inout_coords": "modes",
                "discrete_time": "on",
                "newmark_damp": 0.5e-4,
                "discr_method": "newmark",
                "dt": wing.dt,
                "proj_modes": "undamped",
                "use_euler": "off",
                "num_modes": args.modes,
                "print_info": "off",
                "gravity": "on",
                "remove_sym_modes": "on",
                "remove_dofs": [],
            },
            "aero_settings": {
                "dt": wing.dt,
                "ScalingDict": {
                    "length": 0.5 * wing.c_ref,
                    "speed": reference_speed,
                    "density": args.density,
                },
                "integr_order": 2,
                "density": wing.rho,
                "remove_predictor": False,
                "use_sparse": True,
                "remove_inputs": ["u_gust"],
                "rom_method": ["Krylov"],
                "rom_method_settings": {"Krylov": rom_settings},
            },
        },
    }
    wing.config["AsymptoticStability"] = {
        "print_info": True,
        "velocity_analysis": [args.u_min, args.u_max, args.u_points],
    }
    wing.config.write()
    return wing


def main() -> None:
    args = _parse_args()
    import sharpy.sharpy_main

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    case_name = f"goland_uvlm_rho{int(args.density * 1000):04d}"
    route = str(workdir / case_name)
    os.makedirs(route, exist_ok=True)

    wing = build(args, route, case_name)
    sharpy.sharpy_main.main(["", route + "/" + case_name + ".sharpy"])

    speeds, growth, frequency = _read_velocity_analysis(Path(route) / "output")
    if speeds.size == 0:
        print("no velocity analysis output found")
        return

    print("\n  speed [m/s]   max Re(lambda)   omega [rad/s]")
    for u, g, w in zip(speeds, growth, frequency):
        print(f"  {u:10.2f}   {g:+13.4f}   {w:11.2f}")

    flutter_speed, flutter_omega = _crossing(speeds, growth, frequency)
    payload = {
        "density": args.density,
        "speeds": speeds.tolist(),
        "max_real": growth.tolist(),
        "frequency_rad_s": frequency.tolist(),
        "flutter_speed": flutter_speed,
        "flutter_omega": flutter_omega,
        "aegis_flutter_speed": AEGIS_FLUTTER_SPEED,
        "aegis_flutter_omega": AEGIS_FLUTTER_OMEGA,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if flutter_speed is None:
        print("\n  no crossing inside the swept range")
    else:
        difference = 100.0 * (flutter_speed - AEGIS_FLUTTER_SPEED) / AEGIS_FLUTTER_SPEED
        print(f"\n  SHARPy UVLM      flutter = {flutter_speed:.2f} m/s "
              f"at {flutter_omega:.1f} rad/s")
        print(f"  AEGIS strip      flutter = {AEGIS_FLUTTER_SPEED:.2f} m/s "
              f"at {AEGIS_FLUTTER_OMEGA:.1f} rad/s")
        print(f"  strip theory is {difference:+.1f} % relative to UVLM")
    print(f"\nwrote {args.out}")


def _read_velocity_analysis(output_dir: Path):
    """Read the velocity-analysis table AsymptoticStability writes."""
    for path in sorted(output_dir.rglob("*velocity_analysis*")):
        try:
            raw = np.loadtxt(path)
        except (OSError, ValueError):
            continue
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.shape[1] >= 3:
            return raw[:, 0], raw[:, 1], raw[:, 2]
    return np.asarray([]), np.asarray([]), np.asarray([])


def _crossing(speeds, growth, frequency):
    for index in range(len(speeds) - 1):
        if growth[index] <= 0.0 < growth[index + 1]:
            span = growth[index + 1] - growth[index]
            weight = -growth[index] / span if span else 0.0
            speed = speeds[index] + weight * (speeds[index + 1] - speeds[index])
            return float(speed), float(frequency[index])
    return None, None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", type=float, default=1.225)
    parser.add_argument("--chordwise", type=int, default=16)
    parser.add_argument("--spanwise", type=int, default=32)
    parser.add_argument("--wake-factor", type=int, default=10)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--u-min", type=float, default=120.0)
    parser.add_argument("--u-max", type=float, default=190.0)
    parser.add_argument("--u-points", type=int, default=71)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--workdir", default="/work/uvlm_cases")
    parser.add_argument("--out", default="/work/goland_uvlm_flutter.json")
    return parser.parse_args()


if __name__ == "__main__":
    main()
