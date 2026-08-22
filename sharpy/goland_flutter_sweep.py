"""Independent flutter check on the Goland wing using SHARPy's UVLM solver.

Runs inside the SHARPy Docker image. The point is a cross-validation with
genuinely different aerodynamics: AEGIS models the air with strip theory plus a
two-state Wagner lag and a modal Galerkin structure, while SHARPy uses an
unsteady vortex-lattice method coupled to a geometrically exact beam.

The structural data is the same in both -- SHARPy's own Goland template carries
GJ = 0.987581e6, EI = 9.77221e6, m = 35.71 kg/m, J = 8.64 kg m, elastic axis at
33 per cent and centre of gravity at 43 per cent chord, which is exactly the
AEGIS wing. So a disagreement in flutter speed is a disagreement about the
aerodynamics, which is the thing worth testing.

For each freestream speed the script builds the case, trims it, projects onto
structural modes, assembles the linearised aeroelastic system and reads its
eigenvalues. Flutter is the lowest speed at which any eigenvalue has positive
real part.

Usage (inside the container)::

    python goland_flutter_sweep.py --speeds 120 130 140 150 --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from pathlib import Path

import numpy as np

# AEGIS reference values, for the comparison printed at the end.
AEGIS_FLUTTER_SPEED = 137.35
AEGIS_FLUTTER_OMEGA = 69.34


def build_case(speed: float, args, route: str, case_name: str):
    """Configure a Goland case linearised about its equilibrium at ``speed``."""
    import sharpy.cases.templates.flying_wings as wings

    wing = wings.GolandControlSurface(
        M=args.chordwise,
        N=args.spanwise,
        Mstar_fact=args.wake_factor,
        u_inf=speed,
        alpha=0.0,
        cs_deflection=[0.0] * args.n_surfaces,
        n_control_surfaces=args.n_surfaces,
        rho=args.density,
        physical_time=1.0,
        n_surfaces=args.aero_surfaces,
        route=route,
        case_name=case_name,
    )
    wing.sigma = 1.0
    wing.clean_test_files()
    wing.update_derived_params()
    wing.set_default_config_dict()
    wing.generate_aero_file()
    wing.generate_fem_file()

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
        "mstar": args.wake_factor * args.chordwise,
        "freestream_dir": wing.u_inf_direction,
        "wake_shape_generator": "StraightWake",
        "wake_shape_generator_input": {
            "u_inf": speed,
            "u_inf_direction": wing.u_inf_direction,
            "dt": wing.dt,
        },
    }
    aero_settings = {
        "rho": args.density,
        "print_info": "off",
        "horseshoe": "off",
        "num_cores": args.cores,
        "n_rollup": 0,
        "rollup_dt": wing.dt,
        "rollup_aic_refresh": 1,
        "rollup_tolerance": 1e-4,
        "velocity_field_generator": "SteadyVelocityField",
        "velocity_field_input": {
            "u_inf": speed,
            "u_inf_direction": wing.u_inf_direction,
        },
    }
    wing.config["StaticUvlm"] = dict(aero_settings, print_info="off")
    wing.config["StaticCoupled"] = {
        "print_info": "off",
        "max_iter": 200,
        "n_load_steps": 1,
        "tolerance": 1e-8,
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
            # Gravity off: the AEGIS model has no static aeroelastic droop, so
            # leaving it on would linearise about a different equilibrium and
            # confound the aerodynamic comparison with a trim difference.
            "gravity_on": "off",
        },
    }
    wing.config["Modal"] = {
        "NumLambda": args.modes,
        "rigid_body_modes": "off",
        "print_matrices": "off",
        "save_data": "off",
        "continuous_eigenvalues": "off",
        "dt": 0,
        "plot_eigenvalues": False,
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
                "gravity": "off",
                "remove_dofs": [],
            },
            "aero_settings": {
                "dt": wing.dt,
                "integr_order": 2,
                "density": args.density,
                "remove_predictor": False,
                "use_sparse": True,
                "vortex_radius": 1e-6,
                "convert_to_ct": "on",
            },
            "use_euler": "off",
            "track_body": "off",
        },
    }
    wing.config["AsymptoticStability"] = {
        "print_info": True,
        "export_eigenvalues": True,
        "num_evals": args.n_eigenvalues,
        "reference_velocity": speed,
        # [u_min, u_max, n_points]: sweeps the freestream from one linearisation
        # rather than re-trimming the case at every speed.
        "velocity_analysis": [args.sweep_min, args.sweep_max, args.sweep_points],
        "frequency_cutoff": args.frequency_cutoff,
    }
    wing.config.write()
    return wing


def run_speed(speed: float, args, workdir: Path) -> dict:
    import sharpy.sharpy_main

    case_name = f"goland_u{int(round(speed * 100)):06d}"
    route = str(workdir / case_name)
    os.makedirs(route, exist_ok=True)

    wing = build_case(speed, args, route, case_name)
    data = sharpy.sharpy_main.main(["", route + "/" + case_name + ".sharpy"])

    eigenvalues = _read_eigenvalues(Path(route) / "output", case_name, wing.dt)
    if eigenvalues.size == 0:
        return {"speed": speed, "error": "no eigenvalues produced"}

    worst = eigenvalues[int(np.argmax(eigenvalues.real))]
    del data
    return {
        "speed": float(speed),
        "max_real": float(worst.real),
        "frequency_rad_s": float(abs(worst.imag)),
        "n_eigenvalues": int(eigenvalues.size),
    }


def _read_eigenvalues(output_dir: Path, case_name: str, dt: float) -> np.ndarray:
    """Locate the exported eigenvalues and convert to continuous time if needed."""
    candidates = list(output_dir.rglob("eigenvalues.dat")) + list(
        output_dir.rglob("stability/*.dat")
    )
    for path in candidates:
        try:
            raw = np.loadtxt(path)
        except (OSError, ValueError):
            continue
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.shape[1] < 2:
            continue
        values = raw[:, 0] + 1j * raw[:, 1]
        # SHARPy exports continuous-time eigenvalues from AsymptoticStability.
        # A discrete-time export would sit on/inside the unit circle; detect that
        # and convert, rather than silently mixing the two conventions.
        if np.all(np.abs(values) <= 1.0 + 1e-6) and np.max(np.abs(values.real)) <= 1.0:
            values = np.log(values + 1e-300) / dt
        return values
    return np.asarray([])


def main() -> None:
    args = _parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    results = []
    for speed in args.speeds:
        print(f"--- u_inf = {speed:.2f} m/s", flush=True)
        try:
            entry = run_speed(speed, args, workdir)
        except Exception as error:  # noqa: BLE001 - report and continue the sweep
            # Print the full traceback: a bare message like a shape mismatch is
            # useless without knowing which coupling matrix produced it.
            traceback.print_exc()
            entry = {"speed": float(speed), "error": f"{type(error).__name__}: {error}"}
        results.append(entry)
        if "error" in entry:
            print(f"    FAILED: {entry['error']}", flush=True)
        else:
            print(
                f"    max Re(lambda) = {entry['max_real']:+.4f}   "
                f"omega = {entry['frequency_rad_s']:.2f} rad/s",
                flush=True,
            )
        if args.clean:
            shutil.rmtree(workdir / f"goland_u{int(round(speed * 100)):06d}", ignore_errors=True)

    _summarise(results)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


def _summarise(results: list[dict]) -> None:
    good = [r for r in results if "error" not in r]
    if len(good) < 2:
        print("\nnot enough successful points to bracket flutter")
        return
    good.sort(key=lambda r: r["speed"])
    print("\n  speed [m/s]   max Re(lambda)   omega [rad/s]")
    for entry in good:
        print(
            f"  {entry['speed']:10.2f}   {entry['max_real']:+13.4f}   "
            f"{entry['frequency_rad_s']:11.2f}"
        )

    crossing = None
    for previous, current in zip(good, good[1:]):
        if previous["max_real"] <= 0.0 < current["max_real"]:
            span = current["max_real"] - previous["max_real"]
            weight = -previous["max_real"] / span if span else 0.0
            crossing = previous["speed"] + weight * (current["speed"] - previous["speed"])
            omega = previous["frequency_rad_s"]
            break

    if crossing is None:
        print("\n  no sign change inside the swept range")
        return
    print(f"\n  SHARPy (UVLM) flutter speed  ~ {crossing:.2f} m/s at ~{omega:.1f} rad/s")
    print(f"  AEGIS  (strip theory)        = {AEGIS_FLUTTER_SPEED:.2f} m/s "
          f"at {AEGIS_FLUTTER_OMEGA:.1f} rad/s")
    print(f"  difference                   = {100 * (crossing - AEGIS_FLUTTER_SPEED) / AEGIS_FLUTTER_SPEED:+.1f} %")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speeds", type=float, nargs="+",
                        default=[110.0, 125.0, 140.0, 155.0, 170.0])
    parser.add_argument("--density", type=float, default=1.225)
    parser.add_argument("--chordwise", type=int, default=8, help="M panels")
    parser.add_argument("--spanwise", type=int, default=16, help="N panels")
    parser.add_argument("--wake-factor", type=int, default=5)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--n-surfaces", type=int, default=3)
    parser.add_argument("--n-eigenvalues", type=int, default=60)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--workdir", default="/work/cases")
    parser.add_argument("--out", default="/work/sharpy_flutter.json")
    parser.add_argument("--aero-surfaces", type=int, default=2,
                        help="1 = half wing, 2 = both wings (SHARPy test uses 2)")
    parser.add_argument("--sweep-min", type=float, default=80.0)
    parser.add_argument("--sweep-max", type=float, default=220.0)
    parser.add_argument("--sweep-points", type=int, default=57)
    parser.add_argument("--frequency-cutoff", type=float, default=0.0)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
