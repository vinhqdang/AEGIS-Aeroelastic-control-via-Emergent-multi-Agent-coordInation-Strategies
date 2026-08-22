"""Check the control-surface effectiveness AEGIS assumes, against UVLM.

This targets the weakest documented assumption in the AEGIS plant. Control
surfaces there use quasi-steady thin-airfoil flap derivatives from the Glauert
series -- for a hinge at 75 per cent chord, Cl_delta = 3.826 per radian -- and the
flap's own shed wake is neglected. If UVLM disagrees materially, then the learned
policies were trained against control authority the real flow does not provide,
and every result that depends on them is suspect.

The measurement only needs a static UVLM solve at a sequence of flap deflections,
so it avoids the linear-assembly machinery entirely and runs in seconds rather
than tens of minutes. Lift is differenced against the undeflected case and
converted to a coefficient using the same reference area and dynamic pressure the
AEGIS model uses.

Usage (inside the container)::

    python flap_effectiveness.py --deflections -4 -2 0 2 4 --out flap.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Thin-airfoil value AEGIS uses, for a plain flap hinged at 75 per cent chord.
AEGIS_CL_DELTA = 3.826


def build_case(deflection_deg: float, args, route: str, case_name: str):
    import sharpy.cases.templates.flying_wings as wings

    wing = wings.GolandControlSurface(
        M=args.chordwise,
        N=args.spanwise,
        Mstar_fact=args.wake_factor,
        u_inf=args.speed,
        alpha=args.alpha,
        cs_deflection=[deflection_deg] * args.n_surfaces,
        n_control_surfaces=args.n_surfaces,
        rho=args.density,
        # 0.25 puts the hinge at 75 per cent chord, matching the AEGIS layout.
        pct_flap=0.25,
        physical_time=0.1,
        n_surfaces=2,
        route=route,
        case_name=case_name,
    )
    wing.sigma = 1.0
    wing.clean_test_files()
    wing.update_derived_params()
    wing.set_default_config_dict()
    wing.generate_aero_file()
    wing.generate_fem_file()

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
            "u_inf": args.speed,
            "u_inf_direction": wing.u_inf_direction,
        },
    }

    wing.config["SHARPy"] = {
        "flow": ["BeamLoader", "AerogridLoader", "StaticCoupled", "AeroForcesCalculator"],
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
            "u_inf": args.speed,
            "u_inf_direction": wing.u_inf_direction,
            "dt": wing.dt,
        },
    }
    wing.config["StaticUvlm"] = dict(aero_settings)
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
            "gravity_on": "off",
        },
    }
    wing.config["AeroForcesCalculator"] = {
        "write_text_file": "off",
        "screen_output": "off",
    }
    wing.config.write()
    return wing


def run_deflection(deflection_deg: float, args, workdir: Path) -> dict:
    import sharpy.sharpy_main

    case_name = f"flap{int(round(deflection_deg * 100)):+07d}".replace("+", "p").replace("-", "m")
    route = str(workdir / case_name)
    os.makedirs(route, exist_ok=True)

    build_case(deflection_deg, args, route, case_name)
    data = sharpy.sharpy_main.main(["", route + "/" + case_name + ".sharpy"])

    step = data.aero.timestep_info[-1]
    # Total aerodynamic force, summed over every panel of every surface, in the
    # inertial frame. Index 2 is the vertical component, i.e. lift.
    lift = 0.0
    for surface_forces in step.forces:
        lift += float(np.sum(surface_forces[2, :, :]))
    return {"deflection_deg": float(deflection_deg), "lift_N": lift}


def main() -> None:
    args = _parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    results = []
    for deflection in args.deflections:
        try:
            entry = run_deflection(deflection, args, workdir)
            print(f"  delta = {deflection:+6.2f} deg   lift = {entry['lift_N']:+12.2f} N",
                  flush=True)
        except Exception as error:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            entry = {"deflection_deg": float(deflection),
                     "error": f"{type(error).__name__}: {error}"}
        results.append(entry)

    _summarise(results, args)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


def _summarise(results: list[dict], args) -> None:
    good = [r for r in results if "error" not in r]
    if len(good) < 2:
        print("\nnot enough successful points to fit a slope")
        return

    angles = np.deg2rad([r["deflection_deg"] for r in good])
    lift = np.asarray([r["lift_N"] for r in good])

    # Both wings are modelled, so the reference area is the full span.
    span = 2.0 * 6.096
    chord = 1.8288
    area = span * chord
    dynamic_pressure = 0.5 * args.density * args.speed**2
    coefficient = lift / (dynamic_pressure * area)

    slope = float(np.polyfit(angles, coefficient, 1)[0])
    print(f"\n  UVLM   dCL/d(delta) = {slope:.3f} per rad")
    print(f"  AEGIS  Cl_delta      = {AEGIS_CL_DELTA:.3f} per rad (thin-airfoil, Glauert)")
    print(f"  ratio  UVLM / AEGIS  = {slope / AEGIS_CL_DELTA:.3f}")
    print("\n  A ratio well below 1 means AEGIS over-estimates control authority,")
    print("  which is the documented risk of the quasi-steady flap assumption.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deflections", type=float, nargs="+",
                        default=[-4.0, -2.0, 0.0, 2.0, 4.0])
    parser.add_argument("--speed", type=float, default=137.35)
    parser.add_argument("--density", type=float, default=1.225)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--chordwise", type=int, default=8)
    parser.add_argument("--spanwise", type=int, default=16)
    parser.add_argument("--wake-factor", type=int, default=5)
    parser.add_argument("--n-surfaces", type=int, default=2)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--workdir", default="/work/flap_cases")
    parser.add_argument("--out", default="/work/flap_effectiveness.json")
    return parser.parse_args()


if __name__ == "__main__":
    main()
