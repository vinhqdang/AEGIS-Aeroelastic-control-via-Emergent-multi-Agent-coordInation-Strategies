"""Confirm the container holds the real SHARPy aeroelastic solver.

Worth doing explicitly: the PyPI package named ``sharpy`` is an unrelated billing
API client, so a bare successful ``import sharpy`` proves nothing. This checks for
the aeroelastic solver registry and the beam/UVLM extensions specifically.
"""

import sys


def main() -> int:
    import sharpy

    print("sharpy package:", sharpy.__file__)
    print("python:", sys.version.split()[0])

    import sharpy.utils.solver_interface as solver_interface

    solver_interface.output_documentation = lambda *a, **k: None
    solvers = solver_interface.dictionary_of_solvers(print_info=False)
    print("registered solvers:", len(solvers))

    wanted = [
        "StaticCoupled",
        "DynamicCoupled",
        "BeamLoader",
        "AerogridLoader",
        "StepUvlm",
        "NonLinearDynamicCoupledStep",
    ]
    missing = [name for name in wanted if name not in solvers]
    for name in wanted:
        print(f"  {'ok ' if name in solvers else 'MISSING'} {name}")

    # The compiled libraries are the real test: these are the Fortran beam solver
    # and the C++ UVLM library, which the impostor package obviously lacks.
    import sharpy.structure.models.beam as beam_module
    import sharpy.aero.models.aerogrid as aerogrid_module

    print("beam module:", beam_module.__name__)
    print("aerogrid module:", aerogrid_module.__name__)

    if missing:
        print("FAIL: missing solvers", missing)
        return 1
    print("VERIFIED: real SHARPy aeroelastic solver")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
