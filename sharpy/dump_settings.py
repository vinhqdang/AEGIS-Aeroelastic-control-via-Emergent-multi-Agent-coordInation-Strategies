"""Print the settings each SHARPy solver actually accepts, for this version.

Solver module paths move between SHARPy releases, so everything is resolved
through the solver registry rather than by importing a guessed module path.
"""

import sharpy.utils.solver_interface as solver_interface

WANTED = (
    "LinearAssembler",
    "AsymptoticStability",
    "Modal",
    "StaticCoupled",
    "LinearAeroelastic",
    "LinearBeam",
    "LinearUVLM",
)


def show(name: str) -> None:
    try:
        cls = solver_interface.solver_from_string(name)
    except Exception as error:  # noqa: BLE001 - report and keep going
        print(f"=== {name}: not in registry ({type(error).__name__})\n")
        return
    defaults = getattr(cls, "settings_default", {})
    types = getattr(cls, "settings_types", {})
    print(f"=== {name}  ({cls.__module__})")
    for key in sorted(defaults):
        print(f"  {key:30s} {str(types.get(key, '?')):10s} default={defaults[key]!r}")
    print()


registry = solver_interface.dictionary_of_solvers(print_info=False)
print("registry entries containing 'inear':",
      sorted(k for k in registry if "inear" in k))
print("registry entries containing 'tability':",
      sorted(k for k in registry if "tability" in k))
print()

for name in WANTED:
    show(name)

# The linear-system classes are registered separately from solvers.
try:
    import sharpy.linear.utils.ss_interface as ss_interface

    systems = ss_interface.dictionary_of_systems()
    print("linear systems:", sorted(systems))
    for name in ("LinearAeroelastic", "LinearBeam", "LinearUVLM"):
        cls = systems.get(name)
        if cls is None:
            continue
        print(f"\n=== system {name} ({cls.__module__})")
        defaults = getattr(cls, "settings_default", {})
        types = getattr(cls, "settings_types", {})
        for key in sorted(defaults):
            print(f"  {key:30s} {str(types.get(key, '?')):10s} default={defaults[key]!r}")
except Exception as error:  # noqa: BLE001
    print("could not enumerate linear systems:", type(error).__name__, error)
