"""Print the accepted settings of SHARPy's linear aeroelastic system classes.

The system registry is populated lazily, so the classes are imported directly.
One of them has a misspelled ``settins_default`` attribute in this release, which
is handled rather than allowed to abort the enumeration.
"""

import importlib

MODULES = {
    "LinearAeroelastic": "sharpy.linear.assembler.linearaeroelastic",
    "LinearBeam": "sharpy.linear.assembler.linearbeam",
    "LinearUVLM": "sharpy.linear.assembler.linearuvlm",
}


def settings_of(cls) -> tuple[dict, dict]:
    defaults = getattr(cls, "settings_default", None)
    if defaults is None:
        defaults = getattr(cls, "settins_default", {}) or {}
    return defaults, getattr(cls, "settings_types", {}) or {}


for name, module_path in MODULES.items():
    try:
        module = importlib.import_module(module_path)
    except Exception as error:  # noqa: BLE001
        print(f"=== {name}: import failed ({type(error).__name__}: {error})\n")
        continue
    cls = getattr(module, name, None)
    if cls is None:
        candidates = [a for a in dir(module) if a.lower().startswith("linear")]
        print(f"=== {name}: class not found; candidates {candidates}\n")
        continue
    defaults, types = settings_of(cls)
    print(f"=== {name}  ({cls.__module__})")
    for key in sorted(defaults):
        print(f"  {key:30s} {str(types.get(key, '?')):12s} default={defaults[key]!r}")
    print()
