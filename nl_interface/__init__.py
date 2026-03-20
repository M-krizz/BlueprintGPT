"""Natural-language specification interface for BlueprintGPT.

Heavy imports (runner with shapely deps) are deferred to avoid import errors
in environments without all dependencies installed.
"""

# Light imports that don't require shapely/torch
from nl_interface.adapter import build_backend_spec, route_backend, validate_resolution
from nl_interface.service import blank_current_spec, normalize_current_spec, process_user_request

# Lazy import for heavy dependencies
def __getattr__(name):
    """Lazy import for modules requiring heavy dependencies."""
    if name in ("execute_response", "run_algorithmic_backend", "run_learned_backend"):
        from nl_interface import runner
        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "blank_current_spec",
    "build_backend_spec",
    "execute_response",
    "normalize_current_spec",
    "process_user_request",
    "route_backend",
    "run_algorithmic_backend",
    "run_learned_backend",
    "validate_resolution",
]
