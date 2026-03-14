"""Natural-language specification interface for BlueprintGPT."""

from nl_interface.adapter import build_backend_spec, route_backend, validate_resolution
from nl_interface.runner import execute_response, run_algorithmic_backend, run_learned_backend
from nl_interface.service import blank_current_spec, normalize_current_spec, process_user_request

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
