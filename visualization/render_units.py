from dataclasses import dataclass


_FT_TO_M = 0.3048


@dataclass(frozen=True)
class RenderUnitsConfig:
    unit: str = "m"
    wall_snap_step: float = 0.15
    door_gap_eps: float = 0.10
    door_attach_tol: float = 0.06
    window_gap_eps: float = 0.08
    window_corner_margin: float = 0.25
    window_min_length: float = 0.55


def _to_metres(value: float, unit: str) -> float:
    if unit == "ft":
        return value * _FT_TO_M
    return value


def resolve_render_units(unit: str = "m", wall_snap_step: float | None = None) -> RenderUnitsConfig:
    """Build a unit-aware tolerance config with all values derived from wall snap step.

    All values returned are in metres because the internal geometry pipeline is metre-based.
    """
    normalized = (unit or "m").lower()
    if normalized not in {"m", "ft"}:
        normalized = "m"

    base_step = wall_snap_step if wall_snap_step is not None else (0.15 if normalized == "m" else 0.5)
    base_step_m = _to_metres(base_step, normalized)

    return RenderUnitsConfig(
        unit=normalized,
        wall_snap_step=base_step_m,
        door_gap_eps=max(base_step_m * 0.67, 0.025),
        door_attach_tol=max(base_step_m * 0.40, 0.02),
        window_gap_eps=max(base_step_m * 0.55, 0.02),
        window_corner_margin=max(base_step_m * 1.65, 0.18),
        window_min_length=max(base_step_m * 3.6, 0.45),
    )
