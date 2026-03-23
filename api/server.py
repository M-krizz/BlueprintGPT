from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import List, Literal, Optional
from uuid import uuid4

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"[ENV] Loaded environment from {env_path}")
except ImportError:
    pass  # dotenv not installed, use system env vars

import anyio
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from nl_interface.runner import execute_response
from nl_interface.service import process_user_request, normalize_current_spec
from nl_interface.runner import run_algorithmic_backend, run_planner_backend, run_learned_backend, run_hybrid_backend
from nl_interface.conversation import conversation_manager, ConversationSession
from nl_interface.chat_spec_adapter import default_chat_spec_adapter
from nl_interface.conversation_orchestrator import ConversationOrchestrator
from nl_interface.contracts import GenerationOutcome
from nl_interface.gemini_adapter import (
    extract_spec_from_nl,
    chat_response as gemini_chat,
    is_available as gemini_available,
    process_message as process_nl_message,
    INTENT_DESIGN,
    INTENT_CORRECTION,
    INTENT_QUESTION,
    INTENT_CONVERSATION,
)
from nl_interface.explainer import explain_ranked_designs, generate_comparison_explanation
from nl_interface.correction_handler import handle_correction_request

# Import centralized configuration and logging
from config.constants import (
    RoomTypes, DefaultDimensions, IntentTypes, APIDefaults
)
from utils.processing_logger import ProcessingLogger, DetailedLogger, logger

# Verify intent constants match at startup (prevents subtle comparison bugs)
assert INTENT_DESIGN == IntentTypes.DESIGN, f"Intent constant mismatch: {INTENT_DESIGN!r} != {IntentTypes.DESIGN!r}"
assert INTENT_CORRECTION == IntentTypes.CORRECTION, f"Intent constant mismatch: {INTENT_CORRECTION!r} != {IntentTypes.CORRECTION!r}"
assert INTENT_QUESTION == IntentTypes.QUESTION, f"Intent constant mismatch: {INTENT_QUESTION!r} != {IntentTypes.QUESTION!r}"
assert INTENT_CONVERSATION == IntentTypes.CONVERSATION, f"Intent constant mismatch: {INTENT_CONVERSATION!r} != {IntentTypes.CONVERSATION!r}"


class RoomSpec(BaseModel):
    name: str = Field(..., description="Room identifier, e.g. Bedroom_1")
    type: str = Field(..., description="Room type, e.g. Bedroom, Kitchen")
    area: Optional[float] = Field(None, description="Optional target area in square meters")


class Boundary(BaseModel):
    width: float
    height: float


class GenerateRequest(BaseModel):
    backend_target: Literal["algorithmic", "planner", "learned", "hybrid"] = "hybrid"
    boundary: Boundary
    boundary_polygon: Optional[List[List[float]]] = None
    entrance_point: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    rooms: List[RoomSpec]
    preferences: Optional[dict] = Field(default=None, description="User preference weights")
    output_prefix: Optional[str] = Field(default=None, description="Optional output prefix for artifacts")


class GenerateResponse(BaseModel):
    status: str
    backend_target: str
    strategy_name: Optional[str]
    report_status: Optional[str]
    explanation: Optional[dict]
    requested_rooms: Optional[dict] = None
    generated_rooms: Optional[dict] = None
    room_coverage: Optional[dict] = None
    violations: Optional[list] = None
    artifact_paths: dict
    artifact_urls: Optional[dict] = None
    metrics: dict
    generation_summary: Optional[dict] = None


class ChatGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    boundary: Boundary
    boundary_polygon: Optional[List[List[float]]] = None
    area: Optional[float] = None
    area_unit: str = "sq.m"
    entrance_point: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    output_prefix: Optional[str] = None
    checkpoint_path: Optional[str] = None


class ChatGenerateResponse(BaseModel):
    assistant_text: str
    backend_ready: bool
    backend_target: Optional[str]
    missing_fields: List[str]
    validation_errors: List[str]
    execution: Optional[dict] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    print("\n" + "="*80)
    print("  BlueprintGPT Server STARTED - logging enabled")
    print("  Endpoints:")
    print("    POST /conversation/session/new  -> create session")
    print("    POST /conversation/message      -> conversation API (primary)")
    print("    POST /chat/generate             -> legacy API (fallback, stateless)")
    print("="*80 + "\n")
    yield


app = FastAPI(title="BlueprintGPT API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NoCacheHTMLMiddleware(BaseHTTPMiddleware):
    """Force no-cache on HTML responses so browser always gets fresh frontend."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.endswith(".html") or path in ("/", "/ui", "/ui/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheHTMLMiddleware)

# Serve the lightweight UI under /ui
ui_dir = Path(__file__).parent.parent / "frontend"
outputs_dir = Path(__file__).parent.parent / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)


def _planner_runtime_status() -> dict:
    backend_mode = os.getenv("BLUEPRINT_BACKEND_MODE", "auto").lower()
    auto_core_policy = os.getenv("BLUEPRINT_AUTO_CORE_BACKEND", "algorithmic").lower()
    checkpoint_path = os.getenv("BLUEPRINTGPT_PLANNER_CHECKPOINT", "learned/planner/checkpoints/room_planner.pt")
    checkpoint_exists = Path(checkpoint_path).exists()

    if auto_core_policy == "planner":
        auto_core_backend = "planner"
    elif auto_core_policy in {"planner_if_available", "planner-if-available"} and checkpoint_exists:
        auto_core_backend = "planner"
    else:
        auto_core_backend = "algorithmic"

    return {
        "available": checkpoint_exists,
        "checkpoint_path": checkpoint_path,
        "configured_backend_mode": backend_mode,
        "auto_core_policy": auto_core_policy,
        "auto_core_backend": auto_core_backend,
    }


@app.get("/")
async def root():
    if ui_dir.exists():
        return RedirectResponse(url="/ui/")
    return {"status": "ok"}


# These routes are declared BEFORE app.mount so they take priority over StaticFiles
@app.get("/ui/", include_in_schema=False)
@app.get("/ui/index.html", include_in_schema=False)
async def serve_ui():
    """Serve the UI HTML fresh from disk every time — no browser caching."""
    html_path = ui_dir / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    content = html_path.read_text(encoding="utf-8")
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# Mount AFTER routes — static assets (CSS/JS/images) are served from /ui/*
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=False), name="ui")
app.mount("/outputs", StaticFiles(directory=str(outputs_dir), html=False), name="outputs")

conversation_orchestrator = ConversationOrchestrator(default_chat_spec_adapter())


def process_nl_message(user_message, context=None, conversation_history=None):
    return conversation_orchestrator.process_message(
        user_message,
        context=context or {},
        conversation_history=conversation_history or [],
    )


def extract_spec_from_nl(user_message, conversation_history=None):
    return conversation_orchestrator.extract_spec(
        user_message,
        conversation_history=conversation_history or [],
    )


def gemini_chat(user_message, context=None, conversation_history=None):
    return conversation_orchestrator.chat_reply(
        user_message,
        context=context or {},
        conversation_history=conversation_history or [],
    )


def _clear_prompt_scoped_resolution(resolution: Optional[dict]) -> Optional[dict]:
    if not resolution:
        return resolution
    cleaned = dict(resolution)
    if cleaned.get("boundary_source") == "nl_prompt":
        for key in ("boundary_size", "boundary_polygon", "entrance_point", "site_boundary_size", "boundary_role", "boundary_source"):
            cleaned.pop(key, None)
    if cleaned.get("total_area_source") == "nl_prompt":
        cleaned.pop("total_area", None)
        cleaned.pop("total_area_source", None)
    return cleaned


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/favicon.ico")
async def favicon():
    # HTTP 204 must not include a body per RFC 7231
    return Response(status_code=204)


def _spec_from_request(body: GenerateRequest) -> dict:
    boundary = body.boundary
    if body.boundary_polygon is not None:
        polygon = body.boundary_polygon
    elif boundary and boundary.width > 0 and boundary.height > 0:
        polygon = [
            (0.0, 0.0),
            (boundary.width, 0.0),
            (boundary.width, boundary.height),
            (0.0, boundary.height),
        ]
    else:
        polygon = None
    rooms = []
    for idx, room in enumerate(body.rooms, start=1):
        rooms.append({"name": room.name or f"Room_{idx}", "type": room.type, "area": room.area})

    spec = {
        "occupancy": "Residential",
        "rooms": rooms,
        "boundary_polygon": polygon,
        "entrance_point": body.entrance_point,
        "preference_weights": body.preferences or {},
    }
    return spec


async def _run_backend(body: GenerateRequest):
    spec = _spec_from_request(body)
    prefix = body.output_prefix or f"api_{uuid4().hex[:8]}"

    if body.backend_target == "algorithmic":
        return await anyio.to_thread.run_sync(
            partial(run_algorithmic_backend, spec, output_prefix=prefix),
        )

    if body.backend_target == "planner":
        return await anyio.to_thread.run_sync(
            partial(run_planner_backend, spec, output_prefix=prefix),
        )

    if body.backend_target == "learned":
        return await anyio.to_thread.run_sync(
            partial(run_learned_backend, spec, output_prefix=prefix),
        )

    if body.backend_target == "hybrid":
        # Note: We will import run_hybrid_backend shortly
        return await anyio.to_thread.run_sync(
            partial(run_hybrid_backend, spec, output_prefix=prefix),
        )

    raise HTTPException(status_code=400, detail="Unsupported backend_target")


def _artifact_urls(artifact_paths: Optional[dict]) -> dict:
    urls = {}
    for key, path in (artifact_paths or {}).items():
        if not path:
            continue
        normalized = str(path).replace("\\", "/")
        if normalized.startswith("outputs/"):
            urls[key] = "/" + normalized
        elif normalized.startswith("outputs"):
            urls[key] = "/" + normalized.replace("outputs", "outputs/", 1)
    return urls


def _format_room_counts(room_counts: Optional[dict]) -> str:
    if not room_counts:
        return "none"
    parts = []
    for room_type, count in room_counts.items():
        parts.append(f"{count} {room_type}")
    return ", ".join(parts)


def _format_design_brief(design_data: dict) -> str:
    requested = design_data.get("requested_rooms", {})
    generated = design_data.get("generated_rooms", {})
    coverage = design_data.get("room_coverage", {})
    metrics = design_data.get("metrics", {})
    generation_summary = design_data.get("generation_summary", {})
    engine = (
        design_data.get("winning_source")
        or design_data.get("backend_target")
        or "unknown"
    )

    lines = [
        "## Design Summary",
        f"- Engine: `{engine}`",
        f"- Compliance: `{design_data.get('report_status', 'UNKNOWN')}`",
        f"- Requested program: {_format_room_counts(requested)}",
        f"- Generated program: {_format_room_counts(generated)}",
    ]

    if coverage.get("missing"):
        lines.append(f"- Missing rooms: {', '.join(coverage['missing'])}")
    else:
        lines.append("- Room coverage: complete")

    if metrics.get("fully_connected") is not None:
        lines.append(f"- Fully connected: `{metrics.get('fully_connected')}`")
    if metrics.get("adjacency_satisfaction") is not None:
        lines.append(f"- Adjacency score: `{metrics.get('adjacency_satisfaction')}`")

    raw_valid = generation_summary.get("raw_valid_count")
    repaired_valid = generation_summary.get("repaired_valid_count")
    total_attempts = generation_summary.get("total_attempts")
    planner_summary = design_data.get("planner_summary", {})
    if planner_summary.get("source"):
        lines.append(f"- Planner guidance: `{planner_summary['source']}`")
    if raw_valid is not None and total_attempts:
        lines.append(f"- Raw valid samples: `{raw_valid}/{total_attempts}`")
    if engine == "learned" and repaired_valid:
        if raw_valid == 0:
            lines.append("- Quality note: the learned model needed deterministic repair to produce a usable layout.")
        elif raw_valid is not None and repaired_valid > raw_valid:
            lines.append("- Quality note: the learned model required repair on most shortlisted candidates.")

    return "\n".join(lines)


def _quality_band(value: float) -> str:
    if value >= 0.7:
        return "strong"
    if value >= 0.45:
        return "moderate"
    return "weak"


def _circulation_ratio(metrics: dict) -> float:
    total_area = float(metrics.get("total_area", 0.0) or 0.0)
    if total_area <= 0:
        return 0.0
    walkable_area = float(metrics.get("circulation_walkable_area", 0.0) or 0.0)
    return walkable_area / total_area


def _format_rule_sentence(source: str, target: str, relation: str) -> Optional[str]:
    relation = (relation or "near_to").strip().lower()
    if not source or not target:
        return None
    if relation in {"near_to", "near", "adjacent_to", "adjacent"}:
        return f"{source} should stay close to {target}."
    if relation == "buffer_zone":
        return f"{source} should be buffered from {target} for privacy."
    if relation == "separate":
        return f"{source} should be kept apart from {target} where possible."
    return f"{source} should respect a {relation} relationship with {target}."


def _layout_type_from_spec(spec: Optional[dict]) -> str:
    spec = spec or {}
    if spec.get("layout_type"):
        return str(spec["layout_type"])
    rooms = spec.get("rooms", [])
    bedroom_count = sum(int(room.get("count", 1)) for room in rooms if room.get("type") == "Bedroom")
    return f"{bedroom_count}BHK" if bedroom_count > 0 else "STUDIO"


def _derive_resolution_from_spec(spec: Optional[dict], resolution: Optional[dict], entrance_point: Optional[List[float]] = None) -> dict:
    from nl_interface.auto_dimension_selector import recommend_dimensions
    from nl_interface.constraint_analyzer import calculate_optimal_dimensions

    spec = spec or {}
    resolution = dict(resolution or {})
    rooms = spec.get("rooms", [])
    total_area = resolution.get("total_area")
    explicit_boundary_size = resolution.get("boundary_size")
    site_boundary_size = resolution.get("site_boundary_size")
    boundary_role = resolution.get("boundary_role")
    layout_type = _layout_type_from_spec(spec)
    auto_dimensions = spec.get("auto_dimensions") or {}

    def _base_footprint():
        if auto_dimensions.get("width_m") and auto_dimensions.get("height_m"):
            return float(auto_dimensions["width_m"]), float(auto_dimensions["height_m"])
        if rooms:
            dims = recommend_dimensions(rooms, building_type="residential")
            return float(dims[0]), float(dims[1])
        return 10.0, 8.0

    def _fit_within_site(width_value: float, height_value: float, site_size) -> tuple[float, float]:
        site_w, site_h = float(site_size[0]), float(site_size[1])
        if width_value <= site_w and height_value <= site_h:
            return width_value, height_value
        scale = min(site_w / max(width_value, 1e-6), site_h / max(height_value, 1e-6))
        if scale <= 0:
            return width_value, height_value
        return width_value * scale, height_value * scale

    if total_area:
        optimal = calculate_optimal_dimensions(layout_type, float(total_area))
        width, height = optimal["width_m"], optimal["height_m"]
        if site_boundary_size:
            width, height = _fit_within_site(float(width), float(height), site_boundary_size)
    elif boundary_role == "site" and explicit_boundary_size and len(explicit_boundary_size) == 2:
        site_boundary_size = tuple(explicit_boundary_size)
        width, height = _fit_within_site(*_base_footprint(), site_boundary_size)
    elif explicit_boundary_size and len(explicit_boundary_size) == 2:
        width, height = explicit_boundary_size
    elif site_boundary_size and len(site_boundary_size) == 2:
        width, height = _fit_within_site(*_base_footprint(), site_boundary_size)
    elif auto_dimensions.get("width_m") and auto_dimensions.get("height_m"):
        width = auto_dimensions["width_m"]
        height = auto_dimensions["height_m"]
    elif rooms:
        width, height = recommend_dimensions(rooms, building_type="residential")
    else:
        width, height = resolution.get("boundary_size", (10.0, 8.0))

    width = float(width)
    height = float(height)
    resolution["boundary_size"] = (width, height)
    if site_boundary_size and len(site_boundary_size) == 2:
        resolution["site_boundary_size"] = (float(site_boundary_size[0]), float(site_boundary_size[1]))
    resolution["boundary_polygon"] = [[0, 0], [width, 0], [width, height], [0, height]]
    resolution.setdefault("area_unit", "sq.m")
    resolution.setdefault("boundary_source", "derived")

    if entrance_point:
        resolution["entrance_point"] = tuple(entrance_point)
    elif not resolution.get("entrance_point"):
        entrance_side = spec.get("entrance_side")
        if entrance_side == "North":
            resolution["entrance_point"] = (width / 2.0, 0.0)
        elif entrance_side == "South":
            resolution["entrance_point"] = (width / 2.0, height)
        elif entrance_side == "East":
            resolution["entrance_point"] = (width, height / 2.0)
        elif entrance_side == "West":
            resolution["entrance_point"] = (0.0, height / 2.0)
        else:
            resolution["entrance_point"] = (width / 2.0, 0.0) if width <= height else (0.0, height / 2.0)

    return resolution


def _summarize_inferred_rules(spec: Optional[dict], resolution: Optional[dict]) -> str:
    spec = spec or {}
    resolution = resolution or {}
    lines = ["## Inferred Rules"]

    layout_type = spec.get("layout_type") or _layout_type_from_spec(spec)
    if layout_type:
        lines.append(f"- Program interpreted as `{layout_type}`.")

    total_area = resolution.get("total_area")
    site_boundary_size = resolution.get("site_boundary_size")
    if total_area is not None:
        lines.append(f"- Requested built-up area: `{float(total_area):.1f} sq.m`.")
    elif site_boundary_size:
        site_width, site_height = site_boundary_size
        lines.append(f"- Site plot size resolved to `{float(site_width):.1f}m x {float(site_height):.1f}m`.")
        if resolution.get("boundary_size"):
            width, height = resolution["boundary_size"]
            lines.append(f"- Working building footprint resolved to `{float(width):.1f}m x {float(height):.1f}m` within that plot.")
    elif resolution.get("boundary_size"):
        width, height = resolution["boundary_size"]
        lines.append(f"- Working plot size resolved to `{float(width):.1f}m x {float(height):.1f}m`.")

    if spec.get("entrance_side"):
        lines.append(f"- Entrance assumed on the `{spec['entrance_side']}` side.")

    adjacency = spec.get("adjacency") or spec.get("preferences", {}).get("adjacency", [])
    user_facing_rules = []
    heuristic_rules = []
    for item in adjacency[:4]:
        if isinstance(item, dict):
            sentence = _format_rule_sentence(item.get("source"), item.get("target"), item.get("relation", "near_to"))
            relation = str(item.get("relation", "near_to")).lower()
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            sentence = _format_rule_sentence(item[0], item[1], item[2])
            relation = str(item[2]).lower()
        else:
            sentence = None
            relation = ""

        if not sentence:
            continue
        if relation in {"near_to", "near", "adjacent_to", "adjacent"}:
            user_facing_rules.append(sentence)
        else:
            heuristic_rules.append(sentence)

    if user_facing_rules:
        lines.append("- Primary room relationships:")
        for sentence in user_facing_rules:
            lines.append(f"  - {sentence}")
    if heuristic_rules:
        lines.append("- Planning heuristics applied:")
        for sentence in heuristic_rules:
            lines.append(f"  - {sentence}")

    metadata = spec.get("constraint_metadata", {})
    if metadata.get("min_total_area_sqm") is not None:
        lines.append(f"- Minimum layout area: `{float(metadata['min_total_area_sqm']):.1f} sq.m`.")
    if metadata.get("recommended_area_sqm") is not None:
        lines.append(f"- Preferred layout area: `{float(metadata['recommended_area_sqm']):.1f} sq.m`.")

    return "\n".join(lines)


def _build_design_conversation_reply(design_data: dict, spec: dict, resolution: dict, design_count: int) -> str:
    metrics = design_data.get("metrics", {}) or {}
    generated = design_data.get("generated_rooms", {}) or {}
    layout_type = spec.get("layout_type") or _layout_type_from_spec(spec)
    adjacency = float(metrics.get("adjacency_satisfaction", 0.0) or 0.0)
    alignment = float(metrics.get("alignment_score", 0.0) or 0.0)
    circulation_ratio = _circulation_ratio(metrics)
    fully_connected = bool(metrics.get("fully_connected", False))
    compliance = design_data.get("report_status", "UNKNOWN")
    generated_program = _format_room_counts(generated)
    engine = design_data.get("winning_source") or design_data.get("backend_target") or "unknown"
    room_program = spec.get("room_program") or {}
    zoning_plan = spec.get("zoning_plan") or {}
    assumptions = []
    assumptions.extend((spec.get("semantic_spec") or {}).get("assumptions_used", []) or [])
    assumptions.extend(room_program.get("assumptions_used", []) or [])

    intro = (
        f"I generated {design_count} layout option{'s' if design_count != 1 else ''} for your `{layout_type}` request. "
        f"The selected design is `{compliance}` and came from the `{engine}` generation path."
    )
    coverage = f"It includes the interpreted program: {generated_program}."

    quality_notes: List[str] = []
    quality_notes.append(
        f"Connectivity is {'complete' if fully_connected else 'still incomplete'}, and the room-relationship fit is `{_quality_band(adjacency)}` "
        f"with an adjacency score of `{adjacency:.2f}`."
    )
    quality_notes.append(f"Geometric alignment landed in the `{_quality_band(alignment)}` range at `{alignment:.2f}`.")

    if circulation_ratio >= 0.16:
        quality_notes.append(
            f"Circulation is heavier than ideal at `{circulation_ratio:.2%}` of the plan area, which usually means the corridor is visually dominating the plan."
        )
    else:
        quality_notes.append(f"Circulation is within a usable range at `{circulation_ratio:.2%}` of plan area.")

    if metrics.get("architectural_reasonableness") is not None:
        quality_notes.append(
            f"Overall residential composition scored `{_quality_band(float(metrics.get('architectural_reasonableness', 0.0) or 0.0))}` "
            f"(`{float(metrics.get('architectural_reasonableness', 0.0) or 0.0):.2f}`) after checking frontage, privacy, and room relationships."
        )

    next_step = (
        "If you want, I can refine this exact design next by reducing corridor area, adjusting public/private zoning, "
        "or improving kitchen and bathroom placement."
    )

    inferred_rules = _summarize_inferred_rules(spec, resolution)
    planning_summary = []
    if room_program:
        planning_summary.append(
            "Program build: "
            + ", ".join(
                f"{room.get('name')} as {room.get('semantic_role', room.get('type', '')).replace('_', ' ')}"
                for room in room_program.get("rooms", [])[:4]
            )
            + "."
        )
    if zoning_plan:
        planning_summary.append(
            f"Zoning pattern: `{zoning_plan.get('layout_pattern', 'balanced')}` with `{zoning_plan.get('frontage_room', 'a public room')}` preferred at the entrance frontage."
        )
    if assumptions:
        planning_summary.append("Assumptions used: " + "; ".join(dict.fromkeys(assumptions)) + ".")
    return "\n\n".join(
        [
            intro,
            coverage,
            "## Quality Readout\n" + "\n".join(f"- {note}" for note in quality_notes),
            "\n".join(planning_summary) if planning_summary else "",
            inferred_rules,
            next_step,
        ]
    )


def _resolve_active_design_index(session: ConversationSession) -> Optional[int]:
    if session.selected_design_index is not None and 0 <= session.selected_design_index < len(session.designs):
        return session.selected_design_index
    if session.designs:
        return len(session.designs) - 1
    return None


def _summarize_applied_changes(changes: Optional[List[dict]]) -> str:
    changes = changes or []
    if not changes:
        return ""

    def _one(change: dict) -> str:
        change_type = str(change.get("type") or "").strip()
        if change_type == "move_room":
            room = change.get("room", "the room")
            direction = change.get("direction", "that direction")
            return f"moved {room} toward {direction}"
        if change_type == "resize_room":
            room = change.get("room", "the room")
            size_change = change.get("size_change", "adjusted")
            return f"made {room} {size_change}"
        if change_type == "swap_rooms":
            return f"swapped {change.get('room_a', 'one room')} with {change.get('room_b', 'another room')}"
        if change_type == "add_room":
            return f"added a {change.get('room_type', 'room')}"
        if change_type == "remove_room":
            return f"removed {change.get('room', 'a room')}"
        if change_type == "change_adjacency":
            return (
                f"updated the relationship between {change.get('room_a', 'one room')} and "
                f"{change.get('room_b', 'another room')}"
            )
        return change_type.replace("_", " ")

    items = [_one(change) for change in changes[:3]]
    if len(changes) > 3:
        items.append("applied the remaining requested adjustments")
    return "I updated the current layout request and " + ", ".join(items) + "."


def _message_requests_same_plot(message: str) -> bool:
    lowered = (message or "").strip().lower()
    return any(
        phrase in lowered
        for phrase in (
            "same plot",
            "same site",
            "same dimensions",
            "same size",
            "keep the same plot",
            "keep same plot",
        )
    )


def _explain_generation_error(error_str: str, spec: dict = None, resolution: dict = None) -> str:
    """
    Convert technical error messages into user-friendly explanations with suggestions.

    This function is designed to never fail - it will always return a helpful message.
    """
    try:
        spec = spec or {}
        resolution = resolution or {}

        rooms = spec.get("rooms", []) if isinstance(spec, dict) else []
        room_count = sum(r.get("count", 1) for r in rooms) if rooms else 0

        # Safely get boundary dimensions
        boundary = resolution.get("boundary_size", (12, 15)) if isinstance(resolution, dict) else (12, 15)
        site_boundary = resolution.get("site_boundary_size") if isinstance(resolution, dict) else None
        if not isinstance(boundary, (tuple, list)) or len(boundary) < 2:
            boundary = (12, 15)

        try:
            plot_area = float(boundary[0]) * float(boundary[1])
        except (TypeError, ValueError):
            plot_area = 180
        requested_area = resolution.get("total_area") if isinstance(resolution, dict) else None

        # Build room summary
        room_summary = ", ".join([f"{r.get('count', 1)} {r.get('type')}" for r in rooms]) if rooms else "the requested rooms"

        explanation = "I couldn't turn that request into a layout that passed the quality checks yet. Here's what happened:\n\n"

        error_lower = error_str.lower() if error_str else ""

        # Travel distance issue
        if "travel distance" in error_lower:
            explanation += "**Problem: Rooms are too far apart**\n"
            if room_count > 0:
                if requested_area:
                    area_basis = f"{requested_area:.0f} sq.m requested area"
                elif isinstance(site_boundary, (tuple, list)) and len(site_boundary) >= 2:
                    area_basis = (
                        f"{site_boundary[0]}m x {site_boundary[1]}m site with a {boundary[0]}m x {boundary[1]}m working footprint"
                    )
                else:
                    area_basis = f"{boundary[0]}m x {boundary[1]}m plot ({plot_area:.0f} sq.m)"
                explanation += f"With {room_count} rooms in a {area_basis}, "
            explanation += "the walking distance between important rooms (like bedroom to bathroom) would be too long.\n\n"

        # Adjacency satisfaction issue
        if "adjacency" in error_lower:
            explanation += "**Problem: Room placement conflicts**\n"
            explanation += "The rooms couldn't be arranged to satisfy the expected relationships "
            explanation += "(e.g., kitchen near living room, bathrooms near bedrooms).\n\n"

        if "room area allocation drift too high" in error_lower:
            explanation += "**Problem: The inferred plot is too tight for the requested room sizes**\n"
            explanation += "The packer could not keep all required rooms close to their target areas while still preserving usable circulation and door access.\n\n"

        if "circulation area is too high" in error_lower:
            explanation += "**Problem: Too much of the plan is being consumed by circulation**\n"
            explanation += "The generator found candidate room arrangements, but too much of the footprint was lost to movement space instead of usable rooms.\n\n"

        # Quality gate rejection
        if "quality gate" in error_lower or "rejected all variants" in error_lower:
            explanation += "**What this means:** The system tried multiple layout configurations, but none were stable enough to return as a usable home plan.\n\n"

        # Provide suggestions based on the issue
        explanation += "**Suggestions to fix this:**\n\n"

        # Check if plot might be too small
        min_area_per_room = 12  # rough estimate: 12 sq.m per room minimum
        if room_count > 0 and plot_area < room_count * min_area_per_room:
            min_dim = int((room_count * min_area_per_room) ** 0.5) + 2
            explanation += f"1. **Increase plot size** - Your {room_count} rooms need approximately {room_count * min_area_per_room} sq.m minimum. "
            explanation += f"Current working plot is only {plot_area:.0f} sq.m. Try setting dimensions to at least {min_dim}m x {min_dim}m in Settings.\n\n"
        else:
            if requested_area:
                explanation += f"1. **Keep the requested area but simplify the layout** - The system respected your requested area of {requested_area:.0f} sq.m, but the arrangement still failed the quality gate.\n\n"
            else:
                if "room area allocation drift too high" in error_lower or "circulation area is too high" in error_lower:
                    explanation += f"1. **Use a roomier working plot** - The current inferred boundary is {boundary[0]}m x {boundary[1]}m. A slightly larger envelope usually fixes this specific failure.\n\n"
                else:
                    explanation += f"1. **Try a larger plot** - Increase the plot dimensions in Settings (currently {boundary[0]}m x {boundary[1]}m).\n\n"

        # Suggest fewer rooms
        if room_count > 5:
            explanation += f"2. **Reduce room count** - You requested {room_count} rooms ({room_summary}). Try a simpler configuration like '2BHK' first.\n\n"
        else:
            explanation += "2. **Simplify requirements** - Try removing optional rooms or reducing the number of bathrooms.\n\n"

        explanation += "3. **Remove adjacency constraints** - If you specified rooms that must be near each other, try without those requirements first.\n\n"

        explanation += "**Would you like to try again with:**\n"
        explanation += "- A larger plot size?\n"
        explanation += "- Fewer rooms?\n"
        explanation += "- A standard configuration like '2BHK' or '3BHK'?\n\n"
        explanation += "Tell me which part you want to adjust, and I'll continue from the same conversation."

        return explanation

    except Exception as e:
        # Ultimate fallback - should never happen, but just in case
        print(f"[ERROR] _explain_generation_error failed: {e}")
        return f"""I wasn't able to generate a floor plan for your request.

**What happened:** {error_str}

**Suggestions:**
1. Try a larger plot size (click the Settings gear icon)
2. Request fewer rooms (e.g., try '2BHK' instead of '3BHK')
3. Remove specific room placement requirements

Would you like to try again with different settings?"""


@app.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest):
    try:
        result = await _run_backend(body)
        result["artifact_urls"] = _artifact_urls(result.get("artifact_paths"))
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/generate", response_model=ChatGenerateResponse)
async def chat_generate(body: ChatGenerateRequest):
    print(f"\n{'*'*100}")
    print(f"[CHAT_GENERATE] WARNING: Legacy endpoint /chat/generate called (no session)")
    print(f"[CHAT_GENERATE] Prompt: '{body.prompt}'")
    print(f"[CHAT_GENERATE] Boundary: w={body.boundary.width}, h={body.boundary.height}")
    print(f"{'*'*100}\n")

    resolution = {
        "boundary_size": (body.boundary.width, body.boundary.height),
        "area_unit": body.area_unit,
    }
    if body.boundary_polygon is not None:
        resolution["boundary_polygon"] = body.boundary_polygon
    if body.area is not None:
        resolution["total_area"] = body.area
    if body.entrance_point is not None:
        resolution["entrance_point"] = tuple(body.entrance_point)

    response = process_user_request(body.prompt, current_spec=None, resolution=resolution)

    print(f"[CHAT_GENERATE] backend_ready={response.get('backend_ready')}, missing={response.get('missing_fields')}")

    payload = {
        "assistant_text": response.get("assistant_text", ""),
        "backend_ready": bool(response.get("backend_ready")),
        "backend_target": response.get("backend_target"),
        "missing_fields": list(response.get("missing_fields", [])),
        "validation_errors": list(response.get("validation_errors", [])),
        "execution": None,
    }

    if not response.get("backend_ready"):
        return payload

    try:
        execution = await anyio.to_thread.run_sync(
            partial(
                execute_response,
                response,
                output_dir="outputs",
                output_prefix=body.output_prefix or f"chat_{uuid4().hex[:8]}",
                checkpoint_path=body.checkpoint_path
                or "learned/model/checkpoints/improved_v1.pt",
            )
        )
        execution["artifact_urls"] = _artifact_urls(execution.get("artifact_paths"))
        payload["execution"] = execution
        return payload
    except Exception as exc:
        payload["validation_errors"].append(str(exc))
        return payload


@app.get("/report")
async def get_report(path: str):
    if not path:
        raise HTTPException(status_code=400, detail="Missing path")
    target = Path(path).resolve()
    # Restrict to outputs directory to prevent path traversal
    allowed = outputs_dir.resolve()
    if not str(target).startswith(str(allowed)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    try:
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/ui-config")
async def ui_config():
    return {
        "title": "BlueprintGPT",
        "default_backend": "hybrid",
        "gemini_available": gemini_available(),
        "planner": _planner_runtime_status(),
        "features": {
            "multi_design": True,
            "corrections": True,
            "explanations": True,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Conversation API Models
# ═══════════════════════════════════════════════════════════════════════════════

class ConversationMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's message")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn conversations")
    boundary: Optional[Boundary] = None
    boundary_polygon: Optional[List[List[float]]] = None
    entrance_point: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    use_manual_boundary: bool = Field(False, description="Whether frontend settings should override inferred plot geometry")
    history: Optional[List[dict]] = None
    client_spec: Optional[dict] = None
    generate: bool = Field(True, description="Whether to generate designs if spec is complete")


class ConversationMessageResponse(BaseModel):
    session_id: str
    assistant_text: str
    state: str
    spec_complete: bool
    current_spec: dict
    designs: Optional[List[dict]] = None
    comparison: Optional[str] = None
    needs_info: Optional[List[str]] = None
    conversation_state: Optional[dict] = None
    clarification_request: Optional[str] = None
    assumptions_used: Optional[List[str]] = None
    program_summary: Optional[str] = None
    zoning_summary: Optional[str] = None
    latest_design_summary: Optional[dict] = None
    expert_diagnostics: Optional[dict] = None
    suggested_actions: Optional[List[str]] = None


class CorrectionRequest(BaseModel):
    session_id: str = Field(..., description="Session ID")
    design_index: int = Field(..., ge=0, description="Index of design to correct")
    correction: str = Field(..., min_length=1, description="Correction request in natural language")


class CorrectionResponse(BaseModel):
    success: bool
    assistant_text: str
    changes_applied: Optional[List[dict]] = None
    new_design: Optional[dict] = None
    needs_clarification: Optional[str] = None


class SessionExportResponse(BaseModel):
    session_id: str
    data: str


# ═══════════════════════════════════════════════════════════════════════════════
#  Conversation Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/conversation/message", response_model=ConversationMessageResponse)
async def conversation_message(body: ConversationMessageRequest):
    """
    Handle a conversation message with multi-turn support and intelligent intent routing.

    This endpoint:
    1. Classifies user intent (design, question, correction, conversation)
    2. Routes to appropriate handler based on intent
    3. Maintains conversation state across turns
    4. Generates designs when spec is complete
    5. Provides AI explanations for each design
    """
    # Log user interaction with centralized logging
    ProcessingLogger.log_user_interaction(
        body.message,
        body.session_id,
        body.boundary.model_dump() if body.boundary else None,
        body.entrance_point,
        body.generate
    )

    # Get or create session
    session = conversation_manager.get_or_create_session(body.session_id)
    session_was_new = session.session_id != body.session_id if body.session_id else True
    ProcessingLogger.logger.debug(
        f"Session lookup: requested={body.session_id!r}, got={session.session_id!r}, "
        f"was_new={session_was_new}, existing_rooms={len(session.current_spec.get('rooms', []))}"
    )

    if DetailedLogger.enabled():
        DetailedLogger.log_detailed_state("SESSION", {
            "state": session.state,
            "current_rooms": session.current_spec.get('rooms', [])
        })

    if (session_was_new or not session.current_spec.get("rooms")) and body.client_spec:
        session.current_spec = conversation_orchestrator.enrich_spec(
            normalize_current_spec(body.client_spec),
            session.resolution,
            body.message,
        )
        session.set_planning_context(
            semantic_spec=session.current_spec.get("semantic_spec"),
            room_program=session.current_spec.get("room_program"),
            zoning_plan=session.current_spec.get("zoning_plan"),
        )
        ProcessingLogger.logger.debug(
            f"Hydrated session spec from client with {len(session.current_spec.get('rooms', []))} room types"
        )
    elif session.current_spec.get("rooms") and not session.semantic_spec:
        session.current_spec = conversation_orchestrator.enrich_spec(
            normalize_current_spec(session.current_spec),
            session.resolution,
            body.message,
        )
        session.set_planning_context(
            semantic_spec=session.current_spec.get("semantic_spec"),
            room_program=session.current_spec.get("room_program"),
            zoning_plan=session.current_spec.get("zoning_plan"),
        )

    if body.history and session_was_new and len(session.messages) == 0:
        for item in body.history[-20:]:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and content:
                session.add_message(role, content)

    # Add user message to history
    session.add_message("user", body.message)

    # Set initial resolution from frontend if provided
    resolution = None
    if body.use_manual_boundary and body.boundary:
        resolution = {
            "boundary_size": (body.boundary.width, body.boundary.height),
            "area_unit": APIDefaults.DEFAULT_AREA_UNIT,
            "boundary_source": "manual",
            "boundary_role": "footprint",
        }
        if body.boundary_polygon:
            resolution["boundary_polygon"] = body.boundary_polygon
        if body.entrance_point:
            resolution["entrance_point"] = tuple(body.entrance_point)
        session.set_resolution(resolution)
        session.current_spec = conversation_orchestrator.enrich_spec(session.current_spec, resolution, body.message)
        session.set_planning_context(
            semantic_spec=session.current_spec.get("semantic_spec"),
            room_program=session.current_spec.get("room_program"),
            zoning_plan=session.current_spec.get("zoning_plan"),
        )

    # Build context for intent classification
    latest_design = session.designs[-1].to_dict() if session.designs else None
    context = {
        "state": session.state,
        "num_designs": len(session.designs),
        "spec": session.current_spec,
        "semantic_spec": session.semantic_spec,
        "room_program": session.room_program,
        "zoning_plan": session.zoning_plan,
        "selected_design": session.selected_design_index,
        "latest_design": latest_design,
        "latest_generation_outcome": session.latest_generation_outcome,
        "current_rooms": (
            [
                {"name": f"{room.get('type')}_{idx + 1}", "type": room.get("type")}
                for room in (latest_design.get("rooms", []) or [])
                for idx in range(int(room.get("count", 1) or 1))
            ]
            if latest_design and latest_design.get("rooms")
            else [{"type": r.get("type"), "name": r.get("name")} for r in session.current_spec.get("rooms", [])]
        ),
    }

    # Use the new intelligent message processing with intent classification
    nl_result = process_nl_message(body.message, context, session.get_history(limit=10))

    intent = nl_result.get("intent", IntentTypes.CONVERSATION)
    confidence = nl_result.get('intent_confidence', 0)
    should_generate = nl_result.get('should_generate', False)

    # Log intent classification
    ProcessingLogger.log_intent_classification(
        intent, confidence,
        nl_result.get('intent_reason', ''),
        nl_result.get('design_keywords'),
        nl_result.get('question_type')
    )

    if intent == IntentTypes.DESIGN:
        extracted_spec = nl_result.get("spec", {})
        cli_preview = {}
        if nl_result.get("should_generate"):
            from nl_interface.service import _extract_cli_args
            cli_preview = _extract_cli_args(body.message)

        if (
            extracted_spec
            and extracted_spec.get("rooms")
            and not body.use_manual_boundary
            and not cli_preview.get("boundary_size")
            and not cli_preview.get("total_area")
            and not _message_requests_same_plot(body.message)
        ):
            session.set_resolution(_clear_prompt_scoped_resolution(session.resolution))

        if extracted_spec:
            enriched_extracted_spec = conversation_orchestrator.enrich_spec(
                normalize_current_spec(extracted_spec),
                session.resolution,
                body.message,
            )
            nl_result["spec"] = enriched_extracted_spec
            extracted_spec = enriched_extracted_spec
        rooms = extracted_spec.get("rooms", [])
        total_rooms = sum(r.get('count', 1) for r in rooms)

        ProcessingLogger.log_spec_extraction(
            rooms, total_rooms,
            extracted_spec.get('plot_type'),
            extracted_spec.get('entrance_side'),
            extracted_spec.get('adjacency')
        )

    # Check if natural language contained dimension override
    if intent == INTENT_DESIGN and nl_result.get("should_generate"):
        # Extract any CLI args that might override frontend settings
        from nl_interface.service import _extract_cli_args
        cli_overrides = cli_preview if 'cli_preview' in locals() else _extract_cli_args(body.message)

        # Log dimension processing for debugging
        frontend_dims = resolution.get('boundary_size') if resolution else None
        cli_dims = cli_overrides.get("boundary_size")
        ProcessingLogger.logger.debug(f"Frontend dims: {frontend_dims}, CLI override: {cli_dims}")

        if cli_overrides.get("boundary_size"):
            ProcessingLogger.logger.info(f"Applied dimension override: {cli_overrides['boundary_size']} (from natural language)")
            if resolution:
                resolution["boundary_size"] = tuple(cli_overrides["boundary_size"])
            else:
                resolution = {
                    "boundary_size": tuple(cli_overrides["boundary_size"]),
                    "area_unit": "sq.m",
                }
            resolution["boundary_source"] = "nl_prompt"
            resolution["boundary_role"] = cli_overrides.get("boundary_role", "footprint")
            if resolution.get("boundary_role") == "site":
                resolution["site_boundary_size"] = tuple(cli_overrides["boundary_size"])
            if body.entrance_point:
                resolution["entrance_point"] = tuple(body.entrance_point)
            session.set_resolution(resolution)
        if cli_overrides.get("total_area"):
            if resolution:
                resolution["total_area"] = float(cli_overrides["total_area"])
                resolution["area_unit"] = cli_overrides.get("area_unit", "sq.m")
            else:
                resolution = {
                    "total_area": float(cli_overrides["total_area"]),
                    "area_unit": cli_overrides.get("area_unit", "sq.m"),
                }
            resolution["total_area_source"] = "nl_prompt"
            if body.entrance_point:
                resolution["entrance_point"] = tuple(body.entrance_point)
            session.set_resolution(resolution)
        else:
            ProcessingLogger.logger.debug(f"Using frontend dimensions: {resolution.get('boundary_size') if resolution else 'Default will be applied'}")

    # Handle based on intent
    # Special case: User says "generate" but we already have a spec from previous messages
    msg_lower = body.message.lower().strip()
    is_generate_command = msg_lower in ["generate", "go", "create", "build", "make it", "start", "proceed", "yes", "ok", "okay"]
    has_existing_spec = bool(session.current_spec.get("rooms"))

    # If user just wants to generate and we have an existing spec, use it
    if is_generate_command and has_existing_spec and body.generate:
        ProcessingLogger.logger.info(f"Generate command detected with existing spec - using session spec")
        intent = INTENT_DESIGN
        should_generate = True
        nl_result["should_generate"] = True
        nl_result["spec"] = session.current_spec  # Use existing spec

    # Always store extracted spec in session for future use (even if not generating yet)
    safe_nl_result = nl_result if isinstance(nl_result, dict) else {}
    safe_spec = safe_nl_result.get('spec') if safe_nl_result else {}
    safe_rooms = safe_spec.get('rooms') if isinstance(safe_spec, dict) else []
    ProcessingLogger.logger.debug(
        f"Spec storage check: intent={intent!r}, INTENT_DESIGN={INTENT_DESIGN!r}, "
        f"has_spec={bool(safe_spec)}, spec_rooms={len(safe_rooms) if isinstance(safe_rooms, list) else 0}"
    )
    if intent == INTENT_DESIGN and nl_result and nl_result.get("spec"):
        extracted_spec = nl_result.get("spec", {})
        if extracted_spec.get("rooms"):  # Only store if we have rooms
            ProcessingLogger.logger.info(f"Storing extracted spec in session: {len(extracted_spec.get('rooms', []))} rooms")
            session.update_spec(extracted_spec)
            session.current_spec = conversation_orchestrator.enrich_spec(session.current_spec, session.resolution, body.message)
            session.set_planning_context(
                semantic_spec=session.current_spec.get("semantic_spec"),
                room_program=session.current_spec.get("room_program"),
                zoning_plan=session.current_spec.get("zoning_plan"),
            )

            # Auto-trigger generation for complete design requests
            ProcessingLogger.logger.info("Auto-triggering generation - DESIGN intent with rooms detected")
            nl_result["should_generate"] = True

            default_resolution = _derive_resolution_from_spec(
                extracted_spec,
                session.resolution,
                body.entrance_point,
            )
            width, height = default_resolution["boundary_size"]
            room_count = sum(int(room.get("count", 1)) for room in extracted_spec.get("rooms", []))
            ProcessingLogger.logger.info(f"Dynamic boundary: {width}m x {height}m based on {room_count} rooms")
            session.set_resolution(default_resolution)
            session.current_spec = conversation_orchestrator.enrich_spec(session.current_spec, session.resolution, body.message)
            session.set_planning_context(
                semantic_spec=session.current_spec.get("semantic_spec"),
                room_program=session.current_spec.get("room_program"),
                zoning_plan=session.current_spec.get("zoning_plan"),
            )
        else:
            ProcessingLogger.logger.warning(f"Spec has no rooms - not storing. Spec keys: {list(extracted_spec.keys())}")
    elif intent != INTENT_DESIGN:
        ProcessingLogger.logger.debug(f"Not storing spec: intent {intent!r} != INTENT_DESIGN {INTENT_DESIGN!r}")
    elif not nl_result.get("spec"):
        ProcessingLogger.logger.debug(f"Not storing spec: nl_result has no spec key or spec is empty")

    if intent == INTENT_DESIGN and nl_result.get("should_generate") and body.generate:
        ProcessingLogger.logger.info("Starting design generation pipeline")

        # Extract and update spec
        extracted = nl_result.get("spec", {})
        ProcessingLogger.logger.debug(f"Extracted spec: {extracted}")
        session.update_spec(extracted)
        session.current_spec = conversation_orchestrator.enrich_spec(session.current_spec, session.resolution, body.message)
        session.set_planning_context(
            semantic_spec=session.current_spec.get("semantic_spec"),
            room_program=session.current_spec.get("room_program"),
            zoning_plan=session.current_spec.get("zoning_plan"),
        )

        ProcessingLogger.logger.debug(f"Current session spec: {session.current_spec}")
        ProcessingLogger.logger.debug(f"Current resolution: {session.resolution}")

        # Auto-dimension selection if no dimensions specified or if natural-language area should drive geometry
        current_resolution = session.resolution or {}
        boundary_size = current_resolution.get("boundary_size")

        if current_resolution.get("total_area") or (not boundary_size) or any(float(v) <= 0 for v in boundary_size):
            rooms = session.current_spec.get("rooms", [])
            if rooms:
                ProcessingLogger.logger.info("Calculating working boundary from inferred spec and requested area")
                current_resolution = _derive_resolution_from_spec(session.current_spec, current_resolution, body.entrance_point)
                session.set_resolution(current_resolution)
                session.current_spec = conversation_orchestrator.enrich_spec(session.current_spec, session.resolution, body.message)
                session.set_planning_context(
                    semantic_spec=session.current_spec.get("semantic_spec"),
                    room_program=session.current_spec.get("room_program"),
                    zoning_plan=session.current_spec.get("zoning_plan"),
                )
                ProcessingLogger.logger.debug(f"Updated resolution: {current_resolution}")
            else:
                ProcessingLogger.logger.warning(f"No rooms specified - using default dimensions {boundary_size}")

        # Check if spec is complete via process_user_request
        nl_response = process_user_request(
            body.message,
            current_spec=session.current_spec,
            resolution=session.resolution,
        )
        effective_generation_spec = nl_response.get("current_spec") or session.current_spec

        spec_complete = nl_response.get("backend_ready", False)
        missing = nl_response.get("missing_fields", [])
        backend_spec = nl_response.get("backend_spec")

        ProcessingLogger.log_generation_pipeline(
            spec_complete=spec_complete,
            missing_fields=missing,
            backend_target=nl_response.get('backend_target'),
            backend_ready=bool(backend_spec)
        )

        # Force spec_complete=True for auto-generation since we already validated rooms exist
        if not spec_complete and not missing:
            ProcessingLogger.logger.info("Forcing spec_complete=True for auto-generation (no missing fields)")
            spec_complete = True

            # Also force backend_ready in the nl_response so execute_response works
            nl_response["backend_ready"] = True

            # Build backend_spec if missing
            if not nl_response.get("backend_spec"):
                from nl_interface.adapter import build_backend_spec
                # Use dynamic dimensions if no resolution exists
                if not session.resolution:
                    from nl_interface.auto_dimension_selector import recommend_dimensions
                    rooms_fb = effective_generation_spec.get("rooms", [])
                    if rooms_fb:
                        w_fb, h_fb = recommend_dimensions(rooms_fb, building_type="residential")
                    else:
                        w_fb, h_fb = 10.0, 8.0
                    resolution = {
                        "boundary_size": (w_fb, h_fb),
                        "boundary_polygon": [[0, 0], [w_fb, 0], [w_fb, h_fb], [0, h_fb]],
                        "entrance_point": [w_fb / 2, 0],
                        "area_unit": "sq.m"
                    }
                else:
                    resolution = session.resolution
                backend_spec, _ = build_backend_spec(effective_generation_spec, resolution)
                nl_response["backend_spec"] = backend_spec
                ProcessingLogger.logger.info(f"Built backend_spec with {len(backend_spec.get('rooms', []))} rooms")

        if spec_complete:
            response_text = nl_result.get("response", "Generating your floor plan designs...")
            session.add_message("assistant", response_text)

            # Generate designs
            try:
                session.clear_designs()
                output_prefix = f"conv_{session.session_id[:8]}_{uuid4().hex[:4]}"

                execution = await anyio.to_thread.run_sync(
                    partial(
                        execute_response,
                        nl_response,
                        output_dir="outputs",
                        output_prefix=output_prefix,
                    )
                )

                # Add design with explanation
                design_data = dict(execution)
                design_data["artifact_urls"] = _artifact_urls(design_data.get("artifact_paths"))
                session.add_design(design_data, rank=1)

                # Generate explanation
                explained_designs = explain_ranked_designs([design_data], session.current_spec)
                comparison = generate_comparison_explanation(explained_designs)

                # Log successful generation
                ProcessingLogger.log_generation_result(success=True, design_count=len(explained_designs))

                # Add assistant response
                final_response = _build_design_conversation_reply(
                    design_data=design_data,
                    spec=session.current_spec,
                    resolution=session.resolution or {},
                    design_count=len(explained_designs),
                )
                generation_outcome = GenerationOutcome(
                    stage="generation",
                    success=True,
                    engine=design_data.get("winning_source") or design_data.get("backend_target"),
                    report_status=design_data.get("report_status"),
                    summary="Generated a residential layout candidate that passed the current quality gate.",
                    assumptions_used=(session.semantic_spec or {}).get("assumptions_used", []),
                    metrics=design_data.get("metrics", {}),
                    artifact_urls=design_data.get("artifact_urls", {}),
                )
                reply_payload = conversation_orchestrator.build_reply_payload(
                    session=session,
                    semantic_spec=session.semantic_spec,
                    room_program=session.room_program,
                    zoning_plan=session.zoning_plan,
                    generation_outcome=generation_outcome,
                    suggested_actions=[
                        "Refine this layout by changing kitchen, bedroom, or bathroom placement.",
                        "Adjust plot size or entrance side and regenerate.",
                        "Open expert mode to inspect planning and validation details.",
                    ],
                )
                session.set_planning_context(
                    reply_payload=reply_payload,
                    generation_outcome=generation_outcome.to_dict(),
                )
                session.add_message("assistant", final_response)

                return ConversationMessageResponse(
                    session_id=session.session_id,
                    assistant_text=final_response,
                    state=session.state,
                    spec_complete=True,
                    current_spec=session.current_spec,
                    designs=[d for d in explained_designs],
                    comparison=comparison,
                    needs_info=None,
                    conversation_state=reply_payload.get("conversation_state"),
                    clarification_request=reply_payload.get("clarification_request"),
                    assumptions_used=reply_payload.get("assumptions_used"),
                    program_summary=reply_payload.get("program_summary"),
                    zoning_summary=reply_payload.get("zoning_summary"),
                    latest_design_summary=reply_payload.get("latest_design_summary"),
                    expert_diagnostics=reply_payload.get("expert_diagnostics"),
                    suggested_actions=reply_payload.get("suggested_actions"),
                )

            except Exception as exc:
                # Parse error and provide user-friendly explanation
                error_str = str(exc)
                ProcessingLogger.log_generation_result(
                    success=False,
                    error_msg=error_str
                )
                ProcessingLogger.logger.debug(f"Generation error spec: {session.current_spec}")
                ProcessingLogger.logger.debug(f"Generation error resolution: {session.resolution}")

                error_explanation = _explain_generation_error(error_str, session.current_spec, session.resolution)
                error_explanation += "\n\n" + _summarize_inferred_rules(extracted, session.resolution)
                ProcessingLogger.logger.debug(f"User-friendly explanation generated ({len(error_explanation)} chars)")

                generation_outcome = GenerationOutcome(
                    stage="generation",
                    success=False,
                    engine=nl_response.get("backend_target"),
                    report_status="NON_COMPLIANT",
                    failure_reason=error_str,
                    summary="Generation failed after layout synthesis or quality verification.",
                    assumptions_used=(session.semantic_spec or {}).get("assumptions_used", []),
                )
                reply_payload = conversation_orchestrator.build_reply_payload(
                    session=session,
                    semantic_spec=session.semantic_spec,
                    room_program=session.room_program,
                    zoning_plan=session.zoning_plan,
                    generation_outcome=generation_outcome,
                    suggested_actions=[
                        "Increase the working plot area or provide a custom plot size.",
                        "Reduce optional constraints and regenerate.",
                        "Keep the same program but ask me to rebalance privacy, frontage, or circulation.",
                    ],
                )
                session.set_planning_context(
                    reply_payload=reply_payload,
                    generation_outcome=generation_outcome.to_dict(),
                )
                session.add_message("assistant", error_explanation)
                return ConversationMessageResponse(
                    session_id=session.session_id,
                    assistant_text=error_explanation,
                    state=session.state,
                    spec_complete=False,
                    current_spec=session.current_spec,
                    designs=None,
                    comparison=None,
                    needs_info=["generation_error"],
                    conversation_state=reply_payload.get("conversation_state"),
                    clarification_request=reply_payload.get("clarification_request"),
                    assumptions_used=reply_payload.get("assumptions_used"),
                    program_summary=reply_payload.get("program_summary"),
                    zoning_summary=reply_payload.get("zoning_summary"),
                    latest_design_summary=reply_payload.get("latest_design_summary"),
                    expert_diagnostics=reply_payload.get("expert_diagnostics"),
                    suggested_actions=reply_payload.get("suggested_actions"),
                )
        else:
            # Need more info for design
            response_text = nl_result.get("response", "")
            if missing:
                response_text += f"\n\nI still need: {', '.join(missing)}"
            clarification_request = conversation_orchestrator.build_clarification_request(
                session.semantic_spec,
                session.room_program,
            )
            reply_payload = conversation_orchestrator.build_reply_payload(
                session=session,
                semantic_spec=session.semantic_spec,
                room_program=session.room_program,
                zoning_plan=session.zoning_plan,
                clarification_request=clarification_request,
                suggested_actions=["Answer the missing planning question so I can continue to generation."],
            )
            session.set_planning_context(reply_payload=reply_payload)
            session.add_message("assistant", response_text)

            return ConversationMessageResponse(
                session_id=session.session_id,
                assistant_text=response_text,
                state=session.state,
                spec_complete=False,
                current_spec=session.current_spec,
                designs=None,
                comparison=None,
                needs_info=missing or None,
                conversation_state=reply_payload.get("conversation_state"),
                clarification_request=reply_payload.get("clarification_request"),
                assumptions_used=reply_payload.get("assumptions_used"),
                program_summary=reply_payload.get("program_summary"),
                zoning_summary=reply_payload.get("zoning_summary"),
                latest_design_summary=reply_payload.get("latest_design_summary"),
                expert_diagnostics=reply_payload.get("expert_diagnostics"),
                suggested_actions=reply_payload.get("suggested_actions"),
            )

    elif intent == INTENT_CORRECTION:
        target_design_index = _resolve_active_design_index(session)
        if target_design_index is None:
            response_text = (
                "I can modify the previous layout through this chat, but I need at least one generated design in the session first."
            )
            reply_payload = conversation_orchestrator.build_reply_payload(
                session=session,
                semantic_spec=session.semantic_spec,
                room_program=session.room_program,
                zoning_plan=session.zoning_plan,
                clarification_request="Generate a layout first, then describe the change you want.",
                suggested_actions=[
                    "Generate the first layout for the current room program.",
                    "Then tell me exactly what to move, resize, add, or remove.",
                ],
            )
            session.set_planning_context(reply_payload=reply_payload)
            session.add_message("assistant", response_text)
            return ConversationMessageResponse(
                session_id=session.session_id,
                assistant_text=response_text,
                state=session.state,
                spec_complete=False,
                current_spec=session.current_spec,
                designs=None,
                comparison=None,
                needs_info=["design_generation"],
                conversation_state=reply_payload.get("conversation_state"),
                clarification_request=reply_payload.get("clarification_request"),
                assumptions_used=reply_payload.get("assumptions_used"),
                program_summary=reply_payload.get("program_summary"),
                zoning_summary=reply_payload.get("zoning_summary"),
                latest_design_summary=reply_payload.get("latest_design_summary"),
                expert_diagnostics=reply_payload.get("expert_diagnostics"),
                suggested_actions=reply_payload.get("suggested_actions"),
            )

        result, error = handle_correction_request(body.message, target_design_index, session)
        if error:
            clarification_request = (result.get("parsed") or {}).get("clarification_needed")
            response_text = error
            if not clarification_request:
                response_text = conversation_orchestrator.compose_contextual_reply(
                    user_message=body.message,
                    session=session,
                    fallback_text=error,
                    room_program=session.room_program,
                    zoning_plan=session.zoning_plan,
                )
            reply_payload = conversation_orchestrator.build_reply_payload(
                session=session,
                semantic_spec=session.semantic_spec,
                room_program=session.room_program,
                zoning_plan=session.zoning_plan,
                clarification_request=clarification_request,
                suggested_actions=[
                    "Name the room you want to move, resize, add, or remove.",
                    "If needed, refer to the latest generated layout and I will modify that version.",
                ],
            )
            session.set_planning_context(reply_payload=reply_payload)
            session.add_message("assistant", response_text)
            return ConversationMessageResponse(
                session_id=session.session_id,
                assistant_text=response_text,
                state=session.state,
                spec_complete=False,
                current_spec=session.current_spec,
                designs=None,
                comparison=None,
                needs_info=["correction_clarification"],
                conversation_state=reply_payload.get("conversation_state"),
                clarification_request=reply_payload.get("clarification_request"),
                assumptions_used=reply_payload.get("assumptions_used"),
                program_summary=reply_payload.get("program_summary"),
                zoning_summary=reply_payload.get("zoning_summary"),
                latest_design_summary=reply_payload.get("latest_design_summary"),
                expert_diagnostics=reply_payload.get("expert_diagnostics"),
                suggested_actions=reply_payload.get("suggested_actions"),
            )

        session.current_spec = conversation_orchestrator.enrich_spec(
            normalize_current_spec(result.get("modified_spec", session.current_spec)),
            session.resolution,
            None,
        )
        session.set_planning_context(
            semantic_spec=session.current_spec.get("semantic_spec"),
            room_program=session.current_spec.get("room_program"),
            zoning_plan=session.current_spec.get("zoning_plan"),
        )

        try:
            output_prefix = f"conv_{session.session_id[:8]}_{uuid4().hex[:4]}_edit"
            nl_response = process_user_request(
                "",
                current_spec=session.current_spec,
                resolution=session.resolution,
            )

            spec_complete = nl_response.get("backend_ready", False)
            missing = nl_response.get("missing_fields", [])
            if not spec_complete and not missing:
                nl_response["backend_ready"] = True
                if not nl_response.get("backend_spec"):
                    from nl_interface.adapter import build_backend_spec

                    correction_resolution = session.resolution or _derive_resolution_from_spec(
                        session.current_spec,
                        session.resolution,
                        body.entrance_point,
                    )
                    session.set_resolution(correction_resolution)
                    session.current_spec = conversation_orchestrator.enrich_spec(
                        session.current_spec,
                        session.resolution,
                        None,
                    )
                    session.set_planning_context(
                        semantic_spec=session.current_spec.get("semantic_spec"),
                        room_program=session.current_spec.get("room_program"),
                        zoning_plan=session.current_spec.get("zoning_plan"),
                    )
                    backend_spec, _ = build_backend_spec(session.current_spec, correction_resolution)
                    nl_response["backend_spec"] = backend_spec
            elif missing:
                raise ValueError(
                    "I understood the change, but I still need "
                    + ", ".join(missing)
                    + " before I can regenerate the layout."
                )

            execution = await anyio.to_thread.run_sync(
                partial(
                    execute_response,
                    nl_response,
                    output_dir="outputs",
                    output_prefix=output_prefix,
                )
            )

            design_data = dict(execution)
            design_data["artifact_urls"] = _artifact_urls(design_data.get("artifact_paths"))
            new_rank = len(session.designs) + 1
            session.add_design(design_data, rank=new_rank)

            explained_designs = explain_ranked_designs([design_data], session.current_spec)
            comparison = generate_comparison_explanation(explained_designs)

            response_parts = []
            change_summary = _summarize_applied_changes(result.get("changes"))
            if change_summary:
                response_parts.append(change_summary)
            response_parts.append(
                _build_design_conversation_reply(
                    design_data=design_data,
                    spec=session.current_spec,
                    resolution=session.resolution or {},
                    design_count=len(explained_designs),
                )
            )
            final_response = "\n\n".join(part for part in response_parts if part)

            generation_outcome = GenerationOutcome(
                stage="correction_generation",
                success=True,
                engine=design_data.get("winning_source") or design_data.get("backend_target"),
                report_status=design_data.get("report_status"),
                summary="Applied the requested design changes and regenerated the layout.",
                assumptions_used=(session.semantic_spec or {}).get("assumptions_used", []),
                metrics=design_data.get("metrics", {}),
                artifact_urls=design_data.get("artifact_urls", {}),
            )
            reply_payload = conversation_orchestrator.build_reply_payload(
                session=session,
                semantic_spec=session.semantic_spec,
                room_program=session.room_program,
                zoning_plan=session.zoning_plan,
                generation_outcome=generation_outcome,
                suggested_actions=[
                    "Keep refining this updated layout with another move, resize, add, or remove request.",
                    "Ask why a room ended up in a certain zone and I will explain the planning logic.",
                    "Open expert mode if you want the planning and validation details.",
                ],
            )
            session.set_planning_context(
                reply_payload=reply_payload,
                generation_outcome=generation_outcome.to_dict(),
            )
            session.add_message("assistant", final_response)

            return ConversationMessageResponse(
                session_id=session.session_id,
                assistant_text=final_response,
                state=session.state,
                spec_complete=True,
                current_spec=session.current_spec,
                designs=explained_designs,
                comparison=comparison,
                needs_info=None,
                conversation_state=reply_payload.get("conversation_state"),
                clarification_request=reply_payload.get("clarification_request"),
                assumptions_used=reply_payload.get("assumptions_used"),
                program_summary=reply_payload.get("program_summary"),
                zoning_summary=reply_payload.get("zoning_summary"),
                latest_design_summary=reply_payload.get("latest_design_summary"),
                expert_diagnostics=reply_payload.get("expert_diagnostics"),
                suggested_actions=reply_payload.get("suggested_actions"),
            )
        except Exception as exc:
            ProcessingLogger.log_generation_result(success=False, error_msg=str(exc))
            response_parts = []
            change_summary = _summarize_applied_changes(result.get("changes"))
            if change_summary:
                response_parts.append(change_summary)
            response_parts.append(_explain_generation_error(str(exc), session.current_spec, session.resolution))
            response_parts.append(_summarize_inferred_rules(session.current_spec, session.resolution))
            response_text = "\n\n".join(part for part in response_parts if part)

            generation_outcome = GenerationOutcome(
                stage="correction_generation",
                success=False,
                engine=nl_response.get("backend_target") if isinstance(nl_response, dict) else None,
                report_status="NON_COMPLIANT",
                failure_reason=str(exc),
                summary="The requested change was understood, but the regenerated layout did not pass the current quality gate.",
                assumptions_used=(session.semantic_spec or {}).get("assumptions_used", []),
            )
            reply_payload = conversation_orchestrator.build_reply_payload(
                session=session,
                semantic_spec=session.semantic_spec,
                room_program=session.room_program,
                zoning_plan=session.zoning_plan,
                generation_outcome=generation_outcome,
                suggested_actions=[
                    "Try a smaller move or resize request.",
                    "Allow a slightly larger plot or different entrance-side emphasis.",
                    "Ask me to explain which part of the updated plan caused the failure.",
                ],
            )
            session.set_planning_context(
                reply_payload=reply_payload,
                generation_outcome=generation_outcome.to_dict(),
            )
            session.add_message("assistant", response_text)
            return ConversationMessageResponse(
                session_id=session.session_id,
                assistant_text=response_text,
                state=session.state,
                spec_complete=False,
                current_spec=session.current_spec,
                designs=None,
                comparison=None,
                needs_info=["generation_error"],
                conversation_state=reply_payload.get("conversation_state"),
                clarification_request=reply_payload.get("clarification_request"),
                assumptions_used=reply_payload.get("assumptions_used"),
                program_summary=reply_payload.get("program_summary"),
                zoning_summary=reply_payload.get("zoning_summary"),
                latest_design_summary=reply_payload.get("latest_design_summary"),
                expert_diagnostics=reply_payload.get("expert_diagnostics"),
                suggested_actions=reply_payload.get("suggested_actions"),
            )

    else:
        # INTENT_QUESTION or INTENT_CONVERSATION - use dynamic response
        response_text = nl_result.get("response", "")
        if not response_text:
            # Fallback to the configured chat adapter if process_message did not generate response
            response_text = gemini_chat(body.message, context, session.get_history(limit=10))

        response_text = conversation_orchestrator.compose_contextual_reply(
            user_message=body.message,
            session=session,
            fallback_text=response_text,
            room_program=session.room_program,
            zoning_plan=session.zoning_plan,
        )
        reply_payload = conversation_orchestrator.build_reply_payload(
            session=session,
            semantic_spec=session.semantic_spec,
            room_program=session.room_program,
            zoning_plan=session.zoning_plan,
            suggested_actions=[
                "Ask for a concrete layout change if you want me to regenerate.",
                "Ask why a layout passed or failed and I will explain the planning logic.",
            ],
        )
        session.set_planning_context(reply_payload=reply_payload)

        session.add_message("assistant", response_text)

        return ConversationMessageResponse(
            session_id=session.session_id,
            assistant_text=response_text,
            state=session.state,
            spec_complete=False,
            current_spec=session.current_spec,
            designs=None,
            comparison=None,
            needs_info=None,
            conversation_state=reply_payload.get("conversation_state"),
            clarification_request=reply_payload.get("clarification_request"),
            assumptions_used=reply_payload.get("assumptions_used"),
            program_summary=reply_payload.get("program_summary"),
            zoning_summary=reply_payload.get("zoning_summary"),
            latest_design_summary=reply_payload.get("latest_design_summary"),
            expert_diagnostics=reply_payload.get("expert_diagnostics"),
            suggested_actions=reply_payload.get("suggested_actions"),
        )


@app.post("/conversation/correct", response_model=CorrectionResponse)
async def conversation_correct(body: CorrectionRequest):
    """
    Handle a correction request for a specific design.

    Parses the user's correction in natural language and regenerates
    the design with the requested changes.
    """
    session = conversation_manager.get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if body.design_index >= len(session.designs):
        if not session.designs:
            raise HTTPException(status_code=400, detail="No designs have been generated yet.")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid design index. Available: 0-{len(session.designs) - 1}"
        )

    # Add user message
    session.add_message("user", f"[Correction for design #{body.design_index + 1}] {body.correction}")

    # Handle the correction
    result, error = handle_correction_request(body.correction, body.design_index, session)

    if error:
        session.add_message("assistant", error)
        return CorrectionResponse(
            success=False,
            assistant_text=error,
            needs_clarification=result.get("parsed", {}).get("clarification_needed"),
        )

    if result.get("needs_regeneration"):
        # Update spec and regenerate
        session.current_spec = result.get("modified_spec", session.current_spec)

        try:
            output_prefix = f"corr_{session.session_id[:8]}_{uuid4().hex[:4]}"

            nl_response = process_user_request(
                "",  # Empty prompt, using existing spec
                current_spec=session.current_spec,
                resolution=session.resolution,
            )

            if nl_response.get("backend_ready"):
                execution = await anyio.to_thread.run_sync(
                    partial(
                        execute_response,
                        nl_response,
                        output_dir="outputs",
                        output_prefix=output_prefix,
                    )
                )

                design_data = dict(execution)
                design_data["artifact_urls"] = _artifact_urls(design_data.get("artifact_paths"))

                # Add as new design
                new_design = session.add_design(design_data, rank=len(session.designs))

                # Generate explanation
                explained = explain_ranked_designs([design_data], session.current_spec)
                explanation = explained[0].get("explanation", "") if explained else ""

                response_text = f"I've applied your changes and regenerated the design.\n\n{explanation}"
                session.add_message("assistant", response_text)

                return CorrectionResponse(
                    success=True,
                    assistant_text=response_text,
                    changes_applied=result.get("changes", []),
                    new_design=design_data,
                )

        except Exception as exc:
            error_text = f"Error applying correction: {str(exc)}"
            session.add_message("assistant", error_text)
            return CorrectionResponse(
                success=False,
                assistant_text=error_text,
            )

    return CorrectionResponse(
        success=False,
        assistant_text="Could not process the correction. Please try rephrasing.",
    )


@app.get("/conversation/session/{session_id}")
async def get_session(session_id: str):
    """Get the current state of a conversation session."""
    session = conversation_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "state": session.state,
        "current_spec": session.current_spec,
        "messages": session.get_history(),
        "designs": [d.to_dict() for d in session.designs],
        "selected_design": session.selected_design_index,
    }


@app.post("/conversation/session/new")
async def create_session():
    """Create a new conversation session."""
    session = conversation_manager.create_session()
    print(f"[SESSION] + New session created: {session.session_id}")
    return {"session_id": session.session_id, "state": session.state}


@app.delete("/conversation/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session."""
    if conversation_manager.delete_session(session_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/conversation/session/{session_id}/export", response_model=SessionExportResponse)
async def export_session(session_id: str):
    """Export a conversation session as JSON."""
    data = conversation_manager.export_session(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionExportResponse(session_id=session_id, data=data)


@app.post("/conversation/session/import")
async def import_session(body: dict):
    """Import a conversation session from JSON."""
    data = body.get("data", "")
    session = conversation_manager.import_session(data)
    if not session:
        raise HTTPException(status_code=400, detail="Invalid session data")
    return {"session_id": session.session_id, "state": session.state}


# ═══════════════════════════════════════════════════════════════════════════════
#  Status Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/status")
async def status():
    """Get system status including AI availability."""
    return {
        "status": "ok",
        "gemini_available": gemini_available(),
        "planner": _planner_runtime_status(),
        "active_sessions": len(conversation_manager.sessions),
        "features": {
            "multi_turn_conversation": True,
            "design_explanations": True,
            "corrections": True,
            "multi_design_ranking": True,
        }
    }


# Development entrypoint: uvicorn api.server:app --reload

if __name__ == "__main__":
    import uvicorn
    reload_enabled = os.getenv("BLUEPRINT_DEV_RELOAD", "false").lower() == "true"
    host = os.getenv("BLUEPRINT_HOST", "127.0.0.1")
    port = int(os.getenv("BLUEPRINT_PORT", "8010"))
    print(f"Starting BlueprintGPT server...")
    print(f"Model checkpoint: {os.getenv('LAYOUT_MODEL_CHECKPOINT', 'learned/model/checkpoints/improved_v1.pt')}")
    print(f"Planner checkpoint: {os.getenv('BLUEPRINTGPT_PLANNER_CHECKPOINT', 'not set')}")
    print(f"Auto reload: {'enabled' if reload_enabled else 'disabled'}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"\n>>> Open http://{host}:{port} in your browser <<<\n")
    if reload_enabled:
        uvicorn.run("api.server:app", host=host, port=port, reload=True)
    else:
        uvicorn.run(app, host=host, port=port, reload=False)




