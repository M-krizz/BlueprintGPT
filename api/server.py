from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import List, Literal, Optional
from uuid import uuid4

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
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
from nl_interface.service import process_user_request
from nl_interface.runner import run_algorithmic_backend, run_learned_backend, run_hybrid_backend
from nl_interface.conversation import conversation_manager, ConversationSession
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


class RoomSpec(BaseModel):
    name: str = Field(..., description="Room identifier, e.g. Bedroom_1")
    type: str = Field(..., description="Room type, e.g. Bedroom, Kitchen")
    area: Optional[float] = Field(None, description="Optional target area in square meters")


class Boundary(BaseModel):
    width: float
    height: float


class GenerateRequest(BaseModel):
    backend_target: Literal["algorithmic", "learned", "hybrid"] = "hybrid"
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


app = FastAPI(title="BlueprintGPT API", version="1.0")

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*80)
    print("  BlueprintGPT Server STARTED - logging enabled")
    print("  Endpoints:")
    print("    POST /conversation/session/new  -> create session")
    print("    POST /conversation/message      -> conversation API (primary)")
    print("    POST /chat/generate             -> legacy API (fallback, stateless)")
    print("="*80 + "\n")

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
    else:
        polygon = [
            (0.0, 0.0),
            (boundary.width, 0.0),
            (boundary.width, boundary.height),
            (0.0, boundary.height),
        ]
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
        if not isinstance(boundary, (tuple, list)) or len(boundary) < 2:
            boundary = (12, 15)

        try:
            plot_area = float(boundary[0]) * float(boundary[1])
        except (TypeError, ValueError):
            plot_area = 180

        # Build room summary
        room_summary = ", ".join([f"{r.get('count', 1)} {r.get('type')}" for r in rooms]) if rooms else "the requested rooms"

        explanation = "I wasn't able to generate a floor plan that meets all quality requirements. Let me explain:\n\n"

        error_lower = error_str.lower() if error_str else ""

        # Travel distance issue
        if "travel distance" in error_lower:
            explanation += "**Problem: Rooms are too far apart**\n"
            if room_count > 0:
                explanation += f"With {room_count} rooms in a {boundary[0]}m x {boundary[1]}m plot ({plot_area:.0f} sq.m), "
            explanation += "the walking distance between important rooms (like bedroom to bathroom) would be too long.\n\n"

        # Adjacency satisfaction issue
        if "adjacency" in error_lower:
            explanation += "**Problem: Room placement conflicts**\n"
            explanation += "The rooms couldn't be arranged to satisfy the expected relationships "
            explanation += "(e.g., kitchen near living room, bathrooms near bedrooms).\n\n"

        # Quality gate rejection
        if "quality gate" in error_lower or "rejected all variants" in error_lower:
            explanation += "**What this means:** The system tried multiple layout configurations but none met the minimum quality standards for a functional home.\n\n"

        # Provide suggestions based on the issue
        explanation += "**Suggestions to fix this:**\n\n"

        # Check if plot might be too small
        min_area_per_room = 12  # rough estimate: 12 sq.m per room minimum
        if room_count > 0 and plot_area < room_count * min_area_per_room:
            min_dim = int((room_count * min_area_per_room) ** 0.5) + 2
            explanation += f"1. **Increase plot size** - Your {room_count} rooms need approximately {room_count * min_area_per_room} sq.m minimum. "
            explanation += f"Current plot is only {plot_area:.0f} sq.m. Try setting dimensions to at least {min_dim}m x {min_dim}m in Settings.\n\n"
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
        explanation += "Just let me know what you'd like to adjust!"

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
                or "learned/model/checkpoints/kaggle_test.pt",
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
        body.boundary.dict() if body.boundary else None,
        body.entrance_point,
        body.generate
    )

    # Get or create session
    session = conversation_manager.get_or_create_session(body.session_id)

    if DetailedLogger.enabled():
        DetailedLogger.log_detailed_state("SESSION", {
            "state": session.state,
            "current_rooms": session.current_spec.get('rooms', [])
        })

    # Add user message to history
    session.add_message("user", body.message)

    # Set initial resolution from frontend if provided
    resolution = None
    if body.boundary:
        resolution = {
            "boundary_size": (body.boundary.width, body.boundary.height),
            "area_unit": APIDefaults.DEFAULT_AREA_UNIT,
        }
        if body.boundary_polygon:
            resolution["boundary_polygon"] = body.boundary_polygon
        if body.entrance_point:
            resolution["entrance_point"] = tuple(body.entrance_point)
        session.set_resolution(resolution)

    # Build context for intent classification
    context = {
        "state": session.state,
        "num_designs": len(session.designs),
        "spec": session.current_spec,
        "selected_design": session.selected_design_index,
        "current_rooms": [{"type": r.get("type"), "name": r.get("name")} for r in session.current_spec.get("rooms", [])],
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
        cli_overrides = _extract_cli_args(body.message)

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
            if body.entrance_point:
                resolution["entrance_point"] = tuple(body.entrance_point)
            session.set_resolution(resolution)
        else:
            ProcessingLogger.logger.debug(f"Using frontend dimensions: {resolution.get('boundary_size') if resolution else 'Default will be applied'}")

    # Handle based on intent
    if intent == INTENT_DESIGN and nl_result.get("should_generate") and body.generate:
        ProcessingLogger.logger.info("Starting design generation pipeline")

        # Extract and update spec
        extracted = nl_result.get("spec", {})
        ProcessingLogger.logger.debug(f"Extracted spec: {extracted}")
        session.update_spec(extracted)

        ProcessingLogger.logger.debug(f"Current session spec: {session.current_spec}")
        ProcessingLogger.logger.debug(f"Current resolution: {session.resolution}")

        # Auto-dimension selection if no dimensions specified
        current_resolution = session.resolution or {}
        boundary_size = current_resolution.get("boundary_size")

        if not boundary_size or boundary_size == (12.0, 15.0):  # Default frontend dimensions
            from nl_interface.auto_dimension_selector import recommend_dimensions, explain_dimension_choice

            rooms = session.current_spec.get("rooms", [])
            if rooms:
                ProcessingLogger.logger.info("No custom dimensions specified - calculating optimal size")

                width, height = recommend_dimensions(rooms, building_type="residential")
                explanation = explain_dimension_choice(rooms, width, height)
                ProcessingLogger.logger.info(explanation)

                # Update resolution with auto-calculated dimensions
                if current_resolution:
                    current_resolution["boundary_size"] = (width, height)
                else:
                    current_resolution = {
                        "boundary_size": (width, height),
                        "area_unit": "sq.m",
                    }

                if body.entrance_point:
                    current_resolution["entrance_point"] = tuple(body.entrance_point)
                else:
                    # Auto-center entrance on the shorter side
                    if width <= height:
                        current_resolution["entrance_point"] = (width / 2, 0)  # Bottom center
                    else:
                        current_resolution["entrance_point"] = (0, height / 2)  # Left center

                session.set_resolution(current_resolution)
                ProcessingLogger.logger.debug(f"Updated resolution: {current_resolution}")
            else:
                ProcessingLogger.logger.warning(f"No rooms specified - using default dimensions {boundary_size}")

        # Check if spec is complete via process_user_request
        nl_response = process_user_request(
            body.message,
            current_spec=session.current_spec,
            resolution=session.resolution,
        )

        spec_complete = nl_response.get("backend_ready", False)
        missing = nl_response.get("missing_fields", [])
        backend_spec = nl_response.get("backend_spec")

        ProcessingLogger.log_generation_pipeline(
            spec_complete=spec_complete,
            missing_fields=missing,
            backend_target=nl_response.get('backend_target'),
            backend_ready=bool(backend_spec)
        )

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
                final_response = f"I've generated {len(explained_designs)} design(s) for you.\n\n{comparison}"
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
                ProcessingLogger.logger.debug(f"User-friendly explanation generated ({len(error_explanation)} chars)")

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
                )
        else:
            # Need more info for design
            response_text = nl_result.get("response", "")
            if missing:
                response_text += f"\n\nI still need: {', '.join(missing)}"
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
            )

    elif intent == INTENT_CORRECTION:
        # Handle correction request
        response_text = nl_result.get("response", "I see you want to make changes. Please specify which design to modify.")
        session.add_message("assistant", response_text)
        return ConversationMessageResponse(
            session_id=session.session_id,
            assistant_text=response_text,
            state=session.state,
            spec_complete=False,
            current_spec=session.current_spec,
            designs=None,
            comparison=None,
            needs_info=["design_selection"],
        )

    else:
        # INTENT_QUESTION or INTENT_CONVERSATION - use dynamic response
        response_text = nl_result.get("response", "")
        if not response_text:
            # Fallback to gemini_chat if process_message didn't generate response
            response_text = gemini_chat(body.message, context, session.get_history(limit=10))

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
    print(f"Starting BlueprintGPT server...")
    print(f"Model checkpoint: {os.getenv('LAYOUT_MODEL_CHECKPOINT', 'learned/model/checkpoints/improved_v1.pt')}")
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
