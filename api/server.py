from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import List, Literal, Optional
from uuid import uuid4

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
)
from nl_interface.explainer import explain_ranked_designs, generate_comparison_explanation
from nl_interface.correction_handler import handle_correction_request


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
    Handle a conversation message with multi-turn support.

    This endpoint:
    1. Maintains conversation state across turns
    2. Extracts spec from natural language (using Gemini if available)
    3. Generates multiple ranked designs when spec is complete
    4. Provides AI explanations for each design
    """
    print(f"\n{'#'*100}")
    print(f"[CONVERSATION] Received message endpoint request")
    print(f"[CONVERSATION] Session ID: {body.session_id}")
    print(f"[CONVERSATION] User message: '{body.message}'")
    print(f"[CONVERSATION] Boundary: {body.boundary}")
    print(f"[CONVERSATION] Generate flag: {body.generate}")
    print(f"{'#'*100}\n")

    # Get or create session
    session = conversation_manager.get_or_create_session(body.session_id)
    print(f"[CONVERSATION] Session state: {session.state}")
    print(f"[CONVERSATION] Current spec rooms: {session.current_spec.get('rooms', [])}")

    # Add user message to history
    session.add_message("user", body.message)

    # Set resolution if provided
    if body.boundary:
        resolution = {
            "boundary_size": (body.boundary.width, body.boundary.height),
            "area_unit": "sq.m",
        }
        if body.boundary_polygon:
            resolution["boundary_polygon"] = body.boundary_polygon
        if body.entrance_point:
            resolution["entrance_point"] = tuple(body.entrance_point)
        session.set_resolution(resolution)

    # Extract spec from message using Gemini
    print(f"\n[CONVERSATION] Extracting spec from message using Gemini...")
    extracted = extract_spec_from_nl(body.message, session.get_history(limit=10))
    print(f"[CONVERSATION] Gemini extraction result:")
    print(f"  - intent: {extracted.get('intent')}")
    print(f"  - rooms: {extracted.get('rooms')}")
    print(f"  - adjacency: {extracted.get('adjacency')}")
    print(f"  - plot_type: {extracted.get('plot_type')}")
    print(f"  - style_hints: {extracted.get('style_hints')}")

    # Check if this is a correction request
    if extracted.get("intent") == "correction":
        # Handle as correction
        response_text = "I see you want to make changes. Please use the correction endpoint or specify which design to modify."
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

    # Update spec with extracted data
    print(f"\n[CONVERSATION] Updating session spec with extracted data...")
    session.update_spec(extracted)
    print(f"[CONVERSATION] Updated spec rooms: {session.current_spec.get('rooms', [])}")

    # Check if spec is complete
    print(f"\n[CONVERSATION] Calling process_user_request to check completeness...")
    nl_response = process_user_request(
        body.message,
        current_spec=session.current_spec,
        resolution=session.resolution,
    )

    spec_complete = nl_response.get("backend_ready", False)
    missing = nl_response.get("missing_fields", [])

    print(f"[CONVERSATION] process_user_request result:")
    print(f"  - backend_ready: {spec_complete}")
    print(f"  - missing_fields: {missing}")
    print(f"  - backend_target: {nl_response.get('backend_target')}")
    print(f"  - rooms in nl_response: {nl_response.get('rooms', [])}")

    # Generate response text
    if spec_complete and body.generate:
        response_text = "I have all the information I need. Generating your floor plan designs..."
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
            error_text = f"I encountered an error while generating designs: {str(exc)}"
            session.add_message("assistant", error_text)
            return ConversationMessageResponse(
                session_id=session.session_id,
                assistant_text=error_text,
                state=session.state,
                spec_complete=False,
                current_spec=session.current_spec,
                designs=None,
                comparison=None,
                needs_info=["generation_error"],
            )

    else:
        # Need more information
        context = session.get_context()
        if gemini_available():
            response_text = gemini_chat(body.message, context, session.get_history(limit=10))
        else:
            response_text = nl_response.get("assistant_text", "")
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
