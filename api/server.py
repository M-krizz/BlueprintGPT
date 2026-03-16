from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import List, Literal, Optional
from uuid import uuid4

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from nl_interface.runner import execute_response
from nl_interface.service import process_user_request
from nl_interface.runner import run_algorithmic_backend, run_learned_backend, run_hybrid_backend


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the lightweight UI under /ui
ui_dir = Path(__file__).parent.parent / "frontend"
outputs_dir = Path(__file__).parent.parent / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")
app.mount("/outputs", StaticFiles(directory=str(outputs_dir), html=False), name="outputs")


@app.get("/")
async def root():
    if ui_dir.exists():
        return RedirectResponse(url="/ui")
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/favicon.ico", status_code=204)
async def favicon():
    # Eliminates the 404 console error on browser requests
    return None


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
    target = Path(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    try:
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/ui-config")
async def ui_config():
    return {"title": "BlueprintGPT", "default_backend": "algorithmic"}


# Development entrypoint: uvicorn api.server:app --reload
