# BlueprintGPT Architecture

**Version**: 1.0
**Date**: 2026-03-20
**Status**: Production

---

## Overview

BlueprintGPT follows a **three-layer architecture** that cleanly separates concerns and enables multiple generation approaches:

```
1. Semantic Layer   →   Natural language → Structured requirements
2. Geometric Layer  →   Requirements → Building layout (IR)
3. Rendering Layer  →   Building → SVG/PNG/DXF output
```

This separation allows the system to support:
- **Learned models** (transformer-based)
- **Algorithmic generators** (rule-based)
- **Hybrid approaches** (model + constraints)
- **Multiple output formats** (SVG, DXF, raster)

---

## 1. Semantic Layer

**Purpose**: Convert natural language specifications into structured requirements.

### Components

| Component | Path | Purpose |
|-----------|------|---------|
| **Natural Language Interface** | `nl_interface/` | Parse user text into DesignSpec |
| **Specification Validator** | `constraints/spec_validator.py` | Validate and repair specs |
| **Design Spec** | `core/design_spec.py` | Intermediate representation |

### Data Flow

```
"2 bedrooms, 1 kitchen on 100 sqm plot"
        ↓
nl_interface/service.py::process_user_request()
        ↓
{
  "rooms": [{"type": "Bedroom", "count": 2}, {"type": "Kitchen", "count": 1}],
  "plot_area_sqm": 100,
  "boundary_polygon": [...],
  "preferences": {...},
  "weights": {...}
}
```

### Key Files
- `nl_interface/service.py` — Main parsing logic
- `nl_interface/constants.py` — Room types, relationships, weights
- `constraints/spec_validator.py` — Validation and repair

---

## 2. Geometric Layer

**Purpose**: Convert requirements into spatial layout solutions.

### Building IR (Intermediate Representation)

The **canonical IR** is the `Building` object:

```python
@dataclass
class Building:
    occupancy_type: str = "Residential"
    rooms: List[Room] = field(default_factory=list)
    doors: List[Door] = field(default_factory=list)
    corridors: List[Corridor] = field(default_factory=list)
    exit: Optional[Exit] = None
    # Scene graph representation of spatial relationships
```

Each `Room` contains:
```python
class Room:
    name: str              # "Bedroom1", "Kitchen1"
    room_type: str         # "Bedroom", "Kitchen"
    polygon: List[Tuple]   # [(x1,y1), (x2,y2), ...]
    final_area: float      # Sq meters
    doors: List[Door]      # Connected doors
    # Regulation compliance data
```

### Generation Pipelines

#### Learned Pipeline
```
DesignSpec → LayoutTransformer → RoomBoxes → Building → Repair → Building
```

| Step | Component | Path |
|------|-----------|------|
| **Tokenization** | `LayoutTokenizer` | `learned/data/tokenizer_layout.py` |
| **Generation** | `LayoutTransformer` | `learned/model/architecture.py` |
| **Sampling** | Constrained decoding | `learned/model/sample.py` |
| **Adaptation** | RoomBox → Building | `learned/integration/adapt_layout.py` |
| **Repair** | 8-stage pipeline | `learned/integration/repair_gate.py` |

#### Algorithmic Pipeline
```
DesignSpec → PolygonPacker → Building → Validation → Building
```

| Step | Component | Path |
|------|-----------|------|
| **Area Allocation** | Priority weights | `generator/area_allocator.py` |
| **Packing** | `PolygonPacker` | `geometry/polygon_packer.py` |
| **Corridor Planning** | Rule-based | `generator/corridor_planner.py` |
| **Door Placement** | Accessibility | `generator/door_placer.py` |

#### Hybrid Pipeline
```
DesignSpec → LayoutTransformer → Building → Algorithmic repair → Building
```

Combines learned generation with algorithmic repair for best of both approaches.

### Constraint System

**NBC India 2016 Chapter-4** ground truth is enforced at multiple stages:

| Stage | Component | Enforcement |
|-------|-----------|-------------|
| **Generation** | MinDimProcessor | Masks invalid tokens during sampling |
| **Post-generation** | Repair Gate | 8-stage deterministic repair |
| **Validation** | Ground Truth | Strong floor-plan validation |

**Canonical Source**: `ontology/regulation_data.json`
**Helper Functions**: `constraints/chapter4_helpers.py`

---

## 3. Rendering Layer

**Purpose**: Convert Building IR to visual outputs.

### SVG Generation

```
Building → SVG Renderer → SVG markup → File/HTTP response
```

**Primary Renderer**: `generator/svg_renderer.py`

#### SVG Structure
```xml
<svg viewBox="0 0 800 600">
  <!-- Boundary -->
  <polygon class="boundary" points="..." stroke="#000"/>

  <!-- Rooms -->
  <g id="room-bedroom1" class="room habitable">
    <polygon points="..." fill="#ffebcd" stroke="#666"/>
    <text x="..." y="...">Bedroom 1</text>
    <text x="..." y="..." class="area">12.5 m²</text>
  </g>

  <!-- Doors -->
  <g id="doors">
    <path d="M ... Q ... ..." class="door-arc"/>
  </g>

  <!-- Fixtures (future) -->
  <defs>
    <symbol id="door-swing" viewBox="0 0 90 90">...</symbol>
  </defs>
</svg>
```

#### Style Themes
- **Blueprint style**: Black lines, white background, technical appearance
- **Architectural style**: Colored rooms, filled areas, material textures
- **Furniture style**: Room layouts with furniture symbols

### Export Formats

| Format | Component | Use Case |
|--------|-----------|----------|
| **SVG** | `svg_renderer.py` | Web display, vector editing |
| **PNG** | SVG → raster | Print, thumbnails |
| **DXF** | `dxf_exporter.py` | CAD applications |
| **PDF** | SVG → PDF | Documents, reports |

---

## System Integration

### API Endpoints

**FastAPI Server** (`api/server.py`):

```python
POST /generate/learned
POST /generate/algorithmic
POST /generate/hybrid
GET  /building/{id}/svg
GET  /building/{id}/compliance
```

### Pipeline Selection

```python
def route_generation_request(spec, pipeline_type):
    if pipeline_type == "learned":
        return generate_layout_learned(spec)
    elif pipeline_type == "algorithmic":
        return generate_layout_algorithmic(spec)
    elif pipeline_type == "hybrid":
        return generate_layout_hybrid(spec)
```

### Quality Metrics

**Realism Scoring** (`learned/integration/realism_score.py`):
- **Min-dim compliance** (40%): Chapter-4 violations
- **Aspect ratios** (25%): Sliver detection
- **Travel feasibility** (20%): Distance estimates
- **Corridor continuity** (10%): Connected topology
- **Zoning** (5%): Functional room placement

**Repair Severity** (`RepairReport`):
- Displacement magnitude (meters)
- Violation counts (overlap, min-dim, topology)
- Stage breadth (which repair stages activated)

---

## Data Persistence

### Model Checkpoints
```
learned/model/checkpoints/
├── improved_v1.pt      # 25 epochs, coverage focus
├── improved_v2.pt      # 50 epochs, overlap focus
└── test_quick.pt       # Development/testing
```

**Environment Control**:
```bash
export LAYOUT_MODEL_CHECKPOINT="learned/model/checkpoints/improved_v2.pt"
```

### Training Data
```
learned/data/
├── kaggle_train_expanded.jsonl    # 5,904 samples (24x augmented)
├── kaggle_val_expanded.jsonl      # 496 samples
├── boundary_stats.json           # Calibration data
└── tokenizer_layout.py           # Token encoding/decoding
```

### Configuration Files
```
ontology/
├── regulation_data.json          # Chapter-4 ground truth (canonical)
└── regulatory.owl                # OWL ontology (future)
```

---

## Error Handling & Validation

### Validation Layers

1. **Input Validation** (`nl_interface/`)
   - Room type validation
   - Plot size bounds
   - Constraint conflicts

2. **Generation Validation** (`constraints/`)
   - Structural integrity
   - Polygon validity
   - Area conservation

3. **Compliance Validation** (`ground_truth/`)
   - Chapter-4 strong checking
   - Floor-plan level rules
   - Accessibility requirements

### Error Recovery

| Error Type | Recovery Strategy | Implementation |
|------------|-------------------|----------------|
| **Parse failures** | Graceful degradation | Default fallback specs |
| **Generation failures** | Retry with relaxed constraints | Candidate resampling |
| **Repair failures** | Algorithmic fallback | Hybrid pipeline switch |
| **Compliance failures** | Violation reporting | Detailed error messages |

---

## Performance & Scalability

### Model Inference
- **Batch processing**: Multiple candidates per request
- **GPU utilization**: CUDA support when available
- **Constrained sampling**: Real-time masking during generation

### Caching Strategy
```python
@lru_cache(maxsize=1000)
def get_regulation_constraints(occupancy: str, plot_area: float):
    # Cache Chapter-4 lookups
```

### Monitoring
- **Generation metrics**: Success rate, repair severity
- **Quality tracking**: Realism scores, compliance rates
- **Performance**: Inference time, memory usage

---

## Future Architecture

### Planned Enhancements

1. **Multi-floor Support**
   ```python
   class Building:
       floors: List[Floor]  # Currently implicit single floor
   ```

2. **Real-time Collaboration**
   - WebSocket API for live editing
   - Operational transforms for conflict resolution

3. **Advanced Rendering**
   - 3D visualization via Three.js
   - VR/AR integration
   - Photorealistic materials

4. **ML Pipeline Improvements**
   - RL-based room ordering
   - Diffusion models for layout generation
   - Multi-modal input (sketches, images)

### Extensibility Points

- **New constraint systems**: Additional building codes
- **Custom renderers**: Domain-specific output formats
- **Alternative models**: Plug-in architecture for generators
- **External integrations**: BIM tools, architectural software

---

## References

- [NBC India 2016](https://www.bis.gov.in/): National Building Code regulations
- [Chapter-4 Schema](docs/CHAPTER4_SCHEMA.md): Ground truth documentation
- [MinDim Calibration](docs/MINDIM_CALIBRATION.md): Learned model calibration
- [API Documentation](api/): FastAPI endpoints and schemas

---

**Maintained by**: BlueprintGPT Engineering Team
**Last Updated**: 2026-03-20
**Next Review**: Q2 2026