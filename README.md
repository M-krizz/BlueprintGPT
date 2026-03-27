# BlueprintGPT: AI-Powered Residential Floor Plan Generator

## Overview

**BlueprintGPT** is an intelligent system that generates professional floor plans for residential buildings using natural language processing and advanced algorithmic design techniques. Users can describe their ideal home in plain English, and the system generates multiple optimized floor plan variants that comply with architectural and building regulations.

### Key Features

- 🗣️ **Natural Language Interface**: Describe rooms, dimensions, and preferences in plain English
- 🏠 **Multi-Strategy Generation**: Choose from algorithmic, learned (neural), planner-based, or hybrid approaches
- ✅ **Compliance Validation**: Automatic adherence to building codes and regulations
- 📊 **Design Ranking**: Multiple variants ranked by architectural quality metrics
- 🔄 **Interactive Refinement**: Iterative corrections and adjustments through conversation
- 📐 **Professional Output**: SVG floor plans with CAD-style rendering, dimensions, and compliance reports
- 🧠 **Learned Models**: Neural models trained on diverse residential layouts for human-like designs

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Web UI)                        │
│                   (React/HTML5 Interactive Chat)                │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/WebSocket
┌────────────────────────▼────────────────────────────────────────┐
│                    FastAPI Server (API)                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Chat Endpoints                                          │  │
│  │  - /api/chat (multi-turn conversation)                  │  │
│  │  - /api/generate (floor plan generation)                │  │
│  │  - /api/correct (design refinement)                     │  │
│  │  - /api/explain (design explanation)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────┬─────────────────────────────┬──────────────────┘
                 │                             │
        ┌────────▼──────────┐      ┌──────────▼──────────┐
        │   NL Interface    │      │   Conversation Mgr  │
        │  (Gemini Adapter) │      │  (State Management) │
        └────────┬──────────┘      └──────────┬──────────┘
                 │                             │
        ┌────────▼──────────────────────────────▼──────────┐
        │         Spec Processing Pipeline                 │
        │  ┌─────────────────────────────────────────────┐ │
        │  │ • Extract rooms, dimensions, preferences   │ │
        │  │ • Validate against ontology               │ │
        │  │ • Normalize and enhance specifications    │ │
        │  │ • Route to appropriate backend            │ │
        │  └─────────────────────────────────────────────┘ │
        └────────┬───────────────┬──────────────┬──────────┘
                 │               │              │
    ┌────────────▼──┐  ┌─────────▼─────┐  ┌────▼──────────┐
    │  Algorithmic  │  │  Learned      │  │  Planner      │
    │  Backend      │  │  Neural Model │  │  Backend      │
    │  (Layout Gen) │  │  (Gen Loop)   │  │  (LSTM-based) │
    └────────────┬──┘  └────────┬──────┘  └────┬──────────┘
                 │              │              │
        ┌────────▼──────────────▼──────────────▼────────┐
        │      Design Post-Processing                   │
        │  ┌──────────────────────────────────────────┐ │
        │  │ • Corridor planning and placement       │ │
        │  │ • Door and window placement             │ │
        │  │ • Polygon snapping and alignment       │ │
        │  │ • Wall generation                       │ │
        │  └──────────────────────────────────────────┘ │
        └────────────┬─────────────────────────────────┘
                     │
        ┌────────────▼──────────────┐
        │   Constraint Engine       │
        │  ┌──────────────────────┐ │
        │  │ • Compliance check   │ │
        │  │ • Repair loop       │ │
        │  │ • Regulation engine │ │
        │  └──────────────────────┘ │
        └────────────┬──────────────┘
                     │
        ┌────────────▼──────────────┐
        │   Ranking & Scoring       │
        │  ┌──────────────────────┐ │
        │  │ • Design metrics     │ │
        │  │ • Adjacency score    │ │
        │  │ • Alignment score    │ │
        │  │ • Selection (top-K)  │ │
        │  └──────────────────────┘ │
        └────────────┬──────────────┘
                     │
        ┌────────────▼──────────────┐
        │  Output Generation        │
        │  ┌──────────────────────┐ │
        │  │ • SVG floor plans   │ │
        │  │ • Compliance report │ │
        │  │ • Design stats      │ │
        │  └──────────────────────┘ │
        └──────────────────────────┘
```

---

## Module Breakdown

### 1. **API Layer** (`api/server.py`)
- FastAPI application with CORS middleware
- RESTful endpoints for chat, generation, correction, and explanation
- Request/response validation with Pydantic models
- Session management and conversation tracking

**Key Endpoints:**
- `POST /api/chat` - Multi-turn conversational interface
- `POST /api/generate` - Direct floor plan generation
- `POST /api/correct` - Request design modifications
- `GET /api/explain/{design_id}` - Generate explanations for designs

---

### 2. **Natural Language Processing** (`nl_interface/`)

#### `gemini_adapter.py`
- **Intent Classification**: Detects whether user wants design, correction, question, or conversation
- **NL→Spec Conversion**: Parses natural language into structured floor plan specifications
- **Design Explanation**: Generates human-readable explanations for generated layouts
- **Correction Parsing**: Interprets user feedback for design refinement

#### `service.py`
- **Spec Processing**: Normalizes and validates user specifications
- **Dimension Extraction**: Parses natural language dimensions (e.g., "10x12 meters")
- **Room Program Builder**: Constructs detailed room specifications
- **Constraint Enhancement**: Enriches specs with building code requirements

#### `conversation.py`
- **Session State Management**: Tracks accumulated specifications across turns
- **Conversation History**: Maintains chat context for coherent responses
- **Design Registry**: Stores generated designs and their metadata
- **State Serialization**: Saves/loads conversation sessions

#### `adapter.py`
- **Backend Routing**: Intelligently selects appropriate generation backend
- **Spec Validation**: Ensures specifications match backend requirements
- **Room Mapping**: Converts between external and internal room type names

---

### 3. **Generation Backends**

#### A. **Algorithmic Backend** (`generator/`)
Traditional rule-based approach using geometric algorithms.

**Pipeline:**
1. **Layout Generation** (`layout_generator.py`)
   - Recursive polygon packing (bisection algorithm)
   - Area-based room allocation
   - Aspect ratio optimization

2. **Polygon Packing** (`geometry/polygon_packer.py`)
   - Continuous recursive bisection
   - Maintains exact room areas
   - Handles arbitrary boundary shapes

3. **Door & Window Placement** (`geometry/door_placer.py`, `geometry/window_placer.py`)
   - Strategic door placement on room boundaries
   - Connectivity graph construction
   - Window suggestions for maximum natural light

4. **Corridor Planning** (`geometry/corridor_placer.py`)
   - Hub-based circulation generation
   - Compact corridor networks
   - Circulation space optimization

5. **Wall Generation** (`geometry/walls.py`)
   - Shared wall optimization
   - Wall thickness management
   - Building envelope construction

#### B. **Learned Backend** (`learned/integration/`)
Neural network-based generation using LayoutTransformer.

**Pipeline:**
1. **Model Sampling** (`learned/model/sample.py`)
   - Load trained LayoutTransformer checkpoint
   - Generate diverse room layout candidates
   - Sampling with temperature and top-K/top-P controls

2. **Repair Gate** (`learned/integration/repair_gate.py`)
   - Validate generated layouts
   - Enforce hard constraints (connectivity, minimum areas)
   - Iterative repair of constraint violations

3. **Reranking** (`learned/integration/prerank.py`)
   - Pre-filter candidates by validity
   - Score and rank variants

#### C. **Planner Backend** (`learned/planner/`)
LSTM-based room planner that determines room adjacency and positioning.

**Features:**
- Learns spatial relationships from training data
- Predicts optimal room orderings
- Complements both algorithmic and learned backends

#### D. **Hybrid Backend** (`nl_interface/runner.py`)
Combines multiple backends for best results:
- Generates candidates from learned models
- Falls back to algorithmic for coverage
- Ranks all candidates together

---

### 4. **Core Data Models** (`core/`)

**Building** - Container for rooms, doors, corridors, and exit
**Room** - Individual room with geometry, type, area, and constraints
**Door** - Doorway with connection information
**Exit** - Emergency exit with location
**Corridor** - Circulation space with routing paths

---

### 5. **Geometry Engine** (`geometry/`)

| Module | Purpose |
|--------|---------|
| `polygon_packer.py` | Recursive bisection packing algorithm |
| `allocator.py` | Area allocation and distribution |
| `polygon.py` | Polygon operations and snapping |
| `walls.py` | Wall segment generation and merging |
| `adjacency.py` | Room adjacency analysis |
| `corridor_first_planner.py` | Circulation-first layout generation |
| `zoning.py` | Room zone classification (public/private/service) |

---

### 6. **Constraint Engine** (`constraints/`)

| Module | Purpose |
|--------|---------|
| `rule_engine.py` | Enforce building codes and regulations |
| `spec_validator.py` | Validate input specifications |
| `repair_loop.py` | Fix constraint violations iteratively |
| `compliance_report.py` | Generate detailed compliance reports |

---

### 7. **Graph Analysis** (`graph/`)

| Module | Purpose |
|--------|---------|
| `connectivity.py` | Check room connectivity and reachability |
| `door_graph_path.py` | Calculate travel distances via doors |
| `manhattan_path.py` | Calculate maximum travel distances |

---

### 8. **Configuration & Constants** (`config/`)

**constants.py** - Centralized configuration:
- Room types (Bedroom, Kitchen, Bathroom, LivingRoom, etc.)
- Default dimensions for plot types
- Intent classification constants
- Layout standards and aspect ratios
- Preferred room area ranges

**ontology/** - Building regulations database:
- `regulation_data.json` - Building codes per room type
- `building_spec.schema.json` - JSON schema validation
- Occupancy-specific requirements

---

### 9. **Visualization** (`visualization/`)

**export_svg_blueprint.py** - Professional SVG rendering:
- CAD-style wall rendering
- Room labels with areas
- Door swing arcs and openings
- Window placements
- Dimension strings
- Compass rose and scale bar
- Compliance color coding

---

### 10. **Explainability** (`explain/`)

**llm_explainer.py** - Design explanation generation:
- Deterministic metric summaries
- Optional LLM-powered natural language descriptions
- Violation and compliance reporting

---

## Data Flow Diagram

### Typical User Journey

```
1. User Message (Natural Language)
   ↓
2. Intent Classification (Gemini)
   ├─ Design Request → Extract Spec
   ├─ Correction → Parse Changes
   ├─ Question → Answer Directly
   └─ Conversation → Continue Chat
   ↓
3. Specification Processing
   ├─ Extract dimensions, rooms, preferences
   ├─ Normalize and validate
   ├─ Enhance with constraints
   └─ Route to backend
   ↓
4. Backend Selection
   ├─ Algorithmic: for standard room programs
   ├─ Learned: for diverse/custom programs
   ├─ Planner: for structured planning
   └─ Hybrid: for best results
   ↓
5. Layout Generation
   ├─ Generate K candidates
   ├─ Repair constraint violations
   └─ Compliance validation
   ↓
6. Post-Processing
   ├─ Corridor placement
   ├─ Door/window placement
   ├─ Wall generation
   └─ Alignment optimization
   ↓
7. Ranking & Selection
   ├─ Calculate design metrics
   ├─ Evaluate adjacency
   ├─ Score alignment
   └─ Select top-M variants
   ↓
8. Output Generation
   ├─ Render SVG floor plans
   ├─ Generate compliance report
   ├─ Collect design statistics
   └─ Save artifacts
   ↓
9. Response & Explanation
   ├─ Present designs to user
   ├─ Generate explanations
   ├─ Highlight key features
   └─ Wait for feedback
   ↓
10. [Loop back for corrections]
```

---

## Configuration

### Environment Variables

```bash
# Backend Selection
BLUEPRINT_BACKEND_MODE=auto  # Options: auto, algorithmic, planner, learned, hybrid
BLUEPRINT_AUTO_CORE_BACKEND=algorithmic  # Options: algorithmic, planner, planner_if_available

# Model Checkpoints
BLUEPRINTGPT_CHECKPOINT=learned/model/checkpoints/improved_v1.pt
BLUEPRINTGPT_PLANNER_CHECKPOINT=learned/planner/checkpoints/room_planner.pt

# LLM Configuration
GEMINI_API_KEY=<your-gemini-api-key>
GEMINI_MODEL=gemini-2.5-flash
GEMINI_ENABLED=true

# Server Configuration
API_HOST=localhost
API_PORT=8000
LOG_LEVEL=INFO
```

---

## Installation

### Prerequisites

- Python 3.12+
- pip or conda

### Setup

```bash
# Clone repository
git clone https://github.com/M-krizz/BlueprintGPT.git
cd BlueprintGPT

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Gemini API key and configuration
```

### Running the Server

```bash
# Start FastAPI server
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

# Server runs at: http://localhost:8000
# API docs at: http://localhost:8000/docs
# Frontend at: http://localhost:8000/
```

---

## Usage Examples

### 1. Via Web Interface

1. Open `http://localhost:8000/` in your browser
2. Type your design request: *"I want a 2BHK apartment with an open kitchen-living area and good natural light"*
3. System generates multiple floor plan variants
4. Click designs to see detailed compliance reports
5. Request corrections: *"Make the master bedroom larger and move the kitchen next to dining"*
6. System refines and regenerates designs

### 2. Via API (Python)

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Generate floor plan
request = {
    "backend_target": "hybrid",
    "boundary": {"width": 12.0, "height": 15.0},
    "rooms": [
        {"name": "MasterBedroom", "type": "Bedroom"},
        {"name": "Bedroom2", "type": "Bedroom"},
        {"name": "Kitchen", "type": "Kitchen"},
        {"name": "Bathroom", "type": "Bathroom"},
        {"name": "LivingRoom", "type": "LivingRoom"},
    ],
    "preferences": {
        "adjacency": [
            {"room1": "Kitchen", "room2": "LivingRoom", "weight": 1.5}
        ]
    }
}

response = requests.post(f"{BASE_URL}/api/generate", json=request)
result = response.json()

print(f"Status: {result['status']}")
print(f"Generated layouts: {len(result['artifact_paths'])}")
for i, artifact in enumerate(result['artifact_paths']):
    print(f"  {i+1}. {artifact['svg_path']}")
```

### 3. Via Command Line

```bash
# Generate layout from spec file
python -m nl_interface.cli \
    --spec-file input_spec.json \
    --backend hybrid \
    --output-dir outputs/

# Process natural language request
python -m nl_interface.runner \
    --text "I need a 3BHK with master ensuite" \
    --boundary 15x18 \
    --backend algorithmic
```

---

## Output Artifacts

### Generated Outputs Directory

```
outputs/
├── {session_id}/
│   ├── design_1.svg                 # SVG floor plan
│   ├── design_1_compliance.json     # Compliance report
│   ├── design_1_metrics.json        # Design metrics
│   ├── design_2.svg
│   ├── design_2_compliance.json
│   ├── ...
│   ├── selection_report.json        # Top designs summary
│   └── session_log.json             # Session metadata
```

### SVG Output Features

- **Wall Rendering**: Outer and inner walls with proper thickness
- **Room Labels**: Name + area in each room
- **Dimension Strings**: Boundary and room dimensions
- **Door Symbols**: Swing arcs indicate door swing direction
- **Window Markings**: Window locations with tick marks
- **Entrance**: Marked with special symbol
- **Zone Colors**: Public (yellow), Private (blue), Service (gray)
- **Corridor Hatching**: Diagonal pattern for circulation
- **Title Block**: With scale bar and compass

---

## Design Metrics

The system evaluates each generated layout using multiple metrics:

### Geometric Metrics
- **Alignment Score** (0-1): How well rooms align to grid
- **Circulation Factor**: Percentage of total area used for corridors
- **Aspect Ratio Score**: How close to preferred room proportions
- **Composition Quality**: Visual balance and symmetry

### Functional Metrics
- **Adjacency Satisfaction**: How well user preferences for room proximity are met
- **Connectivity Score**: Efficiency of movement between rooms
- **Room Area Compliance**: Rooms within min/max standards
- **Travel Distance**: Maximum distance from any room to exit

### Architectural Metrics
- **Natural Light Exposure**: Window count and placement
- **Privacy Zones**: Proper separation of public/private areas
- **Wall Efficiency**: Shared walls minimized
- **Structural Feasibility**: Load-bearing walls feasible

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suite

```bash
# Unit tests
pytest tests/unit/test_polygon_packer.py -v

# API tests
pytest tests/unit/test_api_conversation_generation.py -v

# Planner tests
pytest tests/unit/test_planner_inference_defaults.py -v
```

### Generate Coverage Report

```bash
pytest tests/ --cov=. --cov-report=html
```

---

## Architecture Decisions

### Why Multiple Backends?

1. **Algorithmic**: Deterministic, fast, interpretable, good for standard layouts
2. **Learned**: Humanlike, diverse, handles complex adjacencies
3. **Planner**: Structured reasoning, learned spatial relationships
4. **Hybrid**: Best of all worlds, robustly falls back

### Why Polygon Packing?

- **Guarantees**: 100% area utilization, no gaps/overlaps
- **Flexibility**: Works with any boundary shape
- **Accuracy**: Maintains exact room areas
- **Scalability**: Efficient even for 20+ rooms

### Why Repair Loop?

- Generated layouts may violate constraints
- Repair loop fixes violations iteratively:
  1. Room resizing for area compliance
  2. Connectivity restoration
  3. Minimum dimension enforcement
  4. Travel distance adjustment
- Preserves original design intent while achieving compliance

---

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Spec parsing | 50-200ms | NLP, spec extraction |
| Layout generation (algorithmic) | 100-500ms | Per candidate |
| Layout generation (learned) | 150-800ms | Per candidate + model forward pass |
| Repair loop | 50-300ms | Per layout |
| Compliance checking | 20-100ms | Per layout |
| SVG rendering | 100-400ms | Per layout |
| **Total (3 candidates, hybrid)** | **2-4 seconds** | End-to-end including NLP |

---

## Known Limitations

1. **Room Program Size**: Currently optimized for 4-12 rooms (scalability being improved)
2. **Plot Shapes**: Works best with rectangular and L-shaped plots (polygon handling in progress)
3. **Complex Adjacencies**: Limited support for non-planar relationships (planner improvements planned)
4. **Learned Models**: Training data biased toward Indian residential conventions (generalization planned)
5. **Material Specifications**: Not yet generating material specs (future phase)

---

## Future Enhancements

### Phase 2
- [ ] Multi-level/multi-story support
- [ ] Material selection integration
- [ ] Cost estimation
- [ ] 3D visualization

### Phase 3
- [ ] Structural engineering integration
- [ ] MEP (Mechanical, Electrical, Plumbing) routing
- [ ] Furniture placement suggestions
- [ ] Virtual tour generation

### Phase 4
- [ ] Multi-user collaboration
- [ ] Design versioning and branching
- [ ] Regulatory compliance marketplace
- [ ] Integration with CAD software (AutoCAD, Revit)

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Citation

If you use BlueprintGPT in your research, please cite:

```bibtex
@software{blueprintgpt2024,
  title={BlueprintGPT: AI-Powered Residential Floor Plan Generator},
  author={Your Name},
  year={2024},
  url={https://github.com/M-krizz/BlueprintGPT}
}
```

---

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/M-krizz/BlueprintGPT/issues)
- **Email**: support@blueprintgpt.dev
- **Documentation**: [Full Docs](https://blueprintgpt.readthedocs.io)

---

**Last Updated**: March 2026  
**Version**: 3.0.0
