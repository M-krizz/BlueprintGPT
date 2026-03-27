# Architecture Design Document

## Table of Contents

1. [System Overview](#system-overview)
2. [Architectural Patterns](#architectural-patterns)
3. [Module Design](#module-design)
4. [Data Flow](#data-flow)
5. [Design Decisions](#design-decisions)
6. [Integration Points](#integration-points)
7. [Scalability Considerations](#scalability-considerations)

---

## System Overview

### Purpose

BlueprintGPT is an AI-driven system that transforms natural language descriptions into professionally optimized floor plans. It combines:

- **Natural Language Understanding** (Gemini API)
- **Geometric Algorithms** (Polygon packing, graph algorithms)
- **Machine Learning** (LayoutTransformer neural networks)
- **Constraint Satisfaction** (Compliance engine)

### Design Philosophy

1. **Modular**: Each component is independently testable and replaceable
2. **Extensible**: Easy to add new backends or constraint types
3. **Interpretable**: Design decisions are explainable and debuggable
4. **Robust**: Graceful degradation with fallback mechanisms
5. **Performant**: Fast response times for interactive use

---

## Architectural Patterns

### 1. **Pipeline Pattern**

The generation process follows a clear pipeline:

```
Input Spec → Validation → Backend Generation → Post-Processing → 
Constraint Checking → Ranking → Output
```

**Benefit**: Each stage can be independently optimized, tested, and modified.

### 2. **Strategy Pattern**

Multiple generation backends implement the same interface:

```python
Backend (interface):
    - generate(spec, boundary) → [Layout]
    - validate() → bool
    - get_name() → str

Implementations:
    - AlgorithmicBackend
    - LearnedBackend
    - PlannerBackend
    - HybridBackend
```

**Benefit**: Runtime backend selection based on input characteristics.

### 3. **Adapter Pattern**

Multiple adapters normalize external inputs:

```
User Request → [Gemini Adapter] → NL Spec
                [ChatSpec Adapter] → Chat Spec
                [Schema Adapter] → JSON Schema
                ↓
        Canonical Internal Spec
```

**Benefit**: Insulates core logic from external format changes.

### 4. **Constraint-Solving Pattern**

Compliance is handled through:

1. **Rule Engine**: Hard constraints (connectivity, travel distance)
2. **Repair Loop**: Iterative constraint relaxation
3. **Ranking**: Soft constraint optimization (user preferences)

```
Generated Layout → [Rule Engine] → Valid? 
                        ↓ (No)
                  [Repair Loop]
                        ↓
                  [Rule Engine] → Valid? (Retry)
                        ↓ (Yes)
                  [Ranking System]
```

### 5. **Observer Pattern**

Processing logger captures all state transitions:

```python
# Event-driven logging
logger.log_spec_extraction(user_text, extracted_spec)
logger.log_backend_selection(spec, selected_backend)
logger.log_layout_generation(backend, generated_layouts)
logger.log_compliance_check(layouts, passed_layouts)
```

---

## Module Design

### Layer 1: API/Presentation Layer

**File**: `api/server.py`

```python
class FastAPI_App:
    endpoints:
        - POST /api/chat → ConversationOrchestrator
        - POST /api/generate → GenerateRequest → execute_response()
        - POST /api/correct → CorrectionHandler
        - GET /api/explain → ExplainRankedDesigns
        - WebSocket /ws/session → WebSocket handler
```

**Responsibilities**:
- Request validation (Pydantic models)
- Session management
- Response serialization
- CORS and security

---

### Layer 2: NL Processing & Intent Classification

**Files**: `nl_interface/gemini_adapter.py`, `nl_interface/service.py`

#### Intent Types:
```
┌─────────────────────────┐
│   User Message          │
├─────────────────────────┤
│ Intent Classifier       │
├─────────────────────────┤
│ ↓                       │
│ DESIGN        → Extract Spec
│ CORRECTION    → Parse Deltas
│ QUESTION      → Answer from Context
│ CONVERSATION  → Chat Response
└─────────────────────────┘
```

#### Processing Flow:

```python
def process_user_request(text: str, current_spec: Dict) -> Dict:
    # 1. Extract CLI arguments and natural language parameters
    extracted = _extract_cli_args(text)
    
    # 2. Classify intent
    intent = classify_intent(text)
    
    # 3. Parse natural language spec
    nl_spec = parse_nl_spec(text, extracted)
    
    # 4. Merge with current spec
    merged_spec = merge_specs(current_spec, nl_spec)
    
    # 5. Validate against ontology
    validation = validate_spec(merged_spec)
    
    # 6. Route to appropriate backend
    backend = route_backend(merged_spec)
    
    return {
        "intent": intent,
        "nl_spec": nl_spec,
        "merged_spec": merged_spec,
        "backend_target": backend,
        "backend_ready": validation["valid"]
    }
```

---

### Layer 3: Spec Processing & Normalization

**Files**: `nl_interface/adapter.py`, `nl_interface/constraint_analyzer.py`

#### Spec Structure:

```python
CoreSpec = {
    "boundary": {
        "width": float,           # meters
        "height": float,          # meters
        "polygon": [[x,y], ...]   # optional arbitrary boundary
    },
    "entrance_point": [x, y],     # optional entrance location
    "rooms": [
        {
            "name": str,
            "type": str,          # from ontology
            "area": float,        # optional target area
            "min_area": float,    # from ontology defaults
            "max_area": float
        }
    ],
    "preferences": {
        "adjacency": [
            {
                "room1": str,
                "room2": str,
                "weight": float   # 0.0 to 2.0
            }
        ],
        "privacy": {
            "room_name": "public|private|service"
        }
    },
    "weights": {
        "composition": 0.2,
        "connectivity": 0.2,
        "adjacency": 0.3,
        "area": 0.15,
        "alignment": 0.15
    }
}
```

#### Backend Routing Logic:

```
┌─ Spec Analysis ─────────────────┐
│                                 │
│ Room Types Analysis             │
│ ├─ Only core types              │
│ │  (Bedroom, Kitchen, etc.)      │
│ │  → algorithmic/planner/hybrid  │
│ │                                │
│ ├─ Extended types               │
│ │  (Garage, Store, etc.)         │
│ │  → learned                     │
│ │                                │
│ └─ Mixed types                  │
│    → hybrid (best of all)       │
│                                 │
│ Environment Override             │
│ BLUEPRINT_BACKEND_MODE=explicit │
│ → Force backend selection       │
└─────────────────────────────────┘
```

---

### Layer 4: Generation Backends

#### 4A. Algorithmic Backend Pipeline

```
Spec Input
    ↓
[1] Polygon Packing
    - Recursive bisection algorithm
    - Exact area allocation
    - Output: Room polygons
    ↓
[2] Door Placement
    - Placement on room edges
    - Connectivity graph construction
    - Output: Building with doors
    ↓
[3] Corridor Planning (2 variants)
    - Hub-based (LivingRoom center)
    - Corridor-first (sequential)
    - Output: Building with corridors
    ↓
[4] Wall Generation
    - Extract wall segments
    - Merge shared walls
    - Apply thickness
    - Output: Wall network
    ↓
[5] Post-Processing
    - Polygon snapping to grid
    - Aspect ratio enforcement
    - Alignment optimization
    ↓
Output: Candidate Layout
```

**Algorithm Choice**: Recursive Bisection Polygon Packing

```python
def bisect_polygon(poly, ratio, axis='x'):
    """
    Slice polygon so first piece has ratio * area
    
    Benefits:
    - Guarantees exact area allocation
    - Works with ANY boundary shape
    - No gaps or overlaps
    - O(n log n) time complexity
    - Maintains spatial locality
    """
    # Binary search for exact slice coordinate
    # Clip polygon at coordinate
    # Return two pieces
```

#### 4B. Learned Backend Pipeline

```
Spec Input
    ↓
[1] Template Selection
    - Match spec to learned layout template
    - Template: pre-processed room sequence
    ↓
[2] Model Sampling (K times)
    - Load LayoutTransformer checkpoint
    - Forward pass with template input
    - Sampling with temperature control
    - Centroid collapse detection & jitter
    - Output: K layout candidates
    ↓
[3] Layout Adaptation
    - Convert model output to Building object
    - Extract room coordinates, polygons
    - Output: K Building objects
    ↓
[4] Repair Gate
    - Check hard constraints
    - Repair connectivity violations
    - Enforce minimum dimensions
    - Output: Valid Building objects (subset of K)
    ↓
[5] Preranking
    - Fast filtering by validity
    - Discard degenerate layouts
    - Output: Top-K valid layouts
    ↓
Output: Candidate Layouts
```

**Model Architecture**: LayoutTransformer

```
Encoder (Spec Input):
├─ Room embedding (room_id, area)
├─ Boundary embedding (width, height)
├─ Positional encoding
└─ Transformer encoder

Decoder (Layout Output):
├─ Autoregressive room coordinate prediction
├─ Attention to boundary and adjacent rooms
└─ Transformer decoder

Loss:
├─ Coordinate reconstruction loss
├─ Area preservation loss
├─ Adjacency preference loss
└─ Connectivity loss
```

#### 4C. Planner Backend

```
Spec Input
    ↓
[1] Room Planner LSTM
    - Predict room ordering
    - Predict adjacency graph
    - Learned from training data
    ↓
[2] Ordered Packing
    - Use predicted order for packing
    - Respects learned adjacencies
    ↓
[3] Standard Pipeline
    - Door placement
    - Corridor planning
    - Wall generation
    ↓
Output: Candidate Layout
```

---

### Layer 5: Constraint Processing

**Files**: `constraints/rule_engine.py`, `constraints/repair_loop.py`

#### Rule Engine Hierarchy:

```
RuleEngine (ontology/regulation_data.json)
    ├─ Occupancy Rules
    │  ├─ Residential (default)
    │  │  ├─ Room Type Rules
    │  │  │  ├─ Bedroom: min_area=9.5, max_area=16.0
    │  │  │  ├─ Kitchen: min_area=4.5, max_area=10.0
    │  │  │  └─ ...
    │  │  ├─ Corridor Rules
    │  │  │  ├─ min_width: 1.0m
    │  │  │  ├─ hub_connection: LivingRoom
    │  │  │  └─ essential_connections: [Bedroom, Bathroom, WC]
    │  │  ├─ Door Rules
    │  │  │  ├─ min_width: 0.9m
    │  │  │  └─ min_height: 2.2m
    │  │  ├─ Travel Rules
    │  │  │  └─ max_travel_distance: 22.5m
    │  │  └─ Occupancy Rules
    │  │     └─ occupant_load_per_100sqm: 8.0
    │  └─ Hostel (not yet implemented)
    │
    └─ Validation Methods:
        ├─ check_room_areas(building)
        ├─ check_room_dimensions(building)
        ├─ check_connectivity(building)
        ├─ check_travel_distance(building)
        ├─ check_occupancy_load(building)
        └─ check_egress_requirement(building)
```

#### Repair Loop:

```python
def validate_and_repair(building, regulation_file, max_iterations=10):
    """
    Iteratively repair constraint violations
    
    Strategy:
    1. Identify constraint violations
    2. Apply targeted repair (resize, reconnect, etc.)
    3. Re-validate
    4. Repeat until valid or max_iterations reached
    """
    
    for iteration in range(max_iterations):
        violations = engine.validate(building)
        if not violations:
            return building, "COMPLIANT"
        
        repaired = _repair_violations(building, violations, engine)
        if not repaired:
            return building, f"NON_COMPLIANT ({len(violations)} violations)"
        
        building = repaired
    
    return building, "PARTIALLY_COMPLIANT"
```

---

### Layer 6: Ranking & Scoring

**Files**: `generator/ranking.py`, `generator/composition_metrics.py`

#### Scoring Components:

```python
SCORE = w_area * area_score
       + w_alignment * alignment_score
       + w_adjacency * adjacency_score
       + w_connectivity * connectivity_score
       + w_composition * composition_score

where:
    area_score ∈ [0, 1]        # Room areas within targets
    alignment_score ∈ [0, 1]    # Rooms aligned to grid
    adjacency_score ∈ [0, 1]    # User preferences satisfied
    connectivity_score ∈ [0, 1] # Efficient circulation
    composition_score ∈ [0, 1]  # Visual balance
    
    Σ w_i = 1.0                # Normalized weights
```

#### Metrics Calculation:

```python
def calculate_metrics(building, spec, boundary):
    metrics = {
        # Geometric
        "alignment_score": alignment_score(building),
        "circulation_factor": circulation_area / total_area,
        "wall_efficiency": shared_walls / total_walls,
        
        # Functional
        "adjacency_satisfaction": check_adjacency_preferences(building, spec),
        "connectivity": evaluate_connectivity(building),
        "travel_distance": max_travel_distance(building),
        
        # Architectural
        "room_area_compliance": check_all_areas_in_range(building),
        "fully_connected": is_fully_connected(building),
        "corridor_compactness": 1.0 - (corridor_length / max_corridor_length),
        
        # Compliance
        "violations_count": len(violations),
        "compliance_status": "COMPLIANT" | "PARTIAL" | "NON_COMPLIANT",
    }
    return metrics
```

#### Selection Logic:

```python
def rank_layout_variants(candidates, spec):
    """
    Rank and select top-M variants
    
    Process:
    1. Score all candidates
    2. Sort by score (descending)
    3. Remove duplicates (identical layouts)
    4. Return top-M (default: 3)
    """
    
    scores = [score_variant(c, spec) for c in candidates]
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    # Deduplicate
    seen = set()
    unique = []
    for layout, score in ranked:
        sig = signature(layout)  # Geometric signature
        if sig not in seen:
            seen.add(sig)
            unique.append((layout, score))
    
    return unique[:TOP_M]
```

---

### Layer 7: Output Generation

**Files**: `visualization/export_svg_blueprint.py`, `constraints/compliance_report.py`

#### SVG Rendering Pipeline:

```
Building Object
    ↓
[1] SVG Canvas Setup
    - Calculate scale and dimensions
    - Create SVG root element
    - Set up viewBox
    ↓
[2] Wall Rendering
    - Extract and merge wall segments
    - Apply wall thickness
    - Draw outer walls (thick)
    - Draw inner walls (thin)
    ↓
[3] Room Rendering
    - Draw room polygons
    - Fill with zone colors
    - Add room labels with areas
    ↓
[4] Door Rendering
    - Draw door openings (gaps in walls)
    - Draw door swings (arcs)
    - Mark door types (entry, interior)
    ↓
[5] Window Rendering
    - Draw window locations (segments)
    - Draw window ticks
    - Mark window types
    ↓
[6] Corridor Rendering
    - Apply corridor fill (diagonal hatching)
    - Draw corridor paths
    ↓
[7] Dimensions & Labels
    - Draw room dimension strings
    - Draw boundary dimensions
    - Add room area labels
    ↓
[8] Legend & Title
    - Add scale bar
    - Add compass rose
    - Add title block
    - Add compliance status
    ↓
Output: SVG String
```

#### Compliance Report Structure:

```json
{
    "metadata": {
        "generation_date": "2024-03-27T10:30:00Z",
        "backend": "hybrid",
        "status": "COMPLIANT"
    },
    "specification": {
        "boundary": {"width": 12.0, "height": 15.0},
        "total_area": 180.0,
        "rooms": [...]
    },
    "geometry": {
        "total_area": 180.0,
        "built_area": 168.5,
        "circulation_area": 11.5,
        "circulation_factor": 0.064
    },
    "rooms": [
        {
            "name": "MasterBedroom",
            "type": "Bedroom",
            "area": 18.5,
            "area_status": "COMPLIANT",
            "min_area": 9.5,
            "max_area": 16.0,
            "dimensions": [5.2, 3.56]
        }
    ],
    "compliance": {
        "all_rooms_compliant": true,
        "fully_connected": true,
        "max_travel_distance": 18.2,
        "max_allowed_distance": 22.5,
        "travel_distance_status": "COMPLIANT",
        "violations": []
    },
    "metrics": {
        "design_score": 0.87,
        "adjacency_satisfaction": 0.92,
        "alignment_score": 0.78
    }
}
```

---

## Data Flow

### Request to Response Flow

```
                     ┌─────────────────┐
                     │  User Request   │
                     │  (Natural Lng)  │
                     └────────┬────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Intent Classifier  │
                    │ (Gemini Adapter)   │
                    └─────────┬──────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
         ┌──────▼───┐  ┌──────▼───┐  ┌────▼─────┐
         │  Design  │  │Correction│  │ Question │
         │Request   │  │Request   │  │ / Chat   │
         └──────┬───┘  └──────┬───┘  └────┬─────┘
                │             │           │
         ┌──────▼─────────────▼───────────▼────┐
         │  Spec Processing Pipeline          │
         ├────────────────────────────────────┤
         │ 1. Extract parameters             │
         │ 2. Normalize                       │
         │ 3. Enhance with constraints       │
         │ 4. Validate against ontology      │
         │ 5. Route to backend               │
         └──────┬──────────────────────────────┘
                │
         ┌──────▼─────────────────────────────┐
         │  Backend Selection & Execution     │
         ├────────────────────────────────────┤
         │ ┌─────────────────────────────┐   │
         │ │ Algorithmic Backend         │   │
         │ │ (Geometric algorithms)      │   │
         │ └─────────────────────────────┘   │
         │                 ↓                 │
         │ ┌─────────────────────────────┐   │
         │ │ Generate K candidates       │   │
         │ │ (Polygon packing, doors)    │   │
         │ └─────────────────────────────┘   │
         └──────┬──────────────────────────────┘
                │
         ┌──────▼─────────────────────────────┐
         │  Post-Processing                  │
         ├────────────────────────────────────┤
         │ 1. Corridor planning              │
         │ 2. Door/window refinement         │
         │ 3. Wall generation               │
         │ 4. Polygon alignment             │
         └──────┬──────────────────────────────┘
                │
         ┌──────▼─────────────────────────────┐
         │  Constraint Validation            │
         ├────────────────────────────────────┤
         │ 1. Hard constraints (Rule Engine) │
         │ 2. Repair violations (Repair Loop)│
         │ 3. Re-validate                    │
         └──────┬──────────────────────────────┘
                │
         ┌──────▼─────────────────────────────┐
         │  Ranking & Selection              │
         ├────────────────────────────────────┤
         │ 1. Calculate metrics for each     │
         │ 2. Score using weighted formula  │
         │ 3. Rank by score (descending)    │
         │ 4. Select top-M variants         │
         └──────┬──────────────────────────────┘
                │
         ┌──────▼─────────────────────────────┐
         │  Output Generation                │
         ├────────────────────────────────────┤
         │ 1. Render SVG floor plans        │
         │ 2. Generate compliance reports   │
         │ 3. Collect design statistics     │
         │ 4. Prepare explanations          │
         └──────┬──────────────────────────────┘
                │
         ┌──────▼─────────────────────────────┐
         │  Response Assembly                │
         ├────────────────────────────────────┤
         │ {                                 │
         │   status: "success",              │
         │   designs: [...],                 │
         │   artifacts: {...},               │
         │   metrics: {...},                 │
         │   explanation: {...}              │
         │ }                                 │
         └──────────────────────────────────┘
```

---

## Design Decisions

### 1. **Polygon Packing Algorithm Selection**

**Decision**: Use Continuous Recursive Bisection

**Rationale**:
- ✅ Guarantees exact area allocation
- ✅ Works with arbitrary boundary shapes
- ✅ No gaps or overlaps by construction
- ✅ Deterministic and efficient
- ✅ Spatially local (adjacent rooms in source tree stay adjacent)

**Alternative Considered**: Guillotine cuts, Maximal rectangle packing, Force-directed layout
- ❌ Guillotine: Wasteful (gaps), inefficient for complex boundaries
- ❌ Maximal rectangles: Complex, assumes rectangular rooms
- ❌ Force-directed: Non-deterministic, slow, hard to control

### 2. **Multi-Backend Strategy**

**Decision**: Implement multiple generation backends

**Rationale**:
- **Algorithmic**: Fast (100ms), deterministic, good for standard layouts, interpretable
- **Learned**: Humanlike (500ms), diverse, handles complex adjacencies, non-deterministic
- **Planner**: Structured reasoning, learned spatial relationships
- **Hybrid**: Best results by combining strengths

**Benefits**:
- User can choose quality vs. speed tradeoff
- Graceful degradation (fallback if one backend unavailable)
- A/B testing and comparison
- Continuous improvement (new backends without replacing old)

### 3. **Repair Loop Pattern**

**Decision**: Iterative constraint repair rather than generative rejection

**Rationale**:
- Generated layouts often violate soft constraints
- Rejection-based sampling wastes computation
- Iterative repair preserves design intent
- Transparent about violation types

**Algorithm**:
1. Identify violated constraints
2. Apply targeted repair (e.g., resize room if too small)
3. Re-validate
4. Repeat until valid or max iterations

### 4. **Ontology-Driven Validation**

**Decision**: Centralize building codes in JSON/OWL ontology

**Files**:
- `ontology/regulation_data.json` - Building codes per room/occupancy
- `ontology/building_spec.schema.json` - JSON schema for specs
- `ontology/regulatory.owl` - OWL ontology (future semantic reasoning)

**Benefits**:
- Easy to update regulations without code changes
- Supports multiple jurisdictions
- Enables constraint-aware generation
- Facilitates compliance reporting

### 5. **Weighted Ranking Function**

**Decision**: Combine multiple metrics with configurable weights

```python
SCORE = Σ(w_i × metric_i)   where Σ w_i = 1.0
```

**Rationale**:
- Different users prioritize different aspects
- Weights allow personalization
- Transparent and interpretable scores
- Easy to adjust by tuning weights

**Metric Categories**:
1. **Geometric**: Alignment, aspect ratio, wall efficiency
2. **Functional**: Connectivity, travel distance, adjacency
3. **Architectural**: Natural light, privacy, structural feasibility

### 6. **Layered API Design**

**Decision**: Separate layers for API, NL processing, generation, output

**Rationale**:
- Clear separation of concerns
- Easy to test each layer independently
- Easy to replace or extend components
- Supports multiple input/output formats

**Layers**:
1. API (FastAPI)
2. NL Interface (Gemini)
3. Spec Processing (Normalization, validation)
4. Generation (Backend engines)
5. Post-Processing (Corridors, walls)
6. Constraint Checking (Rule engine, repair)
7. Ranking (Scoring and selection)
8. Output (SVG, reports)

---

## Integration Points

### External Integrations

#### 1. **Gemini API** (NL Understanding)
- **File**: `nl_interface/gemini_adapter.py`
- **Purpose**: Intent classification, spec extraction, explanation generation
- **Fallback**: Empty responses if API unavailable

#### 2. **Building Regulations Database**
- **File**: `ontology/regulation_data.json`
- **Purpose**: Provides building code constraints
- **Update Process**: Manual JSON updates (future: API integration)

#### 3. **Trained Model Checkpoints**
- **Algorithmic**: Deterministic (no checkpoint needed)
- **Learned**: `learned/model/checkpoints/improved_v1.pt` (LayoutTransformer)
- **Planner**: `learned/planner/checkpoints/room_planner.pt` (LSTM planner)
- **Fallback**: Switch to algorithmic if models unavailable

### Internal Integration Points

#### 1. **Spec → Backend Routing**
- Input: Normalized spec with room types
- Decision: Route to algorithmic/learned/planner/hybrid
- Output: Backend target identifier

#### 2. **Generation → Validation**
- Input: Generated layouts
- Check: Hard constraints (connectivity, areas)
- Output: Valid/invalid status + violations list

#### 3. **Validation → Repair**
- Input: Invalid layout + violation list
- Repair: Apply constraint-specific fixes
- Output: Repaired layout (may still have violations)

#### 4. **Layouts → Ranking**
- Input: Multiple candidate layouts
- Score: Multi-metric ranking function
- Output: Ranked and deduplicated list (top-M)

#### 5. **Ranking → Output**
- Input: Top-ranked layouts
- Render: SVG visualization + compliance reports
- Output: Artifacts (files) + metadata (JSON)

---

## Scalability Considerations

### Current Limits

| Dimension | Limit | Reason |
|-----------|-------|--------|
| Room Count | 4-12 | Training data, graph algorithms |
| Plot Area | 80-500 m² | Typical residential |
| Aspect Ratio | 1:1 to 3:1 | Architectural norms |
| Boundary Complexity | Low-complexity polygons | Algorithm assumptions |

### Scaling Strategies

#### 1. **Vertical Scaling** (More rooms)
- **Approach**: Multi-level/hierarchical generation
  1. Generate zones (public/private/service)
  2. Generate layouts within each zone
  3. Merge and optimize zone connections
- **Benefit**: Handle 20-50 room programs

#### 2. **Horizontal Scaling** (Faster generation)
- **Approach**: Parallel backend execution
  ```python
  # Run all backends in parallel, return best
  tasks = [
      run_algorithmic(spec),
      run_learned(spec),
      run_planner(spec),
  ]
  results = await asyncio.gather(*tasks)
  best = rank_variants(results)
  ```
- **Benefit**: Response time independent of backend count

#### 3. **Model Scaling** (Better quality)
- **Approach**: Ensemble learning
  ```python
  # Train multiple models with different initializations
  ensemble = [model_v1, model_v2, model_v3]
  samples = [m.sample(spec) for m in ensemble]
  # Combine and rerank
  ```
- **Benefit**: Higher quality, more diverse outputs

#### 4. **Data Scaling** (Broader applicability)
- **Approach**: Transfer learning and fine-tuning
  - Base model: Trained on diverse layouts
  - Fine-tune: Per-region building codes
- **Benefit**: Adaptable to different jurisdictions

---

## Performance Optimization

### Algorithm Optimization

| Operation | Optimization | Speedup |
|-----------|--------------|---------|
| Polygon bisection | Binary search (35 iterations) | O(35) vs O(n²) |
| Connectivity check | Graph DFS (memoized) | O(n+e) |
| Wall merging | Spatial indexing | O(log n) vs O(n²) |
| Scoring | Cached metrics | Amortized O(1) |

### Caching Strategies

1. **Model Caching**: Load once, reuse
2. **Regulation Caching**: Load ontology once at startup
3. **Spec Caching**: Cache parsed specs for repeated queries
4. **Metrics Caching**: Cache room metrics across variants

### Async/Parallel Execution

```python
# Parallel backend execution
async def execute_parallel(spec):
    tasks = [
        run_backend("algorithmic", spec),
        run_backend("learned", spec),
        run_backend("planner", spec),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid_results = [r for r in results if not isinstance(r, Exception)]
    return rank_variants(valid_results)
```

---

## Error Handling & Degradation

### Graceful Degradation

```
User Request
    ↓
Try: Run preferred backend
    ├─ Success → Return result
    └─ Failure → Try fallback
        ├─ Success → Return result
        └─ Failure → Try emergency fallback
            ├─ Success → Return result
            └─ Failure → Return error + cached result

Cascade:
Preferred → Fallback1 → Fallback2 → Error
hybrid   → algorithmic → geometric → error
learned  → algorithmic → geometric → error
planner  → algorithmic → geometric → error
```

### Error Types & Recovery

| Error | Type | Recovery |
|-------|------|----------|
| Invalid spec | User Error | Validation error message |
| Constraint violation | Generation | Repair loop |
| Model unavailable | System | Switch backend |
| Gemini API down | External | Use cached spec |
| Memory exhausted | Resource | Reduce K (candidates) |
| Timeout | Performance | Return partial results |

---

## Testing Strategy

### Unit Tests
- Module-level testing (one component)
- Fast (< 100ms each)
- Examples: polygon bisection, connectivity check, score calculation

### Integration Tests
- Pipeline-level testing (multiple components)
- Medium speed (< 1s each)
- Examples: spec → generation → ranking, API endpoints

### End-to-End Tests
- Full user workflow testing
- Slower (2-5s each)
- Examples: natural language → floor plans, correction → refined plans

### Performance Tests
- Benchmark critical paths
- Monitor metrics: response time, memory usage, parallelization efficiency

---

## Deployment Architecture

### Development
```
Frontend (React dev server: localhost:3000)
         ↓ HTTP
API Server (FastAPI dev: localhost:8000)
         ↓
Backend Engines (In-process)
```

### Production
```
CDN: Static assets (index.html, CSS, JS)
     ↓
     Load Balancer
     ↓
[API Server 1] [API Server 2] [API Server 3]
     ↓ (shared cache)
     Redis: Session cache, spec cache
     ↓
     Model Server: Loaded models (GPU)
     ↓
     Database: Session history, artifact logs
```

---

## Future Architecture Improvements

1. **Event-Driven Architecture**
   - Kafka/RabbitMQ for async task processing
   - Decouples generation from response
   - Supports long-running generation jobs

2. **Microservices**
   - Separate services for each backend
   - Separate service for constraint engine
   - Separate service for visualization

3. **GraphQL API**
   - More flexible querying
   - Reduced over-fetching
   - Better for complex data

4. **Knowledge Graph**
   - Semantic OWL reasoning
   - Constraint reasoning (SMT solver)
   - Design pattern matching

5. **3D Integration**
   - 3D view generation
   - Structural analysis
   - MEP routing

---

**Version**: 1.0  
**Last Updated**: March 2026  
**Author**: BlueprintGPT Architecture Team
