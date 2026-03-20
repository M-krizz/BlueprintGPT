# Chapter-4 Ground Truth Schema

Single source of truth for NBC India 2016 Chapter 4 bye-law compliance.

**Canonical file:** `ontology/regulation_data.json`

## Schema Overview

```
regulation_data.json
├── chapter4_occupant_load         # Table 4.1 - Occupant loads per 100 sq.m
├── chapter4_egress                # Section 4.8 - Travel, corridors, stairs, exits
├── chapter4_lighting_ventilation  # Section 4.9 - Openings, lighting depth, shafts
├── chapter4_wet_area              # Construction rules (floor, wall, drain)
├── chapter4_open_space            # Interior/exterior open space mins
└── {Occupancy}                    # Per-occupancy rules
    ├── occupant_load_per_100sqm
    ├── max_travel_distance
    ├── circulation_factor
    ├── rooms                      # Room minimums
    ├── corridor
    ├── door
    ├── exit
    ├── stair
    └── chapter4_residential       # Plot bucket rules (Residential only)
        └── plot_buckets
            ├── upto_50sqm
            └── above_50sqm
```

---

## 1. Occupant Load Table (Table 4.1)

**Path:** `chapter4_occupant_load`

| Occupancy | Occupants per 100 sq.m |
|-----------|------------------------|
| Residential | 8.0 |
| Educational | 25.0 |
| Institutional | 6.6 |
| Assembly (with fixed seats/dance) | 166.6 |
| Assembly (without seating/dining) | 66.6 |
| Mercantile (street/sales basement) | 33.3 |
| Mercantile (upper sales) | 16.6 |
| Business | 10.0 |
| Industrial | 10.0 |
| Storage | 3.3 |
| Hazardous | 10.0 |

**Usage:**
```python
from constraints.chapter4_helpers import load_regulation_data
reg = load_regulation_data()
per_100 = reg["chapter4_occupant_load"]["Residential"]  # 8.0
```

---

## 2. Residential Min Sizes (Table 4.2)

**Path:** `Residential.chapter4_residential.plot_buckets`

Plot bucket selection based on plot area:
- `upto_50sqm`: Plot area <= 50 sq.m
- `above_50sqm`: Plot area > 50 sq.m

### Plot <= 50 sq.m

| Category | min_area | min_width | min_height |
|----------|----------|-----------|------------|
| Habitable | 7.5 | 2.1 | 2.75 |
| Kitchen | 3.3 | 1.5 | 2.75 |
| Bathroom | 1.2 | 1.0 | 2.2 |
| WC | 1.0 | 0.9 | 2.2 |
| BathWC | 1.8 | 1.0 | 2.2 |

Doors: habitable 0.8m, service 0.75m, height 2.0m
Stair width: 0.75m

### Plot > 50 sq.m

| Category | min_area | min_width | min_height |
|----------|----------|-----------|------------|
| Habitable | 9.5 | 2.4 | 2.75 |
| Kitchen | 4.5 | 1.5 | 2.75 |
| Pantry | 3.0 | 1.4 | 2.75 |
| Bathroom | 1.8 | 1.2 | 2.2 |
| WC | 1.1 | 0.9 | 2.2 |
| BathWC | 2.8 | 1.2 | 2.2 |
| Garage | 14.85 | 2.75 (min_length 5.4) | 2.4 |

Doors: habitable 0.9m/2.2m, service 0.75m/2.0m
Stair width: 0.9m, passage width: 1.0m

**Usage:**
```python
from constraints.chapter4_helpers import get_min_room_dims, plot_bucket

bucket = plot_bucket(plot_area_sqm=45.0)  # "upto_50sqm"
dims = get_min_room_dims("Bedroom", plot_area_sqm=45.0)
# {'min_area': 7.5, 'min_width': 2.1, 'min_height': 2.75}
```

---

## 3. Egress (Section 4.8)

**Path:** `chapter4_egress`

### 3.1 Travel Distance Limits (Section 4.8.4)

| Occupancy | Max Distance (m) |
|-----------|------------------|
| Residential | 22.5 |
| Educational | 22.5 |
| Institutional | 22.5 |
| Hazardous | 22.5 |
| Assembly | 30.0 |
| Business | 30.0 |
| Mercantile | 30.0 |
| Industrial | 30.0 |
| Storage | 30.0 |

### 3.2 Corridor Min Width (Section 4.8.7)

| Type | Min Width (m) |
|------|---------------|
| residential_dwelling_unit | 1.0 |
| residential_hostel | 1.25 |
| assembly | 2.0 |
| hotel | 1.5 |
| hospital | 2.4 |
| others | 1.5 |

### 3.3 Stair Min Width (Section 4.8.6)

| Type | Min Width (m) |
|------|---------------|
| residential_low_rise | 0.9 |
| residential_other | 1.25 |
| assembly | 2.0 |
| institutional | 2.0 |
| educational | 1.5 |
| others | 1.5 |

### 3.4 Exit Doors

- min_width: 1.0m (hospital: 1.5m)
- min_height: 2.0m
- must_open_outward: true
- no_sliding_overhead_revolving: true
- landing_required: true

### 3.5 Internal Doors (from plot bucket)

| Plot Size | Habitable | Service |
|-----------|-----------|---------|
| <= 50 sq.m | 0.80m x 2.0m | 0.75m x 2.0m |
| > 50 sq.m | 0.90m x 2.2m | 0.75m x 2.0m |

### 3.6 Exit Capacity (Table 4.3)

Persons per 50cm unit width:

| Occupancy | Stair | Ramp | Door |
|-----------|-------|------|------|
| Residential | 25 | 50 | 75 |
| Educational | 25 | 50 | 75 |
| Institutional | 25 | 50 | 75 |
| Assembly | 40 | 50 | 75 |
| Business | 50 | 50 | 75 |
| Mercantile | 50 | 50 | 75 |
| Industrial | 50 | 50 | 75 |
| Storage | 50 | 50 | 75 |
| Hazardous | 25 | 50 | 75 |

**Usage:**
```python
from constraints.chapter4_helpers import (
    get_travel_distance_limit,
    get_corridor_min_width,
    get_stair_min_width_by_occupancy,
    get_exit_capacity,
    get_exit_door_dims,
    get_door_dims,
)

travel = get_travel_distance_limit("Residential")  # 22.5
corridor = get_corridor_min_width("Residential", "dwelling_unit")  # 1.0
stair = get_stair_min_width_by_occupancy("Residential", "low_rise")  # 0.9
capacity = get_exit_capacity("Residential", "stair")  # 25
exit_door = get_exit_door_dims("Residential")  # {'min_width': 1.0, 'min_height': 2.0}
int_door = get_door_dims("Bedroom", plot_area_sqm=60.0)  # {'min_width': 0.9, 'min_height': 2.2}
```

---

## 4. Lighting & Ventilation (Section 4.9)

**Path:** `chapter4_lighting_ventilation`

| Rule | Value |
|------|-------|
| Opening area ratio (habitable + kitchen) | >= 1/10 floor area |
| Max lighting depth | 7.5m from opening |
| Kitchen window min | 1.0 sq.m |
| Bathroom/WC vent opening | >= 0.37 sq.m (or shaft) |

### Ventilation Shaft Min Sizes (Table 4.5)

| Building Height | Min Area | Min Dimension |
|-----------------|----------|---------------|
| <= 9m | 1.5 sq.m | 1.0m |
| <= 12.5m | 3.0 sq.m | 1.2m |
| <= 15m | 4.0 sq.m | 1.5m |
| > 15m | Mechanical ventilation required |

**Usage:**
```python
from constraints.chapter4_helpers import (
    get_opening_ratio,
    get_max_lighting_depth,
    get_kitchen_window_min,
    get_bathroom_vent_min,
    get_shaft_min,
)

ratio = get_opening_ratio()  # 0.10
depth = get_max_lighting_depth()  # 7.5
kitchen_win = get_kitchen_window_min()  # 1.0
bath_vent = get_bathroom_vent_min()  # 0.37
shaft = get_shaft_min(building_height_m=10.0)  # {'min_area_sqm': 3.0, 'min_dimension_m': 1.2}
```

---

## 5. Wet Area Construction

**Path:** `chapter4_wet_area`

| Rule | Requirement |
|------|-------------|
| Floor | Impervious, slope to drain |
| Wall finish | Impervious up to 1.0m |
| WC exclusivity | Not used for any other purpose |
| Sewage | Municipal sewer or septic tank |

(Schematic enforcement only - recorded in compliance report)

---

## 6. Open Space

**Path:** `chapter4_open_space`

| Rule | Value |
|------|-------|
| Habitable rooms abut | Open space or verandah |
| Interior open space min | 3.0m (when height <= 12.5m) |

---

## Consuming Components

| Component | File | Uses |
|-----------|------|------|
| Rule Engine | `constraints/rule_engine.py` | Full Chapter-4 via helpers |
| Chapter-4 Helpers | `constraints/chapter4_helpers.py` | Canonical loader |
| Spec Validator | `constraints/spec_validator.py` | Room type validation |
| Repair Gate | `learned/integration/repair_gate.py` | Min enforcement, travel |
| NL Runner | `nl_interface/runner.py` | Design gate thresholds |
| Ranking | `generator/ranking.py` | Travel distance scoring |
| Explain | `explain/context_builder.py` | Evidence fields |

---

## Adding New Rules

1. Add data to `ontology/regulation_data.json`
2. Add lookup function in `constraints/chapter4_helpers.py`
3. Wire into consuming components
4. Add tests in `tests/unit/test_chapter4_rules.py`

---

## Note: IBC Ground Truth

The `ground_truth/` directory contains a **separate** IBC 2018 (International Building Code) rule system focused on:
- Building-level occupancy classification (A-U groups)
- Construction types (I-V) and fire ratings
- Height/area limitations

This is a different code system from Chapter-4 NBC India. Use the `ground_truth/validator.py` for IBC-style validation if needed, but Chapter-4 remains the primary source for residential floor plan compliance.
