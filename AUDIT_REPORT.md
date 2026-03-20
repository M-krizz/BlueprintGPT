# AUDIT_REPORT.md — Chapter-4 Ground Truth Unification

**Date:** 2026-03-20
**Status:** ALL 6 STEPS COMPLETE ✅

---

## Executive Summary

Successfully unified Chapter-4 bye-law enforcement across all BlueprintGPT pipelines:
- **Single canonical source**: `ontology/regulation_data.json`
- **9+ components verified** to use consistent helpers
- **All tests passing**: 30/30 Chapter-4 unit tests
- **New capabilities**: Realism scoring, MinDim processor, ground_truth validation bridge

---

## STEP 1: Unify Chapter-4 Ground Truth ✅

### Changed Files

| File | Change |
|------|--------|
| `constraints/chapter4_helpers.py` | Added `get_occupant_load_per_100sqm`, `get_wet_area_rules`, `get_open_space_rules`, `get_chapter4_summary` |
| `ground_truth/chapter4_validator.py` | **NEW** - Floor plan compliance validator bridge (420 lines) |
| `docs/CHAPTER4_SCHEMA.md` | **NEW** - Comprehensive schema documentation |
| `AUDIT_REPORT.md` | Updated with complete audit results |

### Canonical Chapter-4 Schema

**Location:** `ontology/regulation_data.json`

```
chapter4_occupant_load         → Table 4.1 (per 100 sq.m)
chapter4_egress                → Sec 4.8 (travel, corridors, stairs, exits)
  ├── travel_distance          → 22.5m (Residential), 30.0m (Assembly)
  ├── corridor_min_width       → 1.0m (dwelling unit), 1.25m (hostel)
  ├── stair_min_width          → 0.9m (low rise), 1.25m (other)
  └── exit_capacity            → Table 4.3 (stair/ramp/door capacity)
chapter4_lighting_ventilation  → Sec 4.9 (openings, depth, shafts)
  ├── opening_area_ratio       → 0.10 (10% floor area)
  ├── kitchen_window           → 1.0 sq.m min
  └── bathroom_vent            → 0.37 sq.m min
Residential.chapter4_residential.plot_buckets → Table 4.2
  ├── upto_50sqm              → Habitable 7.5/2.1m, Kitchen 3.3/1.5m
  └── above_50sqm             → Habitable 9.5/2.4m, Kitchen 4.5/1.5m
```

### Components Verified

All confirmed to use `ontology/regulation_data.json`:

✅ `constraints/rule_engine.py` — via chapter4_helpers
✅ `constraints/chapter4_helpers.py` — canonical loader
✅ `constraints/spec_validator.py` — room type validation
✅ `learned/integration/repair_gate.py` — min enforcement, travel
✅ `learned/integration/model_generation_loop.py` — generation
✅ `learned/evaluate_generation.py` — evaluation metrics
✅ `nl_interface/runner.py` — design gate thresholds
✅ `generator/layout_generator.py` — algorithmic generation
✅ `gui/layout_editor.py` — GUI validation

---

## STEP 2: Audit and Reproduce ✅

### NL CLI Tests

✅ **Test (a)**: Complete prompt with boundary
```
Input: '3 bedrooms, 2 bathrooms, 1 kitchen, 1 living room on 10 marla with north entrance'
Resolution: boundary_size=(15, 12)
Result: backend_ready=True, missing_fields=[], 4 rooms extracted
```

✅ **Test (b)**: Missing boundary prompt
```
Input: '2 bedrooms and 1 bathroom'
Result: backend_ready=False, missing_fields=['plot_type', 'entrance_side', 'boundary_polygon']
```

### Smoke Tests

✅ **Algorithmic Smoke**: COMPLIANT
```
Variant: balanced
Status: COMPLIANT
Connectivity: True
Travel: 13.68 / 22.5 m
```

✅ **Learned Smoke**: COMPLIANT
```
Variant: learned-generation
Status: COMPLIANT
Connectivity: True
Travel: 18.18 / 22.5 m
```

---

## STEP 3: Fix Discrepancies ✅

### Fixes Applied

| File | Issue | Fix |
|------|-------|-----|
| `learned/integration/repair_gate.py:496` | Direct `engine.data[occ]` access | → `engine.get_corridor_min_width(occ)` |
| `learned/evaluate_generation.py:444` | Direct `engine.data[occ]` access | → `engine.get_corridor_min_width(occ)` |
| `learned/integration/validate_and_repair.py:295` | Direct `engine.data[occ]` access | → `engine.get_corridor_min_width(occ)` |

### Before/After

**Before**: Hardcoded fallback `1.2m` corridor width
**After**: Proper Chapter-4 Section 4.8.7 lookup → `1.0m` for Residential dwelling units

---

## STEP 4: Improve Transformer Realism ✅

### New Components

| Component | Purpose | Lines |
|-----------|---------|-------|
| `learned/model/sample.py::MinDimProcessor` | Enforce Chapter-4 mins during sampling | 150 |
| `learned/integration/realism_score.py` | Pre-rank realism scoring | 280 |

### MinDimProcessor

Feature-flagged logit processor that masks coordinate bins violating Chapter-4 minimums:
- **x2 position (tail_len==4)**: Mask bins where width < min_width
- **y2 position (tail_len==5)**: Mask bins where height < min_height OR area < min_area
- **Plot bucket aware**: Uses `upto_50sqm` vs `above_50sqm` minimums

```bash
# Enable via environment variable
export LEARNED_MINDIM_PROCESSOR_ENABLED=true
```

### Realism Scoring

Pre-rank scoring components (weighted 100%):
- **Min-dim violations** (40%): Count of rooms below Chapter-4 minimums
- **Aspect ratio** (25%): Penalty for slivers (AR > 3.0)
- **Travel feasibility** (20%): Estimated travel distance
- **Corridor continuity** (10%): Connected topology score
- **Zoning** (5%): Proper room placement (stub)

**Example Output**:
```
Overall: 0.5463
Min-dim violations: 4
Min-dim score: 0.522
Aspect ratio score: 1.0
Corridor score: 0.5
Travel score: 0.0
```

---

## STEP 5: Wire Ground Truth Validator ✅

### Integration Points

| Component | Integration |
|-----------|-------------|
| `constraints/compliance_report.py` | Added `run_ground_truth_validation()` helper |
| `constraints/compliance_report.py:115` | Optional `ground_truth_validation` section in report |

### Usage

```python
from constraints.compliance_report import run_ground_truth_validation

# Optional strong validation (gracefully fails if not available)
result = run_ground_truth_validation(
    building,
    plot_area_sqm=75.0,
    corridor_width=1.2,
    travel_distance=20.0,
)

if result:
    print(f"Compliant: {result['chapter4_compliant']}")
    print(f"Violations: {len(result['chapter4_violations'])}")
```

**Test Result**:
```
Ground truth validation result:
  Compliant: False
  Plot bucket: upto_50sqm
  Violations: 2
    - [MAJOR] Kitchen1: area 3.00 sq.m < min 3.3 sq.m
    - [MAJOR] Corridor width 0.90m < min 1.0m
```

---

## STEP 6: Final Verification ✅

### Test Results

✅ **Chapter-4 Unit Tests**: 26/26 PASSED (0.50s)
✅ **Rule Engine Tests**: 4/4 PASSED (0.23s)
✅ **Total**: 30/30 tests passing

```
tests/unit/test_chapter4_rules.py::test_plot_bucket_small PASSED
tests/unit/test_chapter4_rules.py::test_room_dims_small_plot PASSED
tests/unit/test_chapter4_rules.py::test_travel_distance_residential PASSED
tests/unit/test_chapter4_rules.py::test_corridor_width_residential PASSED
tests/unit/test_chapter4_rules.py::test_check_chapter4_compliance_pass PASSED
... (26 total)
tests/unit/test_rule_engine.py::test_apply_room_rules_minimums PASSED
... (4 total)
```

### Pre-existing Issues (Not from this work)

⚠️ `test_nl_interface.py::test_validation_rejects_unsupported_room_mentions` — Expects `LivingRoom` invalid but it IS valid in schema
⚠️ `test_overlap_processor.py::TestLastRoomIdx::test_no_room_token_returns_minus_one` — Test uses `[1,2,3,4]` containing ROOM_TOKEN=3

---

## Artifact Summary

| Artifact | Path | Description |
|----------|------|-------------|
| **Canonical Data** | `ontology/regulation_data.json` | Single source of truth |
| **Schema Docs** | `docs/CHAPTER4_SCHEMA.md` | Comprehensive documentation |
| **Helpers** | `constraints/chapter4_helpers.py` | 20+ lookup functions |
| **Validator Bridge** | `ground_truth/chapter4_validator.py` | Floor plan validator |
| **Realism Scoring** | `learned/integration/realism_score.py` | Pre-rank scoring |
| **MinDim Processor** | `learned/model/sample.py:415-560` | Constrained sampling |
| **Test File** | `tests/unit/test_chapter4_rules.py` | 26 unit tests |

---

## Before/After Metrics

| Metric | Before | After |
|--------|--------|-------|
| Chapter-4 files | 2 (inconsistent) | 1 (canonical) |
| Direct data access | 3 locations | 0 (all via helpers) |
| Corridor width default | 1.2m (hardcoded) | 1.0m (Chapter-4 Sec 4.8.7) |
| Realism scoring | None | 5-component weighted score |
| Ground truth validation | Separate IBC only | Integrated Chapter-4 + IBC |
| Chapter-4 tests | 26 | 26 (all passing) |

---

## Compliance System Architecture

The codebase now has **TWO complementary compliance systems**:

| System | File | Scope | Code Base |
|--------|------|-------|-----------|
| **Chapter-4 (Primary)** | `ontology/regulation_data.json` | Floor plan level | NBC India 2016 |
| | | - Room sizes, corridors, doors | |
| | | - Travel distance, egress | |
| | | - Lighting/ventilation | |
| **IBC Ground Truth** | `ground_truth/ground_truth.yml` | Building level | IBC 2018 |
| | | - Occupancy classification (A-U) | |
| | | - Construction types (I-V) | |
| | | - Height/area limitations | |

---

## Remaining Limitations

1. **MinDimProcessor experimental**: Normalization heuristic (15m boundary) may not match training data; feature-flagged OFF by default
2. **Zoning score stub**: Not yet implemented (placeholder returns 0.75)
3. **Pre-existing test failures**: 2 tests unrelated to Chapter-4 changes

---

## Recommendations

1. ✅ **Use canonical source**: All new Chapter-4 rules → `ontology/regulation_data.json`
2. ✅ **Use helper functions**: Never access `engine.data` directly; use `chapter4_helpers`
3. ⚠️ **MinDimProcessor**: Enable experimentally with `LEARNED_MINDIM_PROCESSOR_ENABLED=true`
4. ✅ **Realism scoring**: Use in pre-ranking to identify high-quality generations
5. ✅ **Ground truth validation**: Enable optional strong checking in compliance reports

---

## Verification Commands

```bash
# Run Chapter-4 tests
python -m pytest tests/unit/test_chapter4_rules.py -v

# Verify canonical data
python -c "from constraints.chapter4_helpers import get_travel_distance_limit; print(get_travel_distance_limit('Residential'))"
# Expected: 22.5

# Run smoke tests
python -m demo.run_smoke_algorithmic --output-svg outputs/smoke.svg --output-report outputs/smoke.json
python -m learned.integration.run_smoke_learned --output-svg outputs/learned.svg --output-report outputs/learned.json

# Test realism scoring
python -c "from learned.integration.realism_score import compute_realism_score; ..."

# Test ground truth validation
python -c "from constraints.compliance_report import run_ground_truth_validation; ..."
```

---

**Status**: ✅ ALL 6 STEPS COMPLETE — Chapter-4 ground truth unified and enforced uniformly across all pipelines.

