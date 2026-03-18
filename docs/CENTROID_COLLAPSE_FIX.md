# Phase 1 — Centroid Jitter & Overlap Filtering

Complete reference for the Phase 1 short-term improvements to the
learned-layout pipeline (`centroid_utils.py` + `model_generation_loop.py`).

---

## What was built

| Feature | Status | File |
|---------|--------|------|
| Centroid collapse detection | ✅ | `centroid_utils.py` |
| Fixed-sigma jitter | ✅ | `centroid_utils.py` |
| Adaptive jitter (severity-scaled σ) | ✅ | `centroid_utils.py` |
| Directional jitter (boundary-aware bias) | ✅ | `centroid_utils.py` |
| Collapse-severity scoring (0–1) | ✅ | `centroid_utils.py` |
| Early overlap filter + resample | ✅ | `centroid_utils.py` |
| Generation-loop integration | ✅ | `model_generation_loop.py` |
| Extended diagnostics in summary dict | ✅ | `model_generation_loop.py` |
| Unit tests (55 total) | ✅ | `tests/learned/` |

---

## Environment variables (all feature-flagged)

```bash
# ── Jitter core ──────────────────────────────────────────────────
LEARNED_JITTER_ENABLED=true          # master switch (default: true)
LEARNED_JITTER_SIGMA=0.01            # base noise std-dev in [0,1] space

# ── Adaptive sigma ───────────────────────────────────────────────
ADAPTIVE_JITTER_ENABLED=true         # scale sigma by collapse severity
MAX_ADAPTIVE_JITTER_MULTIPLIER=3.0   # sigma × multiplier at severity=1

# ── Directional bias ─────────────────────────────────────────────
DIRECTIONAL_JITTER_ENABLED=true      # boundary-aware inward nudge
BOUNDARY_MARGIN=0.12                 # normalised edge distance that activates bias

# ── Collapse detection thresholds ────────────────────────────────
COLLAPSE_MIN_ROOMS=3
COLLAPSE_MEDIAN_DIST_THRESHOLD=0.02
COLLAPSE_IOU_PAIR_RATIO_THRESHOLD=0.30
IOU_BAD_THRESH=0.5

# ── Early overlap filter ─────────────────────────────────────────
LEARNED_OVERLAP_FILTER_ENABLED=true
OVERLAP_DROP_FRAC=0.4
MAX_RESAMPLE_ON_OVERLAP=2
```

---

## Feature details

### 1. Centroid collapse detection

`detect_centroid_collapse(raw_rooms)` inspects every raw-decoded sample and
returns `(is_collapsed: bool, metrics: dict)`.

Detection triggers when **either** condition holds:

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Median pairwise centroid distance | < 0.02 | All centroids almost identical |
| Fraction of pairs with IoU > 0.5 | > 0.30 | Too many rooms stacked on top of each other |

`metrics` keys:

| Key | Description |
|-----|-------------|
| `median_centroid_distance` | Median of all pairwise centroid distances |
| `pairwise_iou_fraction` | Fraction of pairs with IoU > `IOU_BAD_THRESH` |
| `collapse_severity` | Composite score **0–1** (0 = fine, 1 = total collapse) |

**Severity formula:**

```
severity_dist = max(0, 1 - median_dist / threshold_dist)
severity_iou  = max(0, (iou_frac - threshold_iou) / (1 - threshold_iou))
collapse_severity = min(1, max(severity_dist, severity_iou))
```

---

### 2. Fixed + Adaptive jitter

`jitter_centroids(raw_rooms, sigma, adaptive, collapse_severity, ...)`

When adaptive is **off** (`ADAPTIVE_JITTER_ENABLED=false`):

```
effective_sigma = sigma   (always 0.01 by default)
```

When adaptive is **on** (default):

```
multiplier = 1 + severity × (MAX_MULTIPLIER - 1)
effective_sigma = sigma × multiplier
```

| severity | multiplier (MAX=3) | effective σ (base=0.01) |
|----------|--------------------|-------------------------|
| 0.0 | 1.0 | 0.010 — minimal, preserves model priors |
| 0.5 | 2.0 | 0.020 — moderate collapse |
| 1.0 | 3.0 | 0.030 — total collapse, strong jitter |

**Why this matters:** a mild spatial spread needs only a tiny nudge; a fully
collapsed sample (all rooms at the same point) needs a much larger perturbation
to give the PolygonPacker a usable ordering signal.

---

### 3. Directional (boundary-aware) jitter

`compute_boundary_bias(cx, cy, margin, sigma)` returns a `(bias_x, bias_y)`
correction that is applied *before* the Gaussian noise.

```
if cx < margin:
    bias_x = +((margin - cx) / margin) × sigma   # push right
if cx > 1 - margin:
    bias_x = -((cx - (1-margin)) / margin) × sigma  # push left
# same logic for cy / top / bottom edges
```

Properties:
- **Zero** at centre (cx=0.5, cy=0.5)
- **Scales linearly** from 0 at `margin` to `sigma` at the edge
- **Capped at σ** — never overwhelms the Gaussian noise
- **Additive** — combined effect is `centroid + bias + N(0, σ)`

**Why this matters:** without bias, centroids near `x=0.02` are still pushed
outward ~50% of the time by random noise, then clamped hard against the wall.
With bias, they consistently drift inward into safe packer space.

---

### 4. Early overlap filtering

Before adapting any sample to a building object:

```python
overlap_frac = pairwise_iou_fraction(decoded_rooms, threshold=IOU_BAD_THRESH)
if overlap_frac > OVERLAP_DROP_FRAC:
    # skip this sample; optionally resample
```

This prevents hopelessly overlapping raw boxes from reaching the expensive
repair gate or ontology reasoner.

---

## Diagnostics in the summary dict

`generate_best_layout_from_model(...)` returns `(best_variant, summary)`.

```python
diag = summary["diagnostics"]

diag["jittered_count"]              # times jitter was applied across all candidates
diag["final_best_was_jittered"]     # bool — did the chosen best candidate get jitter?
diag["centroid_collapse_detected"]  # count of samples where collapse was detected
diag["raw_overlap_dropped"]         # count of samples dropped by early filter
diag["resample_attempts_on_overlap"]# total resamples triggered by filter
diag["avg_median_centroid_distance"]# mean metric across all samples
diag["avg_pairwise_iou_fraction"]   # mean fraction of overlapping pairs
diag["avg_collapse_severity"]       # mean severity score (0–1)
```

Use these to drive monitoring dashboards, A/B tests, and threshold tuning.

---

## Running the tests

```bash
# All 55 Phase-1 unit tests
pytest tests/learned/test_jitter_hints.py \
       tests/learned/test_overlap_filter.py \
       tests/learned/test_adaptive_jitter.py -v

# Quick smoke test
pytest tests/learned/test_adaptive_jitter.py -v -k "combined"
```

Test coverage:

| File | Tests | Covers |
|------|-------|--------|
| `test_jitter_hints.py` | 17 | Centroid, IoU, distance, collapse, basic jitter |
| `test_overlap_filter.py` | 13 | Overlap fraction, thresholds, filter decisions |
| `test_adaptive_jitter.py` | 25 | Severity scoring, adaptive σ, boundary bias, combined |

---

## Rollback

Disable everything via env vars (no code change required):

```bash
export LEARNED_JITTER_ENABLED=false
export LEARNED_OVERLAP_FILTER_ENABLED=false
export ADAPTIVE_JITTER_ENABLED=false
export DIRECTIONAL_JITTER_ENABLED=false
```

Or revert only `centroid_utils.py` and the import/call sites in
`model_generation_loop.py`.

---

## Phase 2 — Overlap-Aware Logit Processor ✅

**`OverlapAwareProcessor`** in `learned/model/sample.py`:

Runs *during* autoregressive generation (token-by-token) to mask coordinate bins
that would produce IoU > `IOU_BLOCK_THRESH` with already-placed rooms.

### Environment variables

```bash
LEARNED_OVERLAP_PROCESSOR_ENABLED=false  # master switch (default: false — opt-in)
OVERLAP_PROCESSOR_IOU_THRESH=0.8         # IoU above which y2 bin is masked
OVERLAP_PROCESSOR_CHECK_X2=false         # also check at x2 position (experimental)
OVERLAP_PROCESSOR_IOU_THRESH_X2=0.95     # stricter threshold for x2 proxy check
```

### Design

Token group structure: `ROOM_TOKEN  type_tok  x1_bin  y1_bin  x2_bin  y2_bin`

| tail_len | Position | Check |
|----------|----------|-------|
| 5 | About to sample `y2_bin` | Full box IoU against all prior rooms |
| 4 | About to sample `x2_bin` | Proxy box `(x1, y1, cand_x2, 1.0)` — opt-in |

**Processor chaining** in `constrained_sample_layout`:

```python
if overlap_processor:
    def _chained(logits, seq):
        logits = spec_proc(logits, seq)    # spec constraints first
        logits = overlap_proc(logits, seq)  # overlap check second
        return logits
    processor = _chained
else:
    processor = spec_proc
```

### CLI

```bash
python -m learned.model.sample \
    --checkpoint path/to/model.pt \
    --constrained --spec-file rooms.json \
    --overlap-processor          # activates OverlapAwareProcessor
```

Or set via env var without touching CLI arguments:

```bash
# bash / Linux / macOS
LEARNED_OVERLAP_PROCESSOR_ENABLED=true python -m learned.model.sample ...
```

```powershell
# PowerShell (Windows)
$env:LEARNED_OVERLAP_PROCESSOR_ENABLED="true"; python -m learned.model.sample ...
```

### Tests

```bash
pytest tests/learned/test_overlap_processor.py -v
```

| Test class | What it covers |
|------------|---------------|
| `TestIoU` | Static `_iou()` helper — no-overlap, identical, partial, degenerate |
| `TestBinToNorm` | `_bin_to_norm()` boundary values and midpoint |
| `TestParseCompleteRooms` | Empty, complete, incomplete, inverted-coord cases |
| `TestLastRoomIdx` | Index search for last `ROOM_TOKEN` |
| `TestMaskOverlapBins` | Core masking: empty existing, overlapping bin, distant bin |
| `TestOverlapAwareProcessorCall` | End-to-end: wrong tail lengths, no prior rooms, masking |
| `TestFeatureFlag` | Threshold controls masking width |

---

## Phase 3 — Force-Based Push-Apart Optimizer ✅

**`_force_push_apart`** in `learned/integration/repair_gate.py`:

Runs during Stage 3 (overlap repair) and replaces the greedy push-apart with a
simultaneous-update, mass-proportional spring simulation.

### Environment variables

```bash
REPAIR_FORCE_PUSH_ENABLED=false    # master switch (default: false — opt-in)
REPAIR_FORCE_PUSH_MAX_ITERS=120    # max simulation iterations
REPAIR_FORCE_PUSH_STEP=0.5         # initial step size (scaled by damping each iter)
REPAIR_FORCE_PUSH_DAMPING=0.85     # step *= damping each iteration (prevents oscillation)
REPAIR_FORCE_PUSH_MIN_OV=0.01      # minimum overlap area to count as an overlap
```

### Design improvements over greedy `_push_apart`

| Feature | Greedy | Force-based |
|---------|--------|-------------|
| Update strategy | One room at a time | All rooms simultaneously |
| Force direction | Min-overlap axis | Min-overlap axis |
| Force split | Smaller room moves 100% | Inversely proportional to area |
| Step size | Fixed nudge | Exponentially decaying |
| Convergence | Can oscillate | Damped convergence |

### Activation

```powershell
# PowerShell (Windows)
$env:REPAIR_FORCE_PUSH_ENABLED="true"; python -m your_script ...
```

```bash
# bash / Linux / macOS
REPAIR_FORCE_PUSH_ENABLED=true python -m your_script ...
```

### Tests

```bash
pytest tests/unit/test_force_push_apart.py -v
```

| Test class | What it covers |
|------------|---------------|
| `TestNoOverlap` | Distant / edge-touching rooms stay put |
| `TestSingleOverlap` | Two overlapping rooms separate |
| `TestMultipleOverlaps` | Chain / cluster of overlaps resolved |
| `TestBoundaryRespect` | Rooms clamped inside boundary |
| `TestMassProportionalForce` | Larger room moves less |
| `TestDampingConvergence` | Damping prevents oscillation |
| `TestEdgeCases` | Single room, zero area, None polygon |

---

## Phase 4 — Box Optimizer ✅

**`optimize_box_placement`** in `learned/integration/box_optimizer.py`:

Formal constraint-based optimization for room placement after push-apart fails
to fully resolve overlaps.

### Environment variables

```bash
BOX_OPT_ENABLED=false        # master switch (default: false — opt-in)
BOX_OPT_TIME_LIMIT=2.0       # max solver time in seconds
BOX_OPT_GRID_SCALE=100       # coord → integer scale for CP-SAT
BOX_OPT_MIN_GAP=0.05         # min gap between rooms
```

### Solver hierarchy

1. **OR-Tools CP-SAT** — preferred; exact constraint satisfaction
2. **scipy L-BFGS-B** — fallback gradient-free optimizer
3. **No-op** — if neither available, returns rooms unchanged

### Design

| Feature | Detail |
|---------|--------|
| Decision variables | `(x1, y1)` per room (translation only) |
| Constraints | Rooms inside boundary, `AddNoOverlap2D` |
| Objective | Minimize total Manhattan displacement |
| Integration | Called in `_stage3_overlap_repair` after push-apart |

### Activation

```powershell
# PowerShell (Windows)
$env:BOX_OPT_ENABLED="true"; python your_script.py
```

```bash
# bash / Linux / macOS
BOX_OPT_ENABLED=true python your_script.py
```

### Tests

```bash
pytest tests/unit/test_box_optimizer.py -v
```

| Test class | What it covers |
|------------|---------------|
| `TestBasicFunctionality` | Empty/single room, solver info |
| `TestOverlapResolution` | 2–3 overlapping rooms separated |
| `TestBoundaryRespect` | Rooms stay inside boundary |
| `TestSizePreservation` | Room dimensions unchanged |
| `TestSolverSelection` | Reports solver used, prefer_cpsat flag |
| `TestEdgeCases` | None polygon, identical positions |
