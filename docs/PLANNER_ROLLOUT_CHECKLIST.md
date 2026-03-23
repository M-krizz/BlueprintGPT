# Planner Rollout Checklist

Use this after training a planner checkpoint and running benchmark comparisons.

## Current Repo State

- Trained checkpoint present at `learned/planner/checkpoints/room_planner.pt`
- Benchmark summary present at `learned/planner/checkpoints/summary.json`
- Current benchmark result does **not** justify planner promotion yet:
  - planner wins: 1
  - algorithmic wins: 3
  - average score delta: -0.0180
  - average alignment delta: -0.0718

## Evaluate Current Checkpoint

Run:

```powershell
python -m learned.planner.evaluate_rollout --input learned/planner/checkpoints/summary.json
```

Expected recommendation for the current summary:

- `hold algorithmic default and improve planner training/data`

## Promotion Gate

Do not switch core residential auto-routing to planner unless all of these pass:

- all benchmark cases complete successfully
- planner wins at least as many cases as algorithmic
- planner average design score delta is non-negative
- planner average adjacency delta is non-negative
- planner average alignment delta is non-negative
- planner layouts remain compliant
- planner layouts remain fully connected
- planner room coverage exactly matches the requested program

## Next Work

Improve the planner before re-running rollout evaluation:

- rebalance planner training data toward underrepresented room programs
- add more synthetic teacher cases for realistic 2BHK and 3BHK variants
- improve planner losses so adjacency and spatial alignment matter more
- add planner-side validation slices by room program instead of random-only validation
- consider a reranker for planner-guided candidates before final selection

## Suggested Runtime Setting For Now

Keep:

```env
BLUEPRINT_BACKEND_MODE=auto
BLUEPRINT_AUTO_CORE_BACKEND=algorithmic
```

Move to `planner_if_available` only after the rollout evaluator passes.
