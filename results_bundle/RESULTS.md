# Results Package

## Demo Artifacts

### Learned Best
![Learned Best](blueprint_learned.svg)

### Algorithmic Baseline
![Algorithmic Baseline](blueprint_algorithmic.svg)

## LLM Explanation Metadata

| Run | Used | Provider | Model | Latency (ms) | Warning | Evidence Hash |
| --- | --- | --- | --- | --- | --- | --- |
| Algorithmic | False | deterministic | None | None | None | 1425f95371c08f3af464cbbdbdfd1a19a949d8a75d152d77dc93124d7903f6f4 |
| Learned | False | deterministic | None | None | None | 17dec70326296af1da5d40c908b05dd1b7488b55565936c3a9822516bc38ec0b |

## Output Summary

- Learned status: `COMPLIANT`
- Algorithmic status: `COMPLIANT`
- Learned design score: `0.3347`
- Algorithmic design score: `0.5004`
- Alternatives: 3 shown

## Ablation Table

| Mode | Samples | Raw Validity | Post-Repair Validity | Avg Violations | Avg Travel Margin | Avg Adjacency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| algorithmic_only | 15 | 1.0 | 1.0 | 0.0 | 6.87 | 0.6552 |
| learned_no_repair | 3 | 0.0 | 0.0 | 2.67 | 6.41 | 0.0575 |
| learned_repair | 3 | 0.0 | 0.3333 | 6.67 | 8.55 | 0.3563 |
| learned_repair_kg | 3 | 0.0 | 0.6667 | 5.33 | 6.86 | 0.3563 |

