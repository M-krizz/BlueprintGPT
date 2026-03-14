def _safe_ratio(numerator, denominator, default=0.0):
    if denominator <= 0:
        return default
    return numerator / denominator


def _score_variant(variant):
    metrics = variant.get("metrics", {})
    hard_compliance = 1.0 if metrics.get("fully_connected") and metrics.get("travel_distance_compliant") else 0.0

    circulation_ratio = _safe_ratio(
        metrics.get("circulation_walkable_area", 0.0),
        max(metrics.get("total_area", 0.0), 1e-9),
        default=0.0,
    )
    compactness = max(0.0, 1.0 - min(circulation_ratio, 0.45) / 0.45)
    adjacency_score = metrics.get("adjacency_satisfaction", 0.0)

    allowed = metrics.get("max_allowed_travel_distance", 0.0)
    actual = metrics.get("max_travel_distance", 0.0)
    if allowed > 0:
        travel_margin = max(-1.0, min(1.0, (allowed - actual) / allowed))
    else:
        travel_margin = 0.0

    align = metrics.get("alignment_score", 0.0)

    final_score = (
        0.40 * hard_compliance
        + 0.18 * compactness
        + 0.18 * adjacency_score
        + 0.12 * (travel_margin + 1.0) / 2.0
        + 0.12 * align
    )

    return round(final_score, 4), {
        "hard_compliance": round(hard_compliance, 4),
        "compactness": round(compactness, 4),
        "adjacency": round(adjacency_score, 4),
        "travel_margin": round(travel_margin, 4),
        "circulation_ratio": round(circulation_ratio, 4),
        "alignment": round(align, 4),
    }


def rank_layout_variants(variants):
    ranked = []
    for variant in variants:
        score, breakdown = _score_variant(variant)
        variant["ranking"] = {
            "score": score,
            "breakdown": breakdown,
        }
        ranked.append(variant)

    ranked.sort(key=lambda v: v.get("ranking", {}).get("score", 0.0), reverse=True)
    for idx, variant in enumerate(ranked, start=1):
        variant["rank"] = idx
        variant["recommended"] = idx == 1

    return ranked, 0 if ranked else None
