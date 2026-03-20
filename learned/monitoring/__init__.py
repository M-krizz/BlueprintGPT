"""
BlueprintGPT Monitoring Package

Real-time monitoring and quality assurance tools for model generation.

Key Components:
- quality_dashboard: Real-time quality tracking and alerting
- performance_monitor: Generation timing and resource usage tracking
- model_drift_detector: Automated detection of model output degradation

Usage:
    from learned.monitoring import get_global_dashboard, log_generation_quality

    # Log generation results for monitoring
    log_generation_quality(summary_dict, building_type="residential")

    # Get current quality statistics
    stats = get_global_dashboard().get_current_stats()
"""

from .quality_dashboard import (
    QualityDashboard,
    QualityMetrics,
    get_global_dashboard,
    log_generation_quality,
    get_quality_stats,
    save_dashboard_html
)

__all__ = [
    "QualityDashboard",
    "QualityMetrics",
    "get_global_dashboard",
    "log_generation_quality",
    "get_quality_stats",
    "save_dashboard_html"
]