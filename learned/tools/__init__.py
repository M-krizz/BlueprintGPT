"""
BlueprintGPT Tools Package

Development and debugging tools for model analysis and pipeline optimization.

Key Components:
- debug_cli: Interactive model debugging and inspection tool
- quality_analyzer: Layout quality assessment and scoring tools
- performance_profiler: Detailed performance analysis and benchmarking

Usage:
    # Interactive debugging session
    python -m learned.tools.debug_cli --checkpoint path.pt --interactive

    # Quick analysis
    python -m learned.tools.debug_cli --checkpoint path.pt --analyze
"""

try:
    from .debug_cli import ModelDebugger
    __all__ = ["ModelDebugger"]
except ImportError:
    # Handle missing dependencies gracefully
    __all__ = []