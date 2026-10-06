"""Framework core abstractions and shared result types."""

from gscbench.core.dataset import DatasetSplit, GraphPair
from gscbench.core.result import RuntimeConfig

__all__ = [
    "DatasetSplit",
    "GraphPair",
    "RuntimeConfig",
]
