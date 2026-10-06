"""Model registry entry points."""

from gscbench.core.model import Model
from gscbench.models.registry import build_model, get_model_entry, register_model

__all__ = ["Model", "build_model", "get_model_entry", "register_model"]
