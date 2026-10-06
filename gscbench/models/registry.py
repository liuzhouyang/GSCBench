from typing import Any, Optional, Type

from gscbench.core.model import Model


class ModelEntry:
    def __init__(
        self,
        name: str,
        model_cls: Type[Model],
        config_cls: Optional[type[Any]] = None,
        adapter_cls: Optional[type[Any]] = None,
    ) -> None:
        self.name = name
        self.model_cls = model_cls
        self.config_cls = config_cls
        self.adapter_cls = adapter_cls


MODEL_REGISTRY: dict[str, ModelEntry] = {}


def register_model(
    name: str,
    model_cls: Type[Model],
    config_cls: Optional[type[Any]] = None,
    adapter_cls: Optional[type[Any]] = None,
) -> None:
    if name in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is already registered.")
    MODEL_REGISTRY[name] = ModelEntry(
        name=name,
        model_cls=model_cls,
        config_cls=config_cls,
        adapter_cls=adapter_cls,
    )


def build_model(name: str, **kwargs: object) -> Model:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'.")
    return MODEL_REGISTRY[name].model_cls(**kwargs)


def get_model_entry(name: str) -> ModelEntry:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'.")
    return MODEL_REGISTRY[name]
