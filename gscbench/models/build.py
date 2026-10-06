import inspect
from typing import Optional

from gscbench.models.registry import build_model, get_model_entry


def collect_init_kwargs(
    callable_obj,
    *,
    required_kwargs: dict[str, object],
    optional_sources: dict[str, object],
) -> dict[str, object]:
    parameters = inspect.signature(callable_obj).parameters
    kwargs = dict(required_kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        for name, value in optional_sources.items():
            kwargs.setdefault(name, value)
        return kwargs

    accepted_names = {
        name
        for name, parameter in parameters.items()
        if name != "self" and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    for name, value in optional_sources.items():
        if name in accepted_names and name not in kwargs:
            kwargs[name] = value
    return kwargs


def build_model_and_adapter(
    model_name: str,
    *,
    datasets,
    runtime,
    checkpoint=None,
    model_data: dict[str, object],
    dataset_name: Optional[str] = None,
):
    model_entry = get_model_entry(model_name)
    model_data = dict(model_data)
    if not checkpoint:
        model_data["input_dim"] = max(dataset.input_dim for dataset in datasets)
    input_dim = int(model_data["input_dim"])
    for dataset in datasets:
        dataset.set_input_dim(input_dim)
    model_data = model_entry.model_cls.prepare_config(
        model_data, datasets=datasets, runtime=runtime, checkpoint=checkpoint,
    )

    if model_entry.config_cls is None:
        model_config = dict(model_data)
    else:
        model_config = model_entry.config_cls(
            **collect_init_kwargs(
                model_entry.config_cls,
                required_kwargs={},
                optional_sources=model_data,
            )
        )
    adapter_sources = dict(model_data)
    if dataset_name is not None:
        adapter_sources["dataset_name"] = dataset_name
    adapter_kwargs = collect_init_kwargs(
        model_entry.adapter_cls,
        required_kwargs={
            "input_dim": input_dim,
        },
        optional_sources=adapter_sources,
    )
    model = build_model(model_name, config=model_config)
    adapter = model_entry.adapter_cls(**adapter_kwargs)
    model.adapter_config = adapter_kwargs
    if isinstance(model.config, dict):
        model_config_data = dict(model.config)
    else:
        model_config_data = dict(model.config.__dict__)
    return model, adapter, adapter_kwargs, model_config_data
