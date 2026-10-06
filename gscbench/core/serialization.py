from typing import Any

import torch
import yaml


def jsonify(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() <= 64:
            return value.detach().cpu().tolist()
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "numel": int(value.numel()),
        }
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def yaml_dumps(value: Any) -> str:
    return yaml.safe_dump(jsonify(value), allow_unicode=True, sort_keys=False)
