from pathlib import Path
from typing import Optional, Union


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_runtime_path(path_value: Optional[Union[Path, str]]) -> Optional[Path]:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return project_root() / path


def get_project_relative_path(path_value: Union[Path, str]) -> str:
    path = get_runtime_path(path_value)
    root = project_root()
    if path.is_relative_to(root):
        return str(path.relative_to(root))
    return str(path)
