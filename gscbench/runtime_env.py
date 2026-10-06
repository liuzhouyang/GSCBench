import os
import tempfile
from pathlib import Path
from typing import Optional
from gscbench.core.paths import project_root as get_project_root


def get_runtime_root(project_root: Optional[Path] = None) -> Path:
    root = Path(project_root) if project_root is not None else get_project_root()
    configured = os.environ.get("GSCBENCH_RUNTIME_ROOT")
    if configured:
        runtime_root = root / Path(configured).expanduser()
    else:
        runtime_root = root / ".gscbench-runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    return runtime_root


def configure_runtime_environment(project_root: Optional[Path] = None) -> None:
    runtime_root = get_runtime_root(project_root)
    temp_root = runtime_root / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TMPDIR", str(temp_root))
    os.environ.setdefault("TEMP", str(temp_root))
    os.environ.setdefault("TMP", str(temp_root))
    tempfile.tempdir = str(temp_root)
