import pytest

from .harness import build_batch
from .harness import cleanup_temp_path
from .harness import run_evaluation_smoke
from .harness import set_smoke_seed


SOLVER_CASES = [
    {"id": "hungarian", "name": "hungarian", "pair_limit": 4},
    {"id": "vj", "name": "vj", "pair_limit": 4},
    {"id": "beam", "name": "beam", "pair_limit": 4, "overrides": {"beam_size": 2}},
    {"id": "astar", "name": "astar", "pair_limit": 4, "overrides": {"timeout": 5.0}},
    {
        "id": "fgwalign",
        "name": "fgwalign",
        "pair_limit": 2,
        "overrides": {"patience": 1, "topk": 2, "light": True},
    },
]


@pytest.mark.parametrize("case", SOLVER_CASES, ids=[case["id"] for case in SOLVER_CASES])
def test_solver_evaluation_smoke(case):
    set_smoke_seed()
    cleanup_path = None
    try:
        model, batch, cleanup_path = build_batch(
            case["name"],
            overrides=case.get("overrides"),
            pair_limit=case.get("pair_limit", 2),
            include_gt_mappings=True,
            include_component_targets=False,
        )
        assert model.requires_training is False
        assert model.uses_manual_optimization() is False
        run_evaluation_smoke(model, batch)
    finally:
        cleanup_temp_path(cleanup_path)
