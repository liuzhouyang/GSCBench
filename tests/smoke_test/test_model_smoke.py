from pathlib import Path

import pytest

from .harness import build_batch
from .harness import build_gediot_variant_batch
from .harness import cleanup_temp_path
from .harness import run_evaluation_smoke
from .harness import run_training_smoke
from .harness import set_smoke_seed


LEARNABLE_MODEL_CASES = [
    {"id": "eric", "name": "eric", "pair_limit": 4},
    {"id": "egsc_teacher", "name": "egsc", "variant": "teacher", "pair_limit": 4},
    {"id": "egsc_kd", "name": "egsc", "variant": "kd", "pair_limit": 4},
    {"id": "gedgnn", "name": "gedgnn", "pair_limit": 4},
    {"id": "gediot", "name": "gediot", "pair_limit": 2},
    {
        "id": "gelato",
        "name": "gelato",
        "pair_limit": 4,
        "overrides": {"hidden_dim": 32, "num_layers": 2, "inference_branches": 4},
    },
    {
        "id": "gen",
        "name": "gen",
        "pair_limit": 2,
        "overrides": {"hidden_dim": 16, "output_dim": 8, "num_layers": 2},
    },
    {
        "id": "genn_astar",
        "name": "genn_astar",
        "pair_limit": 2,
        "overrides": {
            "filters_1": 16,
            "filters_2": 8,
            "filters_3": 4,
            "tensor_neurons": 8,
            "bottle_neck_neurons": 8,
            "bins": 4,
        },
    },
    {"id": "gmn", "name": "gmn", "pair_limit": 2},
    {
        "id": "gotsim",
        "name": "gotsim",
        "pair_limit": 2,
        "overrides": {"gcn_size": [16, 8, 4]},
    },
    {
        "id": "graphedx",
        "name": "graphedx",
        "pair_limit": 4,
        "overrides": {"max_node_set_size": 8},
    },
    {
        "id": "graph2region",
        "name": "graph2region",
        "pair_limit": 2,
        "overrides": {
            "num_layers": 2,
            "hidden_dim": 16,
            "output_dim": 8,
            "num_perms": 2,
            "length_pe": 2,
            "max_num_nodes": 8,
        },
    },
    {
        "id": "grasp",
        "name": "grasp",
        "pair_limit": 2,
        "overrides": {"hidden_dim": 16, "tensor_neurons": 8, "reduction": 1},
    },
    {
        "id": "greed",
        "name": "greed",
        "pair_limit": 4,
        "overrides": {"hidden_dim": 16, "output_dim": 32},
    },
    {
        "id": "h2mn",
        "name": "h2mn",
        "pair_limit": 2,
        "overrides": {"hidden_dim": 16, "k": 2},
    },
    {
        "id": "mata",
        "name": "mata",
        "pair_limit": 2,
        "overrides": {
            "filter_1": 16,
            "filter_2": 16,
            "filter_3": 16,
            "random_walk_step": 4,
            "max_degree": 4,
            "topk": 2,
        },
    },
    {
        "id": "noah",
        "name": "noah",
        "pair_limit": 2,
        "overrides": {
            "filters_1": 16,
            "filters_2": 8,
            "filters_3": 4,
            "tensor_neurons": 8,
            "bottle_neck_neurons": 8,
        },
    },
    {"id": "graphsim", "name": "graphsim", "pair_limit": 2},
    {
        "id": "simgnn",
        "name": "simgnn",
        "pair_limit": 2,
        "overrides": {
            "filters_1": 16,
            "filters_2": 8,
            "filters_3": 8,
            "tensor_neurons": 8,
            "bottle_neck_neurons": 8,
            "bins": 4,
        },
    },
    {
        "id": "tagsim",
        "name": "tagsim",
        "pair_limit": 4,
        "include_component_targets": True,
        "overrides": {
            "tensor_neurons": 8,
            "bottle_neck_neurons": 8,
            "component_loss": "mse",
            "component_loss_weight": 1.0,
        },
    },
]


EVALUATION_ONLY_CASES = [
    {
        "id": "gedranker",
        "name": "gedranker",
        "pair_limit": 1,
        "overrides": {
            "hidden_dim": [16, 8, 8, 8, 8, 8],
            "d_hidden_dim": [16, 8, 8],
            "diffusion_steps": 8,
            "inference_diffusion_steps": 2,
            "tau": 1.0,
            "gumbel_iteration": 2,
            "test_k": 2,
            "num_delta_graphs": 4,
            "delta_graph_cutoff": 2,
        },
    },
]


@pytest.mark.parametrize("case", LEARNABLE_MODEL_CASES, ids=[case["id"] for case in LEARNABLE_MODEL_CASES])
def test_learnable_model_training_smoke(case):
    set_smoke_seed()
    cleanup_path = None
    model = None
    try:
        model, batch, cleanup_path = build_batch(
            case["name"],
            variant=case.get("variant"),
            overrides=case.get("overrides"),
            pair_limit=case.get("pair_limit", 2),
            include_gt_mappings=case.get("include_gt_mappings", True),
            include_component_targets=case.get("include_component_targets", False),
        )
        run_training_smoke(model, batch, steps=1)
    finally:
        cleanup_temp_path(cleanup_path)


@pytest.mark.parametrize("case", EVALUATION_ONLY_CASES, ids=[case["id"] for case in EVALUATION_ONLY_CASES])
def test_evaluation_only_model_smoke(case):
    set_smoke_seed()
    cleanup_path = None
    try:
        model, batch, cleanup_path = build_batch(
            case["name"],
            variant=case.get("variant"),
            overrides=case.get("overrides"),
            pair_limit=case.get("pair_limit", 1),
            include_gt_mappings=case.get("include_gt_mappings", True),
            include_component_targets=case.get("include_component_targets", False),
        )
        run_evaluation_smoke(model, batch)
    finally:
        cleanup_temp_path(cleanup_path)


@pytest.mark.parametrize("variant", ["gw", "hot"])
def test_gediot_variant_evaluation_smoke(variant):
    set_smoke_seed()
    model, batch = build_gediot_variant_batch(variant, pair_limit=2)
    run_evaluation_smoke(model, batch)


def test_tagsim_without_component_supervision_smoke():
    set_smoke_seed()
    model, batch, cleanup_path = build_batch(
        "tagsim",
        overrides={"tensor_neurons": 8, "bottle_neck_neurons": 8},
        pair_limit=2,
        include_gt_mappings=False,
        include_component_targets=False,
    )
    try:
        run_training_smoke(model, batch, steps=1)
        run_evaluation_smoke(model, batch)
    finally:
        cleanup_temp_path(cleanup_path)
