from math import exp

from gscbench.core.dataset import GraphPair


def get_objective_settings(target_mode, normalize_target=None):
    if target_mode != "ged":
        raise ValueError("The benchmark target is GED.")
    return "ged", bool(normalize_target)


def get_pair_objective_value(sample: GraphPair, target_mode="ged", *, normalize_target=False) -> float:
    raw_ged = get_pair_raw_value(sample)
    return exp(-raw_ged / sample.get_mean_nodes()) if normalize_target else raw_ged


def get_pair_raw_value(sample: GraphPair, target_mode="ged") -> float:
    return float(sample.metadata.get("raw_ged", sample.target))

