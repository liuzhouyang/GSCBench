from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import networkx as nx

from gscbench.core.dataset import GraphPair
from gscbench.solvers.common import (
    SolverDataAdapter,
    build_solver_model_class,
    get_solver_graph,
)
from gscbench.solvers.common import node_substitution


class AstarConfig:
    def __init__(
        self,
        input_dim: int,
        parallel_workers: int = 0,
        timeout: Optional[float] = None,
    ) -> None:
        self.input_dim = input_dim
        self.parallel_workers = parallel_workers
        self.timeout = timeout


AstarDataAdapter = SolverDataAdapter


def compute_astar_pair(sample: GraphPair, config: AstarConfig) -> float:
    graph_1 = get_solver_graph(sample.graph_1)
    graph_2 = get_solver_graph(sample.graph_2)
    return run_astar(
        graph_1,
        graph_2,
        timeout=config.timeout,
    )


def compute_astar_batch(samples: list[GraphPair], config: AstarConfig) -> list[float]:
    if int(config.parallel_workers) <= 1 or len(samples) <= 1:
        return [compute_astar_pair(sample, config) for sample in samples]

    jobs = []
    for sample in samples:
        graph_1 = get_solver_graph(sample.graph_1)
        graph_2 = get_solver_graph(sample.graph_2)
        jobs.append((graph_1, graph_2, config.timeout))

    with ProcessPoolExecutor(max_workers=int(config.parallel_workers)) as executor:
        return list(executor.map(run_astar_job, jobs))


def run_astar_job(args: tuple[nx.Graph, nx.Graph, Optional[float]]) -> float:
    graph_1, graph_2, timeout = args
    return run_astar(graph_1, graph_2, timeout=timeout)


def run_astar(
    graph_1: nx.Graph,
    graph_2: nx.Graph,
    *,
    timeout: Optional[float],
) -> float:
    result = nx.graph_edit_distance(
        graph_1,
        graph_2,
        node_subst_cost=node_substitution,
        node_del_cost=lambda attrs: 1.0,
        node_ins_cost=lambda attrs: 1.0,
        edge_subst_cost=lambda left, right: 0.0,
        edge_del_cost=lambda attrs: 1.0,
        edge_ins_cost=lambda attrs: 1.0,
        timeout=timeout,
    )
    if result is None:
        return float("inf")
    return float(result)



Astar = build_solver_model_class(
    "Astar",
    compute_astar_pair,
    compute_batch=compute_astar_batch,
)
