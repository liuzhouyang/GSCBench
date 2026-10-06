from typing import Any, Sequence
import os
from pathlib import Path

import torch
from torch_geometric.data import Batch

from gscbench.core.adapter import ModelDataAdapter
from gscbench.core.dataset import GraphPair
from gscbench.core.paths import project_root
from gscbench.models.mata.src.mydegree import MyDegree
from gscbench.models.mata.src.randomWalk import AddRandomWalkPE


class MATABatch:
    def __init__(
        self,
        graph_1: Any,
        graph_2: Any,
        targets: torch.Tensor,
        raw_targets: torch.Tensor,
        avg_v: torch.Tensor,
        matching_targets: torch.Tensor,
        has_gt_mappings: torch.Tensor,
        num_nodes_1: torch.Tensor,
        num_nodes_2: torch.Tensor,
        num_edges_1: torch.Tensor,
        num_edges_2: torch.Tensor,
    ) -> None:
        self.graph_1 = graph_1
        self.graph_2 = graph_2
        self.targets = targets
        self.raw_targets = raw_targets
        self.avg_v = avg_v
        self.matching_targets = matching_targets
        self.has_gt_mappings = has_gt_mappings
        self.num_nodes_1 = num_nodes_1
        self.num_nodes_2 = num_nodes_2
        self.num_edges_1 = num_edges_1
        self.num_edges_2 = num_edges_2


class MATADataAdapter(ModelDataAdapter[MATABatch]):
    def __init__(
        self,
        input_dim: int,
        target_mode: str = "ged",
        normalize_target: bool = True,
        *,
        nonstruc: bool = False,
        max_degree: int = 12,
        random_walk_step: int = 16,
        dataset_name: str = "dataset",
    ) -> None:
        self.input_dim = input_dim
        self.target_mode = str(target_mode).lower()
        self.normalize_target = bool(normalize_target)
        self.nonstruc = bool(nonstruc)
        self.max_degree = max(1, int(max_degree))
        self.random_walk_step = max(1, int(random_walk_step))
        self.dataset_name = str(dataset_name)
        self.degree_transform = MyDegree(self.max_degree)
        self.random_walk_transform = AddRandomWalkPE(self.random_walk_step)
        self.cache_dir = Path(
            os.environ.get(
                "GSCBENCH_MATA_CACHE_DIR",
                str(project_root() / ".gscbench-runtime" / "mata_cache"),
            )
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / f"{self.dataset_name}.pt"
        self._feature_cache = None
        self._cache_dirty = False

    def _graph_id(self, graph: Any) -> int:
        return int(getattr(graph, "gid", getattr(graph, "i", -1)))

    def _load_feature_cache(self) -> dict:
        if self._feature_cache is None:
            try:
                payload = torch.load(str(self.cache_path), map_location="cpu", weights_only=False)
                self._feature_cache = payload if isinstance(payload, dict) else {}
            except (OSError, RuntimeError, TypeError):
                self._feature_cache = {}
        return self._feature_cache

    def _load_cached_features(self, graph: Any) -> bool:
        try:
            row = self._load_feature_cache().get(str(self._graph_id(graph)))
            if row is None:
                return False
            cent_pe, rw_pe = row
            if (
                not torch.is_tensor(cent_pe)
                or not torch.is_tensor(rw_pe)
                or int(cent_pe.size(0)) != int(graph.num_nodes)
                or int(rw_pe.size(0)) != int(graph.num_nodes)
                or int(rw_pe.size(1)) != self.random_walk_step
            ):
                return False
            graph.cent_pe = cent_pe
            graph.rw_pe = rw_pe
            return True
        except (OSError, KeyError, RuntimeError, TypeError):
            return False

    def _save_cached_features(self, graph: Any) -> None:
        cache = self._load_feature_cache()
        cache[str(self._graph_id(graph))] = (graph.cent_pe.cpu(), graph.rw_pe.cpu())
        self._cache_dirty = True

    def _flush_feature_cache(self) -> None:
        if not self._cache_dirty:
            return
        temporary = self.cache_path.with_suffix(f".{os.getpid()}.tmp")
        try:
            torch.save(self._feature_cache, str(temporary))
            os.replace(temporary, self.cache_path)
            self._cache_dirty = False
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def collate(self, samples: Sequence[GraphPair]) -> MATABatch:
        graph_1_list = []
        graph_2_list = []
        targets = []
        raw_targets = []
        avg_v = []
        num_nodes_1 = []
        num_nodes_2 = []
        num_edges_1 = []
        num_edges_2 = []

        ordered_samples = [self.order_pair(sample) for sample in samples]
        max_num_nodes_1 = 0
        max_num_nodes_2 = 0

        for sample in ordered_samples:
            graph_1 = sample.graph_1
            graph_2 = sample.graph_2
            self.ensure_node_features(graph_1)
            self.ensure_node_features(graph_2)
            if self.nonstruc:
                self.ensure_constant_features(graph_1)
                self.ensure_constant_features(graph_2)
            else:
                self.ensure_structure_features(graph_1)
                self.ensure_structure_features(graph_2)

            graph_1_list.append(graph_1)
            graph_2_list.append(graph_2)

            n1, n2 = sample.get_num_nodes()
            e1, e2 = sample.get_num_edges()
            num_nodes_1.append(n1)
            num_nodes_2.append(n2)
            num_edges_1.append(e1)
            num_edges_2.append(e2)
            max_num_nodes_1 = max(max_num_nodes_1, n1)
            max_num_nodes_2 = max(max_num_nodes_2, n2)

            targets.append(
                self.get_objective_value(
                    sample,
                    self.target_mode,
                    normalize_target=self.normalize_target,
                )
            )
            raw_targets.append(self.get_raw_objective_value(sample, self.target_mode))
            avg_v.append(sample.get_mean_nodes())

        # Write once per collated batch instead of rewriting the complete
        # dataset cache once for every graph.
        self._flush_feature_cache()

        matching_targets = torch.zeros(
            (len(ordered_samples), max_num_nodes_1, max_num_nodes_2),
            dtype=torch.float32,
        )
        has_gt_mappings = torch.zeros(len(ordered_samples), dtype=torch.bool)

        for batch_index, sample in enumerate(ordered_samples):
            gt_mappings = sample.metadata.get("gt_mappings", [])
            if not gt_mappings:
                continue
            has_gt_mappings[batch_index] = True
            mapping = gt_mappings[0]
            for source_index, target_index in enumerate(mapping):
                target_index = int(target_index)
                if target_index < 0:
                    continue
                if source_index < num_nodes_1[batch_index] and target_index < num_nodes_2[batch_index]:
                    matching_targets[batch_index, source_index, target_index] = 1.0

        return MATABatch(
            graph_1=Batch.from_data_list(graph_1_list),
            graph_2=Batch.from_data_list(graph_2_list),
            targets=torch.tensor(targets, dtype=torch.float32),
            raw_targets=torch.tensor(raw_targets, dtype=torch.float32),
            avg_v=torch.tensor(avg_v, dtype=torch.float32),
            matching_targets=matching_targets,
            has_gt_mappings=has_gt_mappings,
            num_nodes_1=torch.tensor(num_nodes_1, dtype=torch.long),
            num_nodes_2=torch.tensor(num_nodes_2, dtype=torch.long),
            num_edges_1=torch.tensor(num_edges_1, dtype=torch.long),
            num_edges_2=torch.tensor(num_edges_2, dtype=torch.long),
        )

    def order_pair(self, sample: GraphPair) -> GraphPair:
        graph_1 = sample.graph_1
        graph_2 = sample.graph_2
        n1, n2 = sample.get_num_nodes()
        gid_1 = int(sample.metadata.get("graph_id_1", getattr(graph_1, "gid", getattr(graph_1, "i", -1))))
        gid_2 = int(sample.metadata.get("graph_id_2", getattr(graph_2, "gid", getattr(graph_2, "i", -1))))

        should_swap = n1 > n2 or (n1 == n2 and gid_1 > gid_2)
        if not should_swap:
            return sample

        gt_mappings = sample.metadata.get("gt_mappings", [])
        inverted_mappings = []
        for mapping in gt_mappings:
            inverted = [-1] * int(graph_2.num_nodes)
            for source_index, target_index in enumerate(mapping):
                target_index = int(target_index)
                if target_index < 0:
                    continue
                if source_index < int(graph_1.num_nodes) and target_index < int(graph_2.num_nodes):
                    inverted[target_index] = int(source_index)
            inverted_mappings.append(inverted)

        metadata = dict(sample.metadata)
        metadata["graph_id_1"] = gid_2
        metadata["graph_id_2"] = gid_1
        metadata["num_nodes_1"] = int(graph_2.num_nodes)
        metadata["num_nodes_2"] = int(graph_1.num_nodes)
        metadata["num_edges_1"] = int(sample.metadata.get("num_edges_2", sample.count_edges(graph_2)))
        metadata["num_edges_2"] = int(sample.metadata.get("num_edges_1", sample.count_edges(graph_1)))
        metadata["gt_mappings"] = inverted_mappings
        return GraphPair(
            graph_1=graph_2,
            graph_2=graph_1,
            target=sample.target,
            metadata=metadata,
        )

    def ensure_constant_features(self, graph: Any) -> None:
        if getattr(graph, "x", None) is None:
            graph.x = torch.ones((graph.num_nodes, 1), dtype=torch.float32)
            return
        if graph.x.dim() == 1:
            graph.x = graph.x.unsqueeze(-1).float()
        else:
            graph.x = graph.x.float()

    def ensure_structure_features(self, graph: Any) -> None:
        # Load both derived features together when available.  This avoids
        # recomputing degree features before discovering a valid cache entry.
        if self._load_cached_features(graph):
            return

        degree_attr = getattr(graph, "cent_pe", None)
        if degree_attr is None or degree_attr.size(0) != graph.num_nodes:
            self.degree_transform(graph)

        random_walk_attr = getattr(graph, "rw_pe", None)
        if (
            random_walk_attr is None
            or random_walk_attr.size(0) != graph.num_nodes
            or random_walk_attr.size(1) != self.random_walk_step
        ):
            self.random_walk_transform(graph)
        self._save_cached_features(graph)
