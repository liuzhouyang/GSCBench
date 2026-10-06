"""Compose benchmark test views from one prediction per pair."""
import torch

from gscbench.core.dataset import DatasetSplit
from gscbench.data.gscdataset import PairSplit
from gscbench.core.metrics import compute_metrics, compute_ranking_metrics


def evaluate_test(trainer, dataset):
    names = ("test", "test_train_val", "test_test")
    main, first, second = [dataset.get_split(name) for name in names]
    limit = trainer.runtime.params.get("max_eval_pairs")
    if limit is not None:
        selected = main.samples.entries[:max(1, int(limit))]
        main = DatasetSplit("test", PairSplit(main.samples.dataset, selected))
        selected_rows = set(selected[:, 0].tolist())
        views = []
        for split in (first, second):
            entries = split.samples.entries
            mask = torch.tensor([int(row) in selected_rows for row in entries[:, 0]], dtype=torch.bool)
            views.append(DatasetSplit(split.name, PairSplit(split.samples.dataset, entries[mask])))
        first, second = views

    positions = {tuple(entry): i for i, entry in enumerate(main.samples.entries.tolist())}
    caches, indices, supplemental = [], [], {}
    for split in (first, second):
        trainer.log(f"Evaluating {split.name}: {len(split.samples)} pairs")
        _, _, metrics, details = trainer.evaluate(split, return_cache=True)
        caches.append(details["predictions"])
        supplemental[split.name] = metrics
        indices.extend(positions[tuple(entry)] for entry in split.samples.entries.tolist())
    order = torch.argsort(torch.tensor(indices, dtype=torch.long))
    nonempty = [cache for cache in caches if cache[0][""][0].numel()]
    names = (nonempty or caches)[0][0].keys()
    merged = {
        name: tuple(torch.cat([cache[0].get(name, (torch.empty(0),) * 4)[i]
                               for cache in caches])[order] for i in range(4))
        for name in names
    }
    queries = [q for cache in caches for q in cache[1]]
    queries = [queries[i] for i in order.tolist()]
    # Complete the test-query by test-database block using symmetry.
    qq_database = [int(sample.metadata["graph_id_2"]) for sample in second.samples]
    metrics = {}
    for name, (prediction, target, _, _) in merged.items():
        qq = caches[1][0].get(name, (torch.empty(0),) * 4)
        rank_prediction = torch.cat([merged[name][2], qq[2]])
        rank_target = torch.cat([merged[name][3], qq[3]])
        values = compute_metrics(prediction, target)
        values.update(compute_ranking_metrics(
            rank_prediction, rank_target, queries + qq_database,
        ))
        prefix = name + "_" if name else ""
        metrics.update({prefix + key: value for key, value in values.items()})
        ranking = compute_ranking_metrics(
            torch.cat([qq[2], qq[2]]), torch.cat([qq[3], qq[3]]),
            caches[1][1] + qq_database,
        )
        supplemental["test_test"].update({prefix + key: value for key, value in ranking.items()})
    prediction, target, _, _ = merged[""]
    return prediction, target, metrics, supplemental


def evaluate_collections(trainer, datasets):
    results = {}
    for name, dataset in datasets.items():
        _, _, metrics, supplemental = evaluate_test(trainer, dataset)
        results[name] = {"test": metrics, **supplemental}
    return results


def collection_views(results, dataset_names):
    views = {}
    for view in ("test", "test_train_val", "test_test"):
        groups = {name: {"metrics": results[name][view]} for name in dataset_names}
        values = {}
        for group in groups.values():
            for metric, value in group["metrics"].items():
                values.setdefault(metric, []).append(value)
        views[view] = {"groups": groups,
                       "macro_metrics": {key: sum(items) / len(items) for key, items in values.items()}}
    return views
