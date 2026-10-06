# Experiment protocol

GSCBench studies generalization with exact GED supervision across four
settings. Approximate solvers are evaluated against the same labels.

| Setting | Training | Evaluation |
| --- | --- | --- |
| In-collection | One collection | Its held-out test pairs |
| Cross-collection | Pooled source collections | Source and target collections |
| Zero-shot | A completed source model | Other collections |
| Few-shot / fine-tuning | A budgeted target training pool | Target test pairs |

## Splits and metrics

Graph IDs are shuffled with `split_seed`. `val_ratio` and `test_ratio` are
fractions of the collection; each count is rounded down and the remaining
graphs form the training set. The experiment configs use 0.04 and 0.16,
which gives approximately 80% / 4% / 16%. Set both ratios to 0.2 for
60% / 20% / 20%.

| Split | Pair endpoints |
| --- | --- |
| Training | train-train |
| Validation | val-train |
| Test | test-train, test-val and test-test |

Regression metrics use raw GED and count each unordered pair once. Ranking
metrics order candidates for each test query. Test-test scores contribute to
the candidate lists of both endpoint queries.

## Training batches

`batch_size` is the number of graph pairs in a training batch. `num_iters` caps
the number of batches per source collection in each epoch. For example,
`in_collection.yaml` uses `batch_size: 128` and `num_iters: 100`, so training
processes up to 12,800 pairs per source collection per epoch. The actual count
can be smaller when a collection has fewer pairs.

Reducing `batch_size` lowers the memory needed for one forward/backward pass.
When `--batch-size` is supplied, the runner automatically adjusts `num_iters`
to keep `batch_size × num_iters` approximately constant. For example, changing
the full-exact batch size from 128 to 32 changes 100 iterations to 400. If
`--num-iters` is also set to a value different from the configured default,
that value is used directly. When editing a runtime YAML, adjust both values
manually to keep the same number of training pairs.

Gradient accumulation combines several micro-batches before one optimizer
update. It does not change `num_iters` or the number of pairs processed by
itself. To keep the effective batch size and training pair count while reducing
memory, use a smaller `batch_size`, increase `gradient_accumulation_steps`, and
increase `num_iters` proportionally. For example, `batch_size: 32`,
`gradient_accumulation_steps: 4`, and `num_iters: 400` preserve the full-exact
setting's effective batch size of 128 and its 12,800 pairs per source
collection per epoch. Passing `--batch-size 32` performs the iteration increase
automatically; the accumulation setting is still explicit, for example
`--gradient-accumulation-steps 4`. If the iterations are not increased, fewer
pairs and fewer optimizer updates are used in each epoch.

Accumulated gradients are weighted by the number of graph pairs in each
micro-batch, including the final partial batch. GEDRanker uses its own
optimization procedure and requires `gradient_accumulation_steps: 1`.

## GraphEdX graph-size limit

GraphEdX pads each graph's adjacency matrix to `max_node_set_size`. For a new
training run, this limit is set to the largest graph in the training
collections. Checkpoint evaluation uses the limit saved with that checkpoint.
If an evaluation graph exceeds the limit, GraphEdX raises an error; the
collection's pairs are not filtered.

## Experiment schedules

The paper uses split seeds 1729, 3407 and 9679. In-collection and few-shot
scratch use training seeds 0, 1 and 2 with split 1729, then seed 0 with splits
3407 and 9679. The paper's few-shot scratch study spans all 20 learned models
and all 11 benchmark collections, with pair budgets 100, 500 and 2,000.

Cross-collection pretraining uses AIDS, BZR, ENZYMES, MUTAG, NCI1, PTC_MR and
auxiliary `ged_pyg_LINUX` as sources. Targets are COX2, DHFR, PROTEINS,
IMDB-BINARY and ogbg-code2. The provided config uses split seed 1729 and
training seed 0.

Fine-tuning uses the five target collections listed above, budgets 100, 500
and 2,000, split seed 1729 and training seeds 0, 1 and 2. It starts from
pretrained weights with fresh optimizer state. Validation uses val-train pairs;
test pairs are held out for evaluation.

The few-shot pair budget counts exact train-train pairs. Selection prioritizes
graph coverage, then lower pair degree, with ties determined by the split seed.
Budgets are nested for a fixed split.

## Approximate solvers

Hungarian, VJ, Beam and FGWAlign are evaluation-only methods. They report metrics on the
main test set and its test-train-val and test-test views.
