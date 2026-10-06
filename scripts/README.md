# Experiment scripts

Run commands from the project directory. Python module entrypoints run one experiment;
shell scripts repeat the paper's settings for one model.

| Script | Arguments | Runs |
| --- | --- | --- |
| `in_collection.sh` | model, collection, optional variant | Five split/seed settings |
| `cross_collection.sh` | model, optional variant | Configured source/target collections, split 1729, seed 0 |
| `fewshot_scratch.sh` | model, optional collection, optional variant | All 11 benchmark collections by default; budgets 100/500/2000 and five split/seed settings |
| `fewshot_finetune.sh` | model, target, checkpoint, optional variant | Budgets 100/500/2000; split 1729; seeds 0/1/2 |
| `zeroshot.sh` | model, source, target, checkpoint, split seed, training seed, optional variant | One evaluation |
| `approximate.sh` | method, collection | Split seeds 1729/3407/9679 |

The five in-collection and scratch seeds are (1729, 0), (1729, 1), (1729, 2),
(3407, 0) and (9679, 0). Set `SEED_PAIRS` and `PAIR_BUDGETS` to select a
shorter shell schedule; `SPLIT_SEEDS` selects approximate-solver runs.

The paper's few-shot scratch study spans 20 learned models and all 11 benchmark
collections; each script invocation runs the selected model across collections.
Fine-tuning covers `tu_COX2`, `tu_DHFR`, `tu_PROTEINS`, `tu_IMDB-BINARY`
and `ogb_ogbg-code2`.
