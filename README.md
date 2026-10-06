# GSCBench

<p align="center">
  <a href="README.md"><strong>English</strong></a> |
  <a href="README_CN.md"><strong>简体中文</strong></a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9--3.12-3776AB?logo=python&logoColor=white" alt="Python 3.9 to 3.12"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.5.1"></a>
  <a href="https://pytorch-geometric.readthedocs.io/"><img src="https://img.shields.io/badge/PyG-2.6.1-3C2179" alt="PyTorch Geometric 2.6.1"></a>
  <a href="https://huggingface.co/datasets/ush3r/GSCbench"><img src="https://img.shields.io/badge/Hugging_Face-GSCBench-FFD21E?logo=huggingface&logoColor=black" alt="GSCBench dataset"></a>
</p>

GSCBench benchmarks graph similarity models against exact graph edit distance
(GED). It evaluates generalization within a collection, across collections,
without target training, and after few-shot training or fine-tuning. Approximate
GED solvers are included for evaluation.

[📖 Citation](#citation) · [✨ Highlights](#highlights) · [🧭 Code structure](#repository-structure) · [📚 Supported baselines](#supported-baselines) · [🚀 Installation and quick start](#installation-and-quick-start) · [📦 Data](#data) · [🧪 Reproduce the paper](#reproduce-the-paper-experiments) · [🧪 Run one experiment](#run-one-experiment) · [🔧 Configure an experiment](#change-experiment-settings)

## 📖 Citation

> [!IMPORTANT]
> If you use GSCBench in your research, please cite our paper:
> **[Revisiting the Generalization of Neural Graph Edit Distance Models](https://arxiv.org/abs/2610.04644)**.

```bibtex
@article{liu2026revisiting,
  title={Revisiting the Generalization of Neural Graph Edit Distance Models},
  author={Liu, Zhouyang and Liu, Ning and Chen, Yixin and He, Jiezhong and Li, Dongsheng},
  journal={arXiv preprint arXiv:2610.04644},
  year={2026},
  url={https://arxiv.org/abs/2610.04644}
}
```

## ✨ Highlights

- **Exact GED benchmark:** 1,903,452 exact graph pairs across 11 collections,
  with graph objects and one optimal mapping per pair.
- **Generalization studies:** in-collection, cross-collection, zero-shot, and
  few-shot from-scratch or fine-tuning experiments.
- **Model comparison:** 20 learned-model configurations share the benchmark's
  graph-pair interface; Hungarian, VJ, Beam, and FGWAlign are available for evaluation.
- **Ready-to-use data:** download collections from Hugging Face as PyTorch
  bundles or portable JSONL files.

## 🧭 Repository structure

```text
├── configs/
│   ├── experiments/   # experiment protocols and split settings
│   ├── models/        # benchmark overrides for model defaults
│   └── datasets/      # Hugging Face repository and local data directory
├── gscbench/
│   ├── core/          # graph-pair interface, trainer, metrics, and results
│   ├── data/          # released graph and GED-pair loading and splits
│   ├── models/<model>/ # model implementation, wrapper, and adapter
│   ├── runners/       # experiment setup, training, evaluation, and outputs
│   └── solvers/ged/   # approximate GED methods used for evaluation
├── scripts/
│   ├── run_experiment.py # one collection experiment or checkpoint evaluation
│   ├── run_pretrain.py   # cross-collection, few-shot, and fine-tuning runs
│   └── *.sh              # paper run schedules
├── docs/              # experiment protocol
└── experiments/       # run outputs
```

The Python entry points pass the selected config and command-line options to
the runners. The runner loads the collection, builds the selected model and
adapter, then trains or evaluates it and writes the run results. Model source
and its original defaults live together under `gscbench/models/<model>/`;
benchmark-specific model settings live under `configs/models/`.

The available learned-model configs are `egsc`, `eric`, `gedgnn`, `gediot`,
`gedranker`, `gelato`, `gen`, `genn_astar`, `gmn`, `gotsim`, `graph2region`,
`graphedx`, `graphsim`, `grasp`, `greed`, `h2mn`, `mata`, `noah`, `simgnn`,
and `tagsim`. Approximate evaluation supports `hungarian`, `vj`, `beam`, and `fgwalign`.

To evaluate another model, put its implementation and GSCBench adapter under
`gscbench/models/<model>/`, add it to `MODEL_SPECS` in
`gscbench/runners/bootstrap.py`, and put any benchmark overrides in
`configs/models/<model>.yaml`. Then pass that model name to the experiment
command above.

## 📚 Supported baselines

GSCBench includes runnable configurations and benchmark adapters for the
following methods. The first column lists the exact `--model` values. Paper links lead to the publications; code links
lead to the authors' upstream repositories. The GSCBench implementations are
in `gscbench/models/`.

| `--model` key | Title | Paper | Venue | Upstream code |
| --- | --- | --- | --- | --- |
| `gmn` | Graph Matching Networks for Learning the Similarity of Graph Structured Objects | [Link](https://proceedings.mlr.press/v97/li19d.html) | ICML 2019 | [Code](https://github.com/google-deepmind/deepmind-research/tree/master/graph_matching_networks) |
| `simgnn` | SimGNN: A Neural Network Approach to Fast Graph Similarity Computation | [Link](https://arxiv.org/abs/1808.05689) | WSDM 2019 | [Code](https://github.com/benedekrozemberczki/SimGNN) |
| `graphsim` | Learning-based Efficient Graph Similarity Computation via Multi-Scale Convolutional Set Matching | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/5720) | AAAI 2020 | [Code](https://github.com/yunshengb/GraphSim) |
| `egsc` | Slow Learning and Fast Inference: Efficient Graph Similarity Computation via Knowledge Distillation | [Link](https://papers.nips.cc/paper/2021/hash/75fc093c0ee742f6dddaa13fff98f104-Abstract.html) | NeurIPS 2021 | [Code](https://github.com/canqin001/Efficient_Graph_Similarity_Computation) |
| `genn_astar` | Combinatorial Learning of Graph Edit Distance via Dynamic Embedding | [Link](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Combinatorial_Learning_of_Graph_Edit_Distance_via_Dynamic_Embedding_CVPR_2021_paper.html) | CVPR 2021 | [Code](https://github.com/Thinklab-SJTU/GENN-Astar) |
| `gotsim` | Interpretable Graph Similarity Computation via Differentiable Optimal Alignment of Node Embeddings | [Link](https://doi.org/10.1145/3404835.3462960) | SIGIR 2021 | [Code](https://github.com/khoadoan/GraphOTSim) |
| `h2mn` | H2MN: Graph Similarity Learning with Hierarchical Hypergraph Matching Networks | [Link](https://doi.org/10.1145/3447548.3467272) | KDD 2021 | [Code](https://github.com/cszhangzhen/H2MN) |
| `noah` | Noah: Neural-optimized A* Search Algorithm for Graph Edit Distance Computation | [Link](https://doi.org/10.1109/ICDE51399.2021.00056) | ICDE 2021 | [Code](https://github.com/pkumod/Noah-GED) |
| `eric` | Efficient Graph Similarity Computation with Alignment Regularization | [Link](https://arxiv.org/abs/2406.14929) | NeurIPS 2022 | [Code](https://github.com/JhuoW/ERIC) |
| `greed` | GREED: A Neural Framework for Learning Graph Distance Functions | [Link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8d492b8a6201d83d1015af9e264f0bf2-Abstract-Conference.html) | NeurIPS 2022 | [Code](https://github.com/idea-iitd/greed) |
| `tagsim` | TaGSim: Type-aware Graph Similarity Learning and Computation | [Link](https://doi.org/10.14778/3489496.3489513) | VLDB 2022 | [Code](https://github.com/jiyangbai/TaGSim) |
| `gedgnn` | Computing Graph Edit Distance via Neural Graph Matching | [Link](https://doi.org/10.14778/3594512.3594514) | VLDB 2023 | [Code](https://github.com/ChengzhiPiao/GEDGNN) |
| `mata` | MATA*: Combining Learnable Node Matching with A* Algorithm for Approximate Graph Edit Distance Computation | [Link](https://doi.org/10.1145/3583780.3614959) | CIKM 2023 | [Code](https://github.com/jfkey/mata) |
| `graphedx` | Graph Edit Distance with General Costs Using Neural Set Divergence | [Link](https://proceedings.neurips.cc/paper_files/paper/2024/hash/860e5b214c842eaedaa6b4026ee91aac-Abstract-Conference.html) | NeurIPS 2024 | [Code](https://github.com/structlearning/GraphEdX) |
| `gediot` | Computing Approximate Graph Edit Distance via Optimal Transport | [Link](https://doi.org/10.1145/3709673) | SIGMOD 2025 | [Code](https://github.com/chengqihao/ged-via-optimal-transport) |
| `gedranker` | Towards Unsupervised Training of Matching-based Graph Edit Distance Solver via Preference-aware GAN | [Link](https://arxiv.org/abs/2506.01977) | NeurIPS 2025 | [Code](https://github.com/piupiupiuu/GEDRanker) |
| `graph2region` | Graph2Region: Efficient Graph Similarity Learning with Structure and Scale Restoration | [Link](https://arxiv.org/abs/2510.00394) | IEEE TKDE 2025 | [Code](https://github.com/liuzhouyang/Graph2Region) |
| `grasp` | GraSP: Simple yet Effective Graph Similarity Predictions | [Link](https://doi.org/10.1609/aaai.v39i21.34450) | AAAI 2025 | [Code](https://github.com/HaoranZ99/GraSP) |
| `gelato` | Gelato: Graph Edit Distance via Autoregressive Neural Combinatorial Optimization | [Link](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1943190d92efee1fccfc8d6579b0f8d9-Abstract-Conference.html) | ICLR 2026 | [Code](https://github.com/BorgwardtLab/Gelato) |
| `gen` | Rethinking Flexible Graph Similarity Computation: One-Step Alignment with Global Guidance | [Link](https://doi.org/10.1109/ICDE65706.2026.00111) | ICDE 2026 | [Code](https://github.com/liuzhouyang/GEN) |

Hungarian, VJ, Beam, and FGWAlign are available separately for approximate-solver
evaluation; they are not learned-model training configurations.

## 🚀 Installation and quick start

Use Python 3.9–3.12, PyTorch 2.5.1, and PyTorch Geometric 2.6.1. Install
the PyTorch build for your hardware, then install the remaining dependencies.
For CUDA 12.4:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

For CPU or another CUDA build, use the matching [PyTorch](https://docs.pytorch.org/get-started/previous-versions/)
and [PyG](https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html)
packages. From the project directory, start one SimGNN run with:

```bash
python -m scripts.run_experiment --config configs/experiments/in_collection.yaml \
  --model simgnn --dataset-name tu_MUTAG
```

See [Run one experiment](#run-one-experiment) for the command template and
[Reproduce the paper experiments](#reproduce-the-paper-experiments) for the
paper schedules.

## 📦 Data

The benchmark contains 11 collections, 5,523 graphs, and 1,903,452 exact graph
pairs. The repository also provides 89 graphs and 3,916 pairs from
`ged_pyg_LINUX` for auxiliary pretraining.

Data are hosted at [ush3r/GSCbench](https://huggingface.co/datasets/ush3r/GSCbench).
The default config downloads a collection the first time it is used. To
download the repository yourself:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ush3r/GSCbench",
    repo_type="dataset",
    local_dir="datasets",
)
```

The 11 benchmark collections are:

| Collection | Graphs | Exact pairs |
| --- | ---: | ---: |
| `tu_AIDS` | 1,488 | 1,106,325 |
| `tu_BZR` | 404 | 37,781 |
| `tu_COX2` | 465 | 78,100 |
| `tu_DHFR` | 719 | 124,714 |
| `tu_ENZYMES` | 117 | 6,286 |
| `tu_IMDB-BINARY` | 280 | 22,703 |
| `tu_MUTAG` | 175 | 15,220 |
| `tu_NCI1` | 992 | 379,032 |
| `tu_PROTEINS` | 364 | 64,010 |
| `tu_PTC_MR` | 328 | 52,706 |
| `ogb_ogbg-code2` | 191 | 16,575 |

The repository also includes `ged_pyg_LINUX` (89 graphs and 3,916 pairs) as
auxiliary pretraining data; it is separate from the benchmark collections.
Each collection contains `data.pt`, `graphs.jsonl`, `ged_pairs.jsonl`, and
`graph_id_map.csv`. The two JSONL files can rebuild `data.pt`. Splits are
generated by each experiment config from its ratios and `split_seed`; the
released data do not prescribe a fixed split. See the [dataset card](https://huggingface.co/datasets/ush3r/GSCbench)
and [experiment protocol](docs/experiment_protocol.md).

Evaluation uses raw GED. Regression results include MSE, MAE, and RMSE; ranking
results include Kendall, Spearman, and P@10. The experiment protocol describes
the pair groups and ranking query construction.

## 🧪 Reproduce the paper experiments

The Bash scripts run the paper schedules for one model. Each script uses its
matching config in `configs/experiments/`. They require Bash; Python module
commands below run one setting directly.

| Experiment | Question | Script and config |
| --- | --- | --- |
| In-collection | How well does a model generalize to unseen graphs from its training collection? | `in_collection.sh` · `in_collection.yaml` |
| Cross-collection | How does a model trained on source collections perform on held-out target collections? | `cross_collection.sh` · `cross_collection.yaml` |
| Zero-shot | How does a trained source model perform on a target collection without target training? | `zeroshot.sh` · `zeroshot.yaml` |
| Few-shot from scratch | How well can a model learn from a small exact-pair budget on the target collection? | `fewshot_scratch.sh` · `fewshot_scratch.yaml` |
| Fine-tuning | How much does target training improve a source-pretrained model? | `fewshot_finetune.sh` · `fewshot_finetune.yaml` |
| Approximate solver evaluation | How do approximate GED solvers compare with the exact GED labels? | `approximate.sh` · `approximate.yaml` |

For example, reproduce the in-collection schedule for SimGNN on TU-MUTAG:

```bash
bash scripts/in_collection.sh simgnn tu_MUTAG
```

The other schedules take these arguments:

```bash
bash scripts/cross_collection.sh <model_name>
bash scripts/zeroshot.sh <model_name> <source_dataset> <target_dataset> <checkpoint> <split_seed> <seed>
bash scripts/fewshot_scratch.sh <model_name> [dataset_name]
bash scripts/fewshot_finetune.sh <model_name> <target_dataset> <pretrained_checkpoint>
bash scripts/approximate.sh <method> <dataset_name>
```

The in-collection and few-shot scratch scripts run five split/seed settings
for the selected model. The paper's few-shot scratch study spans 20 learned
models and all 11 benchmark collections, with budgets 100, 500, and 2,000 exact
training pairs. Fine-tuning covers COX2, DHFR, PROTEINS,
IMDB-BINARY, and ogbg-code2 and requires a pretrained checkpoint. Cross-
collection training uses the source and target collections in its config;
zero-shot evaluation requires a source checkpoint. Fine-tuning starts from a
source checkpoint and trains on the target pair budget. Use the checkpoint
produced by the intended source run for zero-shot or fine-tuning; the
fine-tuning run starts with a fresh optimizer. Approximate solvers are
evaluation-only. Script arguments and schedules are listed in
[`scripts/README.md`](scripts/README.md).

## 🧪 Run one experiment

Each Python command runs one experiment. Choose the experiment config, then
pass the model and collection:

```bash
python -m scripts.run_experiment --config configs/experiments/in_collection.yaml \
  --model <model_name> --dataset-name <dataset_name>
```

For example, run one SimGNN experiment on TU-MUTAG:

```bash
python -m scripts.run_experiment --config configs/experiments/in_collection.yaml \
  --model simgnn --dataset-name tu_MUTAG
```

Use `scripts.run_pretrain` for cross-collection training, few-shot scratch, or
fine-tuning. For example, train SimGNN from scratch on 100 TU-MUTAG pairs:

```bash
python -m scripts.run_pretrain --config configs/experiments/fewshot_scratch.yaml \
  --model simgnn --train-datasets tu_MUTAG --train-pair-budget 100
```

Command-line options such as `--seed`, `--split-seed`, and `--epochs` override
the selected config for that run.

## 🔧 Change experiment settings

Copy the closest file in `configs/experiments/` when you want persistent
changes. Update the relevant fields there:

| Change | Config location |
| --- | --- |
| Experiment type, collections, split ratios, split seed, training seed, pair budget | `configs/experiments/<experiment>.yaml` |
| Epochs, batch size, gradient accumulation, optimizer, evaluation | The config's `runtime_config` section |
| Model hyperparameters | `configs/models/<model>.yaml` |
| Dataset directory and Hugging Face repository | `configs/datasets/default.yaml` |

Direct use of `GSCDataset` defaults to 60% / 20% / 20%. The supplied paper
configs set 80% / 4% / 16%; change `val_ratio` and `test_ratio` in the
experiment config to use another split. Pair budgets select exact
train-train pairs. Model files under `gscbench/models/` retain the authors'
defaults; benchmark-specific overrides belong in `configs/models/`.

Each run writes to
`experiments/<experiment>/<model>/<collection-or-run-name>/runs/<run_id>/`.
`result.json` contains the metrics and checkpoint reference, `config.yaml`
records the run settings, and training runs save checkpoints under
`checkpoints/`. The terminal summary reports the main test metrics and, when
available, separate D-Q and Q-Q metrics. See the
[experiment protocol](docs/experiment_protocol.md) for batch size,
`num_iters`, gradient accumulation, and GraphEdX behavior.

For exact run schedules see [`scripts/README.md`](scripts/README.md); for
configuration roles see [`configs/README.md`](configs/README.md).
