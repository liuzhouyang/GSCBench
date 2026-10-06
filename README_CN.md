# GSCBench

[English](README.md) | **简体中文**

[![Python 3.9–3.12](https://img.shields.io/badge/Python-3.9--3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) [![PyG 2.6.1](https://img.shields.io/badge/PyG-2.6.1-3C2179)](https://pytorch-geometric.readthedocs.io/) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging_Face-GSCBench-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/ush3r/GSCbench)

GSCBench 使用精确图编辑距离（GED）评估图相似度模型，研究同集合、跨集合、零样本、少样本从头训练和微调。仓库还提供用于评估的近似 GED 求解器。

[📖 引用](#引用) · [✨ 项目亮点](#项目亮点) · [🧭 代码结构](#代码结构) · [📚 支持的基线模型](#支持的基线模型) · [🚀 安装与快速开始](#安装与快速开始) · [📦 数据](#数据) · [🧪 复现论文实验](#复现论文实验) · [🧪 单次实验](#单次实验) · [🔧 修改实验设置](#修改实验设置)

## 📖 引用

> [!IMPORTANT]
> 如果 GSCBench 对你的研究有帮助，请引用我们的论文：
> **[Revisiting the Generalization of Neural Graph Edit Distance Models](https://arxiv.org/abs/2610.04644)**。

```bibtex
@article{liu2026revisiting,
  title={Revisiting the Generalization of Neural Graph Edit Distance Models},
  author={Liu, Zhouyang and Liu, Ning and Chen, Yixin and He, Jiezhong and Li, Dongsheng},
  journal={arXiv preprint arXiv:2610.04644},
  year={2026},
  url={https://arxiv.org/abs/2610.04644}
}
```

## ✨ 项目亮点

- **精确 GED：** 11 个集合包含 1,903,452 个精确图对，并提供图对象和每个图对的一个最优映射。
- **泛化实验：** 同集合、跨集合、零样本、少样本从头训练和微调。
- **模型比较：** 提供 20 个学习模型配置；Hungarian、VJ、Beam 和 FGWAlign 用于近似求解器评估。
- **可直接使用的数据：** 从 Hugging Face 下载 PyTorch 数据包或 JSONL 文件。

## 🧭 代码结构

```text
├── configs/
│   ├── experiments/   # 实验协议、划分和运行配置
│   ├── models/        # benchmark 对模型默认参数的覆盖
│   └── datasets/      # 数据目录和 Hugging Face 设置
├── gscbench/
│   ├── core/          # 图对接口、训练器、指标和结果
│   ├── data/          # 数据加载、图划分和图对采样
│   ├── models/<model>/ # 模型实现、包装器和适配器
│   ├── runners/       # 实验设置、训练、评估和结果写入
│   └── solvers/ged/   # 近似 GED 求解器
├── scripts/
│   ├── run_experiment.py # 单集合实验或 checkpoint 评估
│   ├── run_pretrain.py   # 跨集合、少样本和微调
│   └── *.sh              # 论文实验运行计划
├── docs/              # 实验协议
└── experiments/       # 实验输出
```

Python 入口读取配置和命令行参数，并交给 runner。Runner 加载数据集、构建指定模型和适配器，执行训练或评估并保存结果。模型源代码和作者默认参数位于 `gscbench/models/<model>/`；benchmark 覆盖参数位于 `configs/models/`。

当前学习模型配置：`egsc`、`eric`、`gedgnn`、`gediot`、`gedranker`、`gelato`、`gen`、`genn_astar`、`gmn`、`gotsim`、`graph2region`、`graphedx`、`graphsim`、`grasp`、`greed`、`h2mn`、`mata`、`noah`、`simgnn`、`tagsim`。近似求解器评估支持 `hungarian`、`vj`、`beam`、`fgwalign`。新增模型时，在 `gscbench/models/<model>/` 添加实现和适配器，在 `gscbench/runners/bootstrap.py` 的 `MODEL_SPECS` 登记，并将 benchmark 覆盖参数放入 `configs/models/<model>.yaml`。

## 📚 支持的基线模型

GSCBench 为以下方法提供可运行配置和 benchmark 适配器。论文链接指向
论文页面，代码链接指向作者的上游仓库；GSCBench 中对应的实现位于
`gscbench/models/`。

| `--model` 参数 | 论文标题 | Paper | 会议/期刊 | 代码 |
| --- | --- | --- | --- | --- |
| `gmn` | Graph Matching Networks for Learning the Similarity of Graph Structured Objects | [Link](https://proceedings.mlr.press/v97/li19d.html) | ICML 2019 | [代码](https://github.com/google-deepmind/deepmind-research/tree/master/graph_matching_networks) |
| `simgnn` | SimGNN: A Neural Network Approach to Fast Graph Similarity Computation | [Link](https://arxiv.org/abs/1808.05689) | WSDM 2019 | [代码](https://github.com/benedekrozemberczki/SimGNN) |
| `graphsim` | Learning-based Efficient Graph Similarity Computation via Multi-Scale Convolutional Set Matching | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/5720) | AAAI 2020 | [代码](https://github.com/yunshengb/GraphSim) |
| `egsc` | Slow Learning and Fast Inference: Efficient Graph Similarity Computation via Knowledge Distillation | [Link](https://papers.nips.cc/paper/2021/hash/75fc093c0ee742f6dddaa13fff98f104-Abstract.html) | NeurIPS 2021 | [代码](https://github.com/canqin001/Efficient_Graph_Similarity_Computation) |
| `genn_astar` | Combinatorial Learning of Graph Edit Distance via Dynamic Embedding | [Link](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Combinatorial_Learning_of_Graph_Edit_Distance_via_Dynamic_Embedding_CVPR_2021_paper.html) | CVPR 2021 | [代码](https://github.com/Thinklab-SJTU/GENN-Astar) |
| `gotsim` | Interpretable Graph Similarity Computation via Differentiable Optimal Alignment of Node Embeddings | [Link](https://doi.org/10.1145/3404835.3462960) | SIGIR 2021 | [代码](https://github.com/khoadoan/GraphOTSim) |
| `h2mn` | H2MN: Graph Similarity Learning with Hierarchical Hypergraph Matching Networks | [Link](https://doi.org/10.1145/3447548.3467272) | KDD 2021 | [代码](https://github.com/cszhangzhen/H2MN) |
| `noah` | Noah: Neural-optimized A* Search Algorithm for Graph Edit Distance Computation | [Link](https://doi.org/10.1109/ICDE51399.2021.00056) | ICDE 2021 | [代码](https://github.com/pkumod/Noah-GED) |
| `eric` | Efficient Graph Similarity Computation with Alignment Regularization | [Link](https://arxiv.org/abs/2406.14929) | NeurIPS 2022 | [代码](https://github.com/JhuoW/ERIC) |
| `greed` | GREED: A Neural Framework for Learning Graph Distance Functions | [Link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8d492b8a6201d83d1015af9e264f0bf2-Abstract-Conference.html) | NeurIPS 2022 | [代码](https://github.com/idea-iitd/greed) |
| `tagsim` | TaGSim: Type-aware Graph Similarity Learning and Computation | [Link](https://doi.org/10.14778/3489496.3489513) | PVLDB 2022 | [代码](https://github.com/jiyangbai/TaGSim) |
| `gedgnn` | Computing Graph Edit Distance via Neural Graph Matching | [Link](https://doi.org/10.14778/3594512.3594514) | PVLDB 2023 | [代码](https://github.com/ChengzhiPiao/GEDGNN) |
| `mata` | MATA*: Combining Learnable Node Matching with A* Algorithm for Approximate Graph Edit Distance Computation | [Link](https://doi.org/10.1145/3583780.3614959) | CIKM 2023 | [代码](https://github.com/jfkey/mata) |
| `graphedx` | Graph Edit Distance with General Costs Using Neural Set Divergence | [Link](https://proceedings.neurips.cc/paper_files/paper/2024/hash/860e5b214c842eaedaa6b4026ee91aac-Abstract-Conference.html) | NeurIPS 2024 | [代码](https://github.com/structlearning/GraphEdX) |
| `gediot` | Computing Approximate Graph Edit Distance via Optimal Transport | [Link](https://doi.org/10.1145/3709673) | SIGMOD 2025 (PACMMOD 3(1)) | [代码](https://github.com/chengqihao/ged-via-optimal-transport) |
| `gedranker` | Towards Unsupervised Training of Matching-based Graph Edit Distance Solver via Preference-aware GAN | [Link](https://arxiv.org/abs/2506.01977) | NeurIPS 2025 | [代码](https://github.com/piupiupiuu/GEDRanker) |
| `graph2region` | Graph2Region: Efficient Graph Similarity Learning with Structure and Scale Restoration | [Link](https://arxiv.org/abs/2510.00394) | arXiv 2025 | [代码](https://github.com/liuzhouyang/Graph2Region) |
| `grasp` | GraSP: Simple yet Effective Graph Similarity Predictions | [Link](https://doi.org/10.1609/aaai.v39i21.34450) | AAAI 2025 | [代码](https://github.com/HaoranZ99/GraSP) |
| `gelato` | Gelato: Graph Edit Distance via Autoregressive Neural Combinatorial Optimization | [Link](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1943190d92efee1fccfc8d6579b0f8d9-Abstract-Conference.html) | ICLR 2026 | [代码](https://github.com/BorgwardtLab/Gelato) |
| `gen` | Rethinking Flexible Graph Similarity Computation: One-Step Alignment with Global Guidance | [Link](https://doi.org/10.1109/ICDE65706.2026.00111) | ICDE 2026 | [代码](https://github.com/liuzhouyang/GEN) |

Hungarian、VJ、Beam 和 FGWAlign 另列为近似求解器评估方法，不属于学习模型训练配置。

## 🚀 安装与快速开始

使用 Python 3.9–3.12、PyTorch 2.5.1 和 PyTorch Geometric 2.6.1。CUDA 12.4 环境示例：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

CPU 或其他 CUDA 版本请安装匹配的 [PyTorch](https://docs.pytorch.org/get-started/previous-versions/) 和 [PyG](https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html)。安装后，在项目目录运行一次 SimGNN：

```bash
python -m scripts.run_experiment --config configs/experiments/in_collection.yaml \
  --model simgnn --dataset-name tu_MUTAG
```

命令模板见[单次实验](#单次实验)，论文实验计划见[复现论文实验](#复现论文实验)。

## 📦 数据

GSCBench 的 11 个 benchmark 集合共包含 5,523 张图和 1,903,452 个精确图对：

| 集合 | 图数 | 精确图对数 |
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

仓库另提供 `ged_pyg_LINUX`（89 张图、3,916 个图对）作为辅助预训练数据，不计入以上 benchmark 集合。

数据托管在 [ush3r/GSCbench](https://huggingface.co/datasets/ush3r/GSCbench)。默认配置会在首次使用某集合时自动下载，也可手动下载：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ush3r/GSCbench",
    repo_type="dataset",
    local_dir="datasets",
)
```

每个集合包含 `data.pt`、`graphs.jsonl`、`ged_pairs.jsonl` 和 `graph_id_map.csv`；两个 JSONL 文件可以重建 `data.pt`。数据不绑定固定划分，实验配置根据划分比例和 `split_seed` 生成。字段、GED 定义和数据来源见[数据卡](https://huggingface.co/datasets/ush3r/GSCbench)。评估目标为 raw GED；回归指标包括 MSE、MAE、RMSE，排序指标包括 Kendall、Spearman 和 P@10。划分及 `num_iters`、梯度累计和 GraphEdX 行为见[实验协议](docs/experiment_protocol.md)。

## 🧪 复现论文实验

以下 Bash 脚本按论文设置运行所选模型，每个脚本使用 `configs/experiments/` 中对应的配置。脚本需要 Bash；单次运行可使用下一节的 Python 命令。

| 实验 | 脚本 | 配置 |
| --- | --- | --- |
| 同集合 | `in_collection.sh` | `in_collection.yaml` |
| 跨集合 | `cross_collection.sh` | `cross_collection.yaml` |
| 零样本 | `zeroshot.sh` | `zeroshot.yaml` |
| 少样本从头训练 | `fewshot_scratch.sh` | `fewshot_scratch.yaml` |
| 微调 | `fewshot_finetune.sh` | `fewshot_finetune.yaml` |
| 近似求解器评估 | `approximate.sh` | `approximate.yaml` |

例如，运行 SimGNN 在 TU-MUTAG 上的同集合实验计划：

```bash
bash scripts/in_collection.sh simgnn tu_MUTAG
```

其他脚本参数：

```bash
bash scripts/cross_collection.sh <model_name>
bash scripts/zeroshot.sh <model_name> <source_dataset> <target_dataset> <checkpoint> <split_seed> <seed>
bash scripts/fewshot_scratch.sh <model_name> [dataset_name]
bash scripts/fewshot_finetune.sh <model_name> <target_dataset> <pretrained_checkpoint>
bash scripts/approximate.sh <method> <dataset_name>
```

同集合和少样本从头训练脚本对所选模型运行五组 split/seed。论文少样本实验覆盖 20 个学习模型、全部 11 个集合及 100、500、2,000 三种精确训练图对预算。微调目标为 COX2、DHFR、PROTEINS、IMDB-BINARY 和 ogbg-code2，需要源 checkpoint。跨集合训练的源集合和目标集合由配置指定；零样本评估需要源 checkpoint。微调从源权重初始化并使用新优化器。近似求解器仅用于评估。更多运行细节见 [`scripts/README.md`](scripts/README.md)。

## 🧪 单次实验

每条 Python 命令运行一个实验。选择实验配置，再指定模型和集合：

```bash
python -m scripts.run_experiment --config configs/experiments/in_collection.yaml \
  --model <model_name> --dataset-name <dataset_name>
```

例如，运行一次 SimGNN 在 TU-MUTAG 上的实验：

```bash
python -m scripts.run_experiment --config configs/experiments/in_collection.yaml \
  --model simgnn --dataset-name tu_MUTAG
```

跨集合训练、少样本从头训练和微调用 `scripts.run_pretrain`。例如，使用 100 个 TU-MUTAG 精确训练图对从头训练 SimGNN：

```bash
python -m scripts.run_pretrain --config configs/experiments/fewshot_scratch.yaml \
  --model simgnn --train-datasets tu_MUTAG --train-pair-budget 100
```

`--seed`、`--split-seed`、`--epochs` 等命令行参数会覆盖所选配置中的对应值。

## 🔧 修改实验设置

需要持续使用自定义设置时，复制最接近的 `configs/experiments/` 文件并修改：

| 设置 | 位置 |
| --- | --- |
| 实验类型、数据集合、划分比例、split seed、训练 seed、pair budget | `configs/experiments/<experiment>.yaml` |
| epoch、batch size、梯度累计、优化器、评估参数 | 配置中的 `runtime_config` |
| 模型超参数 | `configs/models/<model>.yaml` |
| 数据目录和 Hugging Face 仓库 | `configs/datasets/default.yaml` |

直接使用 `GSCDataset` 时默认划分为 60% / 20% / 20%；论文配置为 80% / 4% / 16%。修改 `val_ratio` 和 `test_ratio` 可指定其他比例。Pair budget 限制 train-train 精确图对数量。`gscbench/models/` 保留模型作者默认参数；benchmark 覆盖参数放在 `configs/models/`。

每次运行保存在 `experiments/<experiment>/<model>/<collection-or-run-name>/runs/<run_id>/`。`result.json` 包含指标和 checkpoint 引用，`config.yaml` 记录本次运行参数，训练任务的 checkpoint 位于 `checkpoints/`。命令行摘要显示主测试指标，并在可用时分别显示 D-Q 和 Q-Q 指标。配置说明见 [`configs/README.md`](configs/README.md)。
