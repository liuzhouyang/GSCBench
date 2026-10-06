# Configurations

Choose an experiment config and pass its model and collection on the command
line. Configs in `configs/experiments/` contain the split and run settings.
`configs/datasets/default.yaml` sets the dataset directory and Hugging Face
repository. `configs/models/` contains benchmark overrides to the model
settings under `gscbench/models/`.

`python -m scripts.run_experiment --checkpoint` evaluates a trained model. For
fine-tuning, `python -m scripts.run_pretrain --init-from` initializes a new run
from trained weights.

Edit `runtime_config` inside the selected experiment YAML to change epochs,
batch size, optimization, or evaluation settings. Command-line arguments
override those settings for one run. `--runtime-config path/to/runtime.yaml`
selects a separate runtime file instead.

Without a runtime config, `run_experiment` uses `configs/runtime/default.yaml`;
`run_pretrain` uses the runtime settings in `configs/experiments/cross_collection.yaml`.
