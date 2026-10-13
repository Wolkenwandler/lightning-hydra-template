# Lightning-Hydra-Template

[![python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/Lightning-2.6.6+-792ee5?logo=lightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3+-89b8cd)](https://hydra.cc/)
[![ruff](https://img.shields.io/badge/Code%20Style-Ruff-261230)](https://docs.astral.sh/ruff/)

A compact template for training deep learning models with
[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [Hydra](https://hydra.cc/).
It ships a small MNIST example that you replace with your own data and model.

The default trainer uses CUDA. Settings for other devices (e.g. `accelerator: mps` on a Mac)
are kept on a separate branch per device.

Based on [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) (MIT),
trimmed down and updated to current versions.

To start a new project from this template, or to import it into an existing one, see
[TEMPLATE.md](TEMPLATE.md) (written for both people and coding agents).

## Project Structure

```
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks (checkpoint, early stopping, ...)
│   ├── data                     <- Datamodules
│   ├── debug                    <- Debugging setup
│   ├── experiment               <- Experiment configs (versioned hyperparameters)
│   ├── extras                   <- Extra utilities (tags, config printing)
│   ├── hydra                    <- Hydra settings (output dirs, logging)
│   ├── local                    <- Machine-specific configs (git-ignored)
│   ├── logger                   <- csv / tensorboard / wandb
│   ├── model                    <- LightningModules
│   ├── paths                    <- Project paths
│   ├── trainer                  <- default (1 CUDA GPU) / ddp
│   ├── train.yaml               <- Main config for training
│   └── eval.yaml                <- Main config for evaluation
├── data                      <- Datasets (git-ignored)
├── logs                      <- Hydra and Lightning outputs (git-ignored)
├── src
│   ├── data                     <- LightningDataModules
│   ├── models                   <- LightningModules and nets
│   ├── utils                    <- Utility scripts
│   ├── train.py                 <- Training entry point
│   └── eval.py                  <- Evaluation entry point
├── tests                     <- pytest tests
├── environment.yaml          <- Conda environment
└── pyproject.toml            <- Dependencies and tool config (pytest, ruff)
```

## Setup

With conda (recommended, all packages from conda-forge):

```bash
conda env create -f environment.yaml
conda activate lightning-hydra-template
pre-commit install   # optional: run ruff and other checks on every commit
```

Or with pip in an existing environment (install PyTorch for your platform first,
see <https://pytorch.org/get-started/locally/>):

```bash
pip install -e ".[dev]"
```

## Training

```bash
# train with the default config (1 CUDA GPU)
python src/train.py

# multi-GPU with DDP, or CPU
python src/train.py trainer=ddp trainer.devices=4
python src/train.py trainer.accelerator=cpu

# run an experiment config from configs/experiment/
python src/train.py experiment=example

# override any parameter from the command line
python src/train.py trainer.max_epochs=20 data.batch_size=64 model.optimizer.lr=0.0005

# mixed precision, gradient clipping, ...
python src/train.py +trainer.precision=16-mixed +trainer.gradient_clip_val=0.5

# resume from a checkpoint
python src/train.py ckpt_path="/path/to/ckpt/name.ckpt"

# log to tensorboard instead, or turn off metric logging
python src/train.py logger=tensorboard
python src/train.py "logger=[]"
```

Outputs (checkpoints, logs, the resolved config) go to
`logs/<task_name>/runs/<timestamp>/`.
Metrics are logged to `csv/version_0/metrics.csv` and to an offline wandb run in `wandb/`
inside that directory; nothing is uploaded during training. Upload runs afterwards with:

```bash
wandb login   # once per machine
wandb sync logs/train/runs/<timestamp>/wandb/offline-run-*
wandb sync logs/train/multiruns/<timestamp>/*/wandb/offline-run-*   # all jobs of a multirun
```

The wandb project is set in `configs/logger/wandb.yaml`; set `logger.wandb.offline=false` to
log online directly. Each run first asks for tags if none are set
(`extras.enforce_tags`); set `tags=[...]` in the experiment config or on the command line.

## Evaluation

```bash
python src/eval.py ckpt_path="/path/to/ckpt/name.ckpt"
```

## Sweeps

Hydra's built-in multirun runs every combination one after the other:

```bash
python src/train.py -m data.batch_size=32,64,128 model.optimizer.lr=0.001,0.0005
```

Multirun outputs go to `logs/<task_name>/multiruns/<timestamp>/<job_number>/`.

## Debugging

```bash
# 1 epoch on CPU, no callbacks/loggers, DEBUG log level, anomaly detection
python src/train.py debug=default

# 1 train/val/test batch only
python src/train.py debug=default +trainer.fast_dev_run=true

# overfit to a few batches, or profile
python src/train.py debug=default trainer.max_epochs=20 +trainer.overfit_batches=3
python src/train.py debug=default +trainer.profiler=simple
```

## Adding Your Own Model and Data

1. Write a `LightningDataModule` in `src/data/` and a config in `configs/data/`.
2. Write a `LightningModule` (and nets) in `src/models/` and a config in `configs/model/`.
3. Add an experiment config in `configs/experiment/` that combines them.
4. Run `python src/train.py experiment=<name>`.

Keep `save_hyperparameters(...)` to plain values (numbers, strings, lists): pass
`ignore=[...]` for modules and optimizer/scheduler factories, so that checkpoints load
with `torch.load(weights_only=True)` (default since PyTorch 2.6).
See `src/models/mnist_module.py` and [docs/solutions.md](docs/solutions.md).

Hydra 1.4 only instantiates `_target_`s listed in `EXECUTION_WHITELIST`
(`src/utils/hydra_utils.py`; currently `src.*`, `lightning.pytorch.*` and `torch.optim.*`), so add
the package of any other target there, e.g. `"torchvision.transforms.*"`. Pass `_partial_` factories
through `plain_partial(...)` before storing them on a `LightningModule`, as `MNISTLitModule` does,
so that the module stays picklable for `ddp_spawn`.

Values from other config groups can be shared through interpolation, e.g.
`some_param: ${data.some_param}` in a model config.

Machine-specific settings (paths, devices) go in `configs/local/default.yaml` and secrets in
`.env` (see `.env.example`); both are git-ignored. `.env` is loaded automatically and can be
referenced as `${oc.env:MY_VAR}`.

## Tests and Code Quality

```bash
make test        # pytest, skipping tests marked as slow
make test-full   # all tests
make format      # pre-commit run -a (ruff lint + format, codespell, ...)
```

CI (`.github/workflows/ci.yml`) runs pre-commit and pytest on Python 3.10 and 3.12.
It also runs every Monday against the latest dependency releases, so upstream breaking
changes show up without new commits (GitHub pauses scheduled runs after 60 days without
repository activity; re-enable them in the Actions tab).

## License

MIT, see [LICENSE](LICENSE). Keep `LICENSE` in projects created from this template.
