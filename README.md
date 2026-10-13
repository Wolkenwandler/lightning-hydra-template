# Lightning-Hydra-Template

[![python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![accelerate](https://img.shields.io/badge/Accelerate-0.34+-ffd21e?logo=huggingface&logoColor=black)](https://huggingface.co/docs/accelerate)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3+-89b8cd)](https://hydra.cc/)
[![ruff](https://img.shields.io/badge/Code%20Style-Ruff-261230)](https://docs.astral.sh/ruff/)

**`accelerate` branch:** a compact template for training deep learning models with
[Hugging Face Accelerate](https://huggingface.co/docs/accelerate) and [Hydra](https://hydra.cc/),
without PyTorch Lightning. The `main` branch is the same template with PyTorch Lightning.
It ships a small MNIST example that you replace with your own data and model.

The training loop (`src/trainer.py`, about 350 lines with comments) covers mixed precision, gradient accumulation
and clipping, best/last checkpoints, resuming, early stopping and metric logging; everything else
is plain PyTorch that you can read and change.

The default trainer uses CUDA. Settings for other devices (e.g. mps on a Mac) are kept on a
separate branch per device.

Based on [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) (MIT),
trimmed down and updated to current versions.

To start a new project from this template, or to import it into an existing one, see
[TEMPLATE.md](TEMPLATE.md) (written for both people and coding agents).

## Project Structure

```
├── configs                   <- Hydra configs
│   ├── data                     <- Data modules
│   ├── debug                    <- Debugging setup
│   ├── experiment               <- Experiment configs (versioned hyperparameters)
│   ├── extras                   <- Extra utilities (tags, config printing)
│   ├── hydra                    <- Hydra settings (output dirs, logging)
│   ├── local                    <- Machine-specific configs (git-ignored)
│   ├── logger                   <- csv / wandb
│   ├── model                    <- Task models
│   ├── paths                    <- Project paths
│   ├── trainer                  <- Training loop settings (device, precision, checkpoints, ...)
│   ├── train.yaml               <- Main config for training
│   └── eval.yaml                <- Main config for evaluation
├── data                      <- Datasets (git-ignored)
├── logs                      <- Run outputs (git-ignored)
├── scripts
│   └── launch.sh                <- Multi-process (multi-GPU) launcher
├── src
│   ├── data                     <- Data modules
│   ├── models                   <- Task models and nets
│   ├── utils                    <- Utility scripts (loggers, config printing, ...)
│   ├── trainer.py               <- Training loop on top of Accelerate
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

# train on the CPU
python src/train.py trainer.cpu=true

# several GPUs (DDP): scripts/launch.sh wraps `accelerate launch`
scripts/launch.sh --multi_gpu --num_processes=4 src/train.py

# run an experiment config from configs/experiment/
python src/train.py experiment=example

# override any parameter from the command line
python src/train.py trainer.max_epochs=20 data.batch_size=64 model.optimizer.lr=0.0005

# mixed precision, gradient accumulation and clipping
python src/train.py trainer.mixed_precision=bf16 trainer.gradient_accumulation_steps=4 trainer.gradient_clip_val=0.5

# resume from a checkpoint directory (model, optimizer, scheduler, counters, RNG states)
python src/train.py ckpt_path=logs/train/runs/<timestamp>/checkpoints/last

# turn off metric logging
python src/train.py "logger=[]"
```

`data.batch_size` is the global batch size: every batch is split across processes, so it must be
divisible by the number of processes. Multi-process runs need `scripts/launch.sh` (or
`RUN_TIMESTAMP` set by hand) so that all processes write to the same run directory.

Outputs go to `logs/<task_name>/runs/<timestamp>/`: the resolved config (`.hydra/`), the log
file, and `checkpoints/best` / `checkpoints/last` (directories written by
`Accelerator.save_state`; the best one is chosen by `trainer.checkpoint.monitor`).
Metrics are logged to `csv/metrics.csv` and to an offline wandb run in `wandb/` inside that
directory; nothing is uploaded during training. Upload runs afterwards with:

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
python src/eval.py ckpt_path=logs/train/runs/<timestamp>/checkpoints/best
```

Evaluation loads only the model weights (`model.safetensors`) of the checkpoint directory.

## Sweeps

Hydra's built-in multirun runs every combination one after the other:

```bash
python src/train.py -m data.batch_size=32,64,128 model.optimizer.lr=0.001,0.0005
```

Multirun outputs go to `logs/<task_name>/multiruns/<timestamp>/<job_number>/`.

## Debugging

```bash
# 1 epoch on CPU, no loggers or checkpoints, DEBUG log level, anomaly detection
python src/train.py debug=default

# 1 train/val/test batch only
python src/train.py debug=default trainer.limit_train_batches=1 trainer.limit_val_batches=1 trainer.limit_test_batches=1
```

## Adding Your Own Model and Data

1. Write a data module in `src/data/` (`prepare_data`, `setup`, `{train,val,test}_dataloader`)
   and a config in `configs/data/`.
2. Write a task model (an `nn.Module`) and its nets in `src/models/`, and a config in
   `configs/model/`. The trainer needs three methods, see `MNISTModule`:
   - `forward(batch)` returns `"loss"` plus per-sample tensors (e.g. predictions, targets);
   - `compute_metrics(outputs)` turns the per-sample tensors of an epoch into metrics;
   - `configure_optimizers()` returns the optimizer and an optional scheduler config.
3. Add an experiment config in `configs/experiment/` that combines them.
4. Run `python src/train.py experiment=<name>`.

Hydra 1.4 only instantiates `_target_`s listed in `EXECUTION_WHITELIST`
(`src/utils/hydra_utils.py`; currently `src.*` and `torch.optim.*`), so add the package of any
other target there, e.g. `"torchvision.transforms.*"`. Pass `_partial_` factories through
`plain_partial(...)` before storing them on a model, as `MNISTModule` does, so that the model
stays picklable.

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

The multi-process tests start 2 CPU processes with `torchrun` and hide the GPUs from them.

CI (`.github/workflows/ci.yml`) runs pre-commit and pytest on Python 3.10 and 3.12.
It also runs every Monday against the latest dependency releases, so upstream breaking
changes show up without new commits (GitHub pauses scheduled runs after 60 days without
repository activity; re-enable them in the Actions tab).

## License

MIT, see [LICENSE](LICENSE). Keep `LICENSE` in projects created from this template.
