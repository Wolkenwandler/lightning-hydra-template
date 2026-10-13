# 用本模板创建新项目

本文件写给人，也写给智能体：按下面的步骤执行，就能把模板导入到一个新项目或已有项目中。

- 模板仓库：<https://github.com/Wolkenwandler/lightning-hydra-template>，下文用 `<模板>` 表示它的本地 clone 路径。
- 导入工具：`<模板>/scripts/new_project.py`。只用 Python 标准库，Python >= 3.9 即可运行，不需要先装项目依赖。

## 选择分支

| 分支 | 训练框架 | 适合 |
|---|---|---|
| `main` | PyTorch Lightning | 想用现成的功能（callback、多种分布式策略），少写训练代码 |
| `accelerate` | Hugging Face Accelerate 加上精简的训练循环 `src/trainer.py` | 想完全掌控训练循环，或者要和 Hugging Face 生态（transformers 等）配合 |

两个分支的配置结构、输出目录、记录方式（csv 加离线 wandb）和导入方式相同。导入脚本复制的是**运行它的那份 checkout** 里的文件。要从模板主 clone 当前没有检出的分支导入时，不要切换主 clone 的分支，用一个临时 worktree：

```bash
git -C <模板> worktree add /tmp/template-accelerate accelerate
python3 /tmp/template-accelerate/scripts/new_project.py <目标目录>
git -C <模板> worktree remove /tmp/template-accelerate
```

## 快速导入

```bash
python3 <模板>/scripts/new_project.py <目标目录> --dry-run    # 先看会复制哪些文件
python3 <模板>/scripts/new_project.py <目标目录> [--name 项目名]
```

脚本会做这些事：

1. 复制模板里所有被 git 跟踪的文件，下面“不复制的文件”除外。
2. 把项目名写进 `pyproject.toml`、`environment.yaml`、wandb 的 `project` 字段和 README 标题。项目名默认取目标目录名。
3. 在目标目录写一个 `.template-source`，记录模板的远程地址、分支和 commit，以后同步模板更新时要用到。
4. **不覆盖任何已有文件**：只要有一个文件会被覆盖，脚本就什么都不复制，并列出这些冲突文件。
5. 模板有未提交的修改时直接拒绝执行，保证 `.template-source` 里记录的 commit 和实际复制的代码一致。

## 智能体执行步骤

1. 确认 `<模板>` 是哪个 clone，并检查它的状态：`git -C <模板> status --short`，再用 `git -C <模板> log -1` 确认是想要的版本。
2. 如果目标目录是已有的 git 仓库，先运行 `git status --short` 记录基线。不要删除或覆盖用户已有的文件。
3. 运行 `--dry-run`，确认文件清单，然后正式导入。
4. 有冲突时，例如已有 `README.md` 或 `pyproject.toml`：不要删除用户的文件来绕过冲突。把冲突列表报告给用户，确认后再手动合并。
5. 目标目录不是 git 仓库时，不要自动 `git init`，由用户决定。
6. 按下面的“导入后要改什么”逐项修改，然后在服务器上验证，见“验证”一节。
7. 提交时按路径明确暂存，不要用 `git add .`：暂存 `--dry-run` 列出的文件，再加上 `.template-source`。

## 复制哪些文件

| 类别 | 文件 | 导入后 |
|---|---|---|
| 训练框架 | `src/trainer.py`、`src/train.py`、`src/eval.py`、`src/utils/`、`src/__init__.py`、`scripts/launch.sh`、`configs/train.yaml`、`configs/eval.yaml`、`configs/__init__.py` | 一般不改 |
| 配置组 | `configs/{debug,extras,hydra,logger,paths,trainer}/`、`configs/local/.gitkeep` | 按需调整参数 |
| MNIST 示例 | `configs/data/mnist.yaml`、`configs/model/mnist.yaml`、`configs/experiment/example.yaml`、`src/data/mnist_datamodule.py`、`src/models/mnist_module.py`、`src/models/components/simple_dense_net.py`、`tests/test_datamodules.py` | 参照它写自己的数据和模型，写好后删掉 |
| 测试 | `tests/conftest.py`、`tests/helpers/`、`tests/test_{configs,train,eval,sweeps}.py`、`tests/__init__.py` | 保留，用来检查框架没被改坏 |
| 工程 | `pyproject.toml`、`environment.yaml`、`Makefile`、`.pre-commit-config.yaml`、`.github/workflows/ci.yml`、`.gitignore`、`.env.example` | 添加自己的依赖 |
| 目录占位 | `.project-root`（rootutils 靠它定位项目根目录，**必须保留**）、`data/.gitkeep`、`logs/.gitkeep` | 不改 |
| 文档和许可证 | `README.md`、`LICENSE`（MIT 协议，**保留**） | 重写 README 开头的项目介绍，使用说明部分可以保留 |

为什么也复制 MNIST 示例：入口脚本、默认配置和测试都依赖一套能跑的 data 和 model。保留示例，导入后的项目立刻就能训练、测试也能通过，之后再逐步替换。

**不复制的文件**，它们只属于模板本身：

- `TEMPLATE.md`：本文件。
- `scripts/new_project.py`：导入工具。
- `docs/solutions.md`：模板自己的问题记录。其中通用的结论已经写进 README 的 “Adding Your Own Model and Data” 一节；新项目第一次需要记录问题时，再自己新建这个文件。

## 导入后要改什么

1. **数据**：新建 `src/data/<名字>_datamodule.py` 和 `configs/data/<名字>.yaml`，提供 `prepare_data`、`setup` 和三个 `*_dataloader` 方法。配置里的 `batch_size` 是全局 batch size，多进程时会平均分给各进程。
2. **模型**：新建 `src/models/<名字>_module.py`，网络结构放在 `src/models/components/`；再新建 `configs/model/<名字>.yaml`。照 `MNISTModule` 实现训练循环需要的三个方法：
   - `forward(batch)`：返回 `"loss"`（用于反向传播的标量），以及逐样本的张量，比如预测和标签。训练循环会把逐样本张量从所有进程收集起来。
   - `compute_metrics(outputs)`：用一个 epoch 的逐样本张量计算指标。
   - `configure_optimizers()`：返回优化器和可选的学习率调度器配置。工厂对象经过 `plain_partial(...)` 再保存，否则用 Hydra 1.4 时模型无法 pickle。
3. **默认配置**：把 `configs/train.yaml` 和 `configs/eval.yaml` 里的 `data: mnist`、`model: mnist` 换成新的配置；在 `configs/experiment/` 下写自己的实验配置。
4. **执行白名单**：配置里新出现的 `_target_` 所在的包，要加进 `src/utils/hydra_utils.py` 的 `EXECUTION_WHITELIST`，比如 `"torchvision.transforms.*"`。
5. **测试**：`tests/test_datamodules.py` 改成测试自己的 DataModule。`test_train`、`test_eval` 和 `test_sweeps` 用的是默认配置，第 3 步改完后会自动测新模型。
6. **删除示例**：上面几步都完成后，删掉 MNIST 相关文件，并检查 `configs/` 里不再引用 `mnist`。
7. **依赖**：新依赖同时加到 `pyproject.toml` 和 `environment.yaml`。
8. **设备**：默认用 CUDA，没有 CUDA 时直接报错，要用 CPU 就设置 `trainer.cpu=true`（`configs/trainer/default.yaml`）。多卡用 `scripts/launch.sh` 启动，命令见 README。其他设备，比如 Mac 的 mps，各开一个分支维护，不要改成自动选择设备。
9. **记录**：默认把指标写到 csv 和离线 wandb。wandb 项目名已经改成新项目名；训练完用 `wandb sync` 手动上传，命令见 README。

## 验证

在服务器上执行，不在本地执行。**优先用目标机器上已有的环境，不急于新建环境**：

1. **先检查已有环境**（例如容器的默认 Python）里的关键包：

   ```bash
   python -c "import sys, torch, accelerate, hydra; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available(), accelerate.__version__, hydra.__version__)"
   python -c "import omegaconf, rootutils, rich, colorlog, wandb, pytest"   # 缺哪个包就会报错
   ```

2. **直接在已有环境里跑**。模板靠 `rootutils` 定位项目根目录，不需要 `pip install -e .`：

   ```bash
   pytest -v                              # 包括 slow 测试
   python src/train.py trainer.max_epochs=1
   python src/train.py debug=default
   ```

   不要在共享环境里运行 `pip install -e .`：它会按 `pyproject.toml` 升级已有的包，影响其他用这个环境的人。`sh` 只有 `tests/test_sweeps.py` 用到，缺少时这几个测试会自动跳过，可以直接运行 `python src/train.py -m ...` 来补测。

3. **缺包或版本不够时**，不要装进共享环境，而是在它上面建一个叠加环境，只补缺的包：

   ```bash
   python -m venv --system-site-packages <叠加环境目录>
   <叠加环境目录>/bin/pip install <缺少的包>
   ```

4. **以上都不行，或者项目需要长期独立维护环境时**，再新建环境：`conda env create -f environment.yaml`。

已有环境里的版本低于 `pyproject.toml` 的下限时，只要测试通过就可以先用。但要在实验记录里写明实际版本。

已经完整验证过的环境组合：

| Python | torch | accelerate | Hydra | 说明 |
|---|---|---|---|---|
| 3.12 | 2.13.0 | 1.15.0 | 1.3.7 / 1.4.0rc1 | conda-forge 新建的环境，加叠加环境补 accelerate |
| 3.11 | 2.8.0+cu128 | 0.34.2 | 1.3.2 | 容器自带的环境，加叠加环境把 accelerate 0.33 换成 0.34.2；没有 `sh`，扫参改为直接运行 |
| 3.10 / 3.12 | 最新 CPU 版 | 最新 | 最新 1.3.x | GitHub CI |

accelerate 0.33 及更早的版本配 torch >= 2.6 时，训练、评估和多卡都能跑，但**无法断点续训**（`load_state` 报 `KeyError: 'step'`），见 `docs/solutions.md` 的 SOL-20261013-03。环境里是这些版本时，用上面的叠加环境补一个 >= 0.34 的 accelerate。

## 同步模板的后续更新

模板以后还会更新，比如依赖升级或修复。可以对照 `.template-source` 里记录的 commit，把更新合进来：

```bash
C=$(sed -n 's/^commit: //p' .template-source)
BR=$(sed -n 's/^branch: //p' .template-source)   # 导入时的模板分支，如 main 或 accelerate
git -C <模板> log --oneline $C..$BR                       # 有哪些更新
git -C <模板> diff $C..$BR -- . ':!TEMPLATE.md' ':!scripts/new_project.py' ':!docs' > /tmp/template.diff
git apply --check /tmp/template.diff && git apply /tmp/template.diff
```

- 如果项目里改过的文件和更新冲突，`--check` 会失败。这时逐个文件对照 diff 手动合并，或者用 `git apply --reject` 处理能应用的部分，再手动处理 `.rej` 文件。
- 涉及项目名的那几行（见“快速导入”第 2 点）在新项目里已经被替换过，合并时保留新项目的名字。
- 合并并验证后，把 `.template-source` 里的 `commit` 更新为 `git -C <模板> rev-parse $BR`，和这次合并放在同一个提交里。
