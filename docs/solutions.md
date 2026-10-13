# Solutions

## SOL-20261013-01

- 状态：pending
- 日期：2026-10-13
- 分类：models / checkpoint 兼容性
- 症状：PyTorch >= 2.6 下 `trainer.test(ckpt_path=...)`、断点续训和 `eval.py` 报 `_pickle.UnpicklingError: Weights only load failed ... Unsupported global: GLOBAL src.models.components.simple_dense_net.SimpleDenseNet`。
- 根因：PyTorch 2.6 起 `torch.load` 默认 `weights_only=True`。`LightningModule.save_hyperparameters()` 把 `net`（`nn.Module`）以及 Hydra `_partial_` 生成的 optimizer/scheduler 工厂写进了 checkpoint 的 `hyper_parameters`；DataModule 的 hparams 里还有 OmegaConf `ListConfig`。这些对象都不在安全反序列化白名单内。
- 解决方案：`save_hyperparameters(logger=False, ignore=["net", "optimizer", "scheduler"])`，并把工厂另存为 `self.optimizer_factory` / `self.scheduler_factory`；在 data 配置加 `_convert_: all`，让 Hydra 传入普通 list。不要用 `weights_only=False` 绕过。新增模型或 DataModule 时，hparams 里只放基本类型。
- 验证：服务器上 `pytest -v`（含 slow），其中 `test_train_eval` 检查从 checkpoint 重新加载后 test acc 与训练结束时一致（误差 < 0.001），`test_train_resume` 检查续训生效。

## SOL-20261013-02

- 状态：pending
- 日期：2026-10-13
- 分类：hydra / 依赖升级兼容性
- 症状：Hydra 1.4.0rc1 + OmegaConf 2.4.0 下，`ddp_spawn` 训练（`test_train_ddp_sim`、`test_hydra_sweep_ddp_sim`）报 `TypeError: Hydra _partial_ factories cannot be pickled before invocation`；每次 `instantiate` 警告 `resolved _target_=... with no _execution_whitelist_ ... will become an error in Hydra 1.5`。
- 根因：Hydra 1.4 引入 execution whitelist：`_target_` 在 `instantiate` 时检查白名单（没有白名单时 1.4 警告、1.5 报错）；`_partial_: true` 返回的 `_DeferredTarget` 故意禁止 pickle，而 `ddp_spawn` 要 pickle 整个 `LightningModule`，模块上保存的 optimizer / scheduler 工厂随之失败。
- 解决方案：`src/utils/hydra_utils.py` 定义代码内的 `EXECUTION_WHITELIST`，`train.py` / `eval.py` 通过 `@hydra.main(**hydra_main_kwargs())` 传入（Hydra 1.3 没有此参数，自动跳过），测试用 autouse fixture 套 `hydra.utils.execution_whitelist(...)`；模块里用 `plain_partial(...)` 把已通过白名单检查的工厂转成普通 `functools.partial`。新增 `_target_` 时把其包加入白名单；白名单不能从配置读取。
- 验证：服务器上分别用 Hydra 1.3.7 和 1.4.0rc1（OmegaConf 2.4.0）跑完整 `pytest -v`（含 slow），以及 CUDA 训练、debug、eval、无 GPU 的 sweep 测试，并确认 `-W error::UserWarning` 下没有白名单警告。
