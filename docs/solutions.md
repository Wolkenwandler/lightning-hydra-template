# Solutions

## SOL-20261013-01

- 状态：pending
- 日期：2026-10-13
- 分类：models / checkpoint 兼容性
- 症状：PyTorch >= 2.6 下 `trainer.test(ckpt_path=...)`、断点续训和 `eval.py` 报 `_pickle.UnpicklingError: Weights only load failed ... Unsupported global: GLOBAL src.models.components.simple_dense_net.SimpleDenseNet`。
- 根因：PyTorch 2.6 起 `torch.load` 默认 `weights_only=True`。`LightningModule.save_hyperparameters()` 把 `net`（`nn.Module`）以及 Hydra `_partial_` 生成的 optimizer/scheduler 工厂写进了 checkpoint 的 `hyper_parameters`；DataModule 的 hparams 里还有 OmegaConf `ListConfig`。这些对象都不在安全反序列化白名单内。
- 解决方案：`save_hyperparameters(logger=False, ignore=["net", "optimizer", "scheduler"])`，并把工厂另存为 `self.optimizer_factory` / `self.scheduler_factory`；在 data 配置加 `_convert_: all`，让 Hydra 传入普通 list。不要用 `weights_only=False` 绕过。新增模型或 DataModule 时，hparams 里只放基本类型。
- 验证：服务器上 `pytest -v`（含 slow），其中 `test_train_eval` 检查从 checkpoint 重新加载后 test acc 与训练结束时一致（误差 < 0.001），`test_train_resume` 检查续训生效。
