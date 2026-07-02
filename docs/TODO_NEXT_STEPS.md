# TODO Next Steps

## P0：立即处理

- [ ] 提交当前清理：`__pycache__/` 删除和 `.gitignore` 更新。
- [ ] 准备真实 skeleton `.npy/.npz` 数据，格式为 `[T,V,C]`，文件名对应官方 sample key，例如 `18_67.npy`。
- [ ] 使用真实 skeleton 数据运行 1 epoch `temporal_mlp` smoke test。
- [ ] 使用真实 skeleton 数据运行 1 epoch `stgcn` smoke test。
- [ ] 确认是否已有真实 RGB frames、C3D/I3D feature 或其他 video feature。

## P1：工程规范

- [ ] 新增 `requirements.txt` 或 `environment.yml`。
- [ ] 在 `train.py` 日志中增加 `input_type`、`model`、`input_dim`、`train_size`、`test_size`。
- [ ] 增加一个小型数据检查脚本，验证 official split、feature path、skeleton path。
- [ ] 如果还需要 synthetic skeleton smoke test，改为脚本生成 synthetic 数据，不再提交大量 `.npy`。
- [ ] 检查 `runs/` 中哪些实验产物需要长期保留，哪些只作为临时 smoke test。

## P2：baseline 实验

- [ ] 完整运行真实 skeleton temporal MLP baseline。
- [ ] 完整运行真实 skeleton ST-GCN-style baseline。
- [ ] 准备真实 video feature baseline。
- [ ] 对比 annotation fallback、video feature、skeleton temporal MLP、skeleton ST-GCN-style 的 Spearman/MSE。
- [ ] 增加多 seed 实验，至少 seed 0/1/2。

## P3：官方复现

- [ ] 下载或准备 MTL-AQA 视频帧。
- [ ] 准备官方 C3D 预训练权重：`c3d.pickle` 或 `C3D_small_PyTorch_Trained_12.pth`。
- [ ] 配置 `MTL-AQA_code_release/opts.py` 中的路径。
- [ ] 解决官方代码硬编码 `.cuda()` 的问题，或配置可用 CUDA 环境。
- [ ] 跑通官方 C3D-AVG 1 epoch smoke test。
- [ ] 跑通官方 MSCADC 1 epoch smoke test。

## P4：科研扩展

- [ ] 将当前 numpy MLP baseline 迁移到 PyTorch。
- [ ] 实现真正 trainable temporal encoder / TCN。
- [ ] 实现真正 trainable ST-GCN。
- [ ] 复现 Joint Relation Graphs 相关骨架建模方法。
- [ ] 准备 FineDiving 第二阶段实验。
