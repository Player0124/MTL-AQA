# TODO Next Steps

## P0：阻塞性任务

- [ ] 准备真实 skeleton `.npy/.npz` 数据，文件名采用 `<video_id>_<sample_id>.npy`，shape 为 `[T,V,C]`。
- [ ] 使用真实 skeleton 数据运行 1 epoch temporal MLP smoke test。
- [ ] 使用真实 skeleton 数据运行 1 epoch ST-GCN-style smoke test。
- [ ] 确认是否已有真实 RGB frames、C3D/I3D feature 或其他 video feature。
- [x] 安装并验证 `scipy`、`PyYAML`、`torch`、`torchvision`。
- [ ] 如果要复现官方 baseline，解决 CUDA/device 问题：当前 `torch 2.12.1+cpu` 可 import，但 `torch.cuda.is_available() = False`。
- [ ] 如果要复现官方 baseline，配置 `MTL-AQA_code_release/opts.py` 中的数据路径。
- [ ] 如果要复现官方 baseline，准备 `c3d.pickle` 或 `C3D_small_PyTorch_Trained_12.pth`。

## P1：近期工程任务

- [ ] 新增 `requirements_minimal.txt`，记录当前 numpy/scipy/PyYAML baseline 依赖。
- [ ] 新增 `environment_official.yml` 或 `requirements_official.txt`，记录官方 pipeline 依赖，并注明当前 PyTorch 为 CPU 版。
- [ ] 在 `train.py` 日志中打印 `input_type`、`model`、`input_dim`、`train_size`、`test_size`。
- [ ] 为 `dataset.py` 增加一个小型 smoke test，验证 official split、feature path、skeleton path。
- [ ] 将 `runs/synthetic_skeleton_smoke/` 明确标注为 synthetic smoke data，避免误用为真实结果。
- [ ] 在 `README_skeleton_baseline.md` 中加入真实 skeleton 文件生成/检查示例。
- [ ] 增加 result table 更新流程，避免手动复制结果。

## P2：baseline 实验任务

- [ ] 完整运行真实 skeleton temporal MLP baseline。
- [ ] 完整运行真实 skeleton ST-GCN-style baseline。
- [ ] 完整运行真实 video feature MLP baseline。
- [ ] 对比 video feature、skeleton temporal MLP、skeleton ST-GCN-style 的 Spearman/MSE。
- [ ] 增加多 seed 实验，至少 seed 0/1/2。
- [ ] 记录每组实验的命令、配置、checkpoint、metrics 和 predictions。

## P3：官方复现任务

- [ ] 下载/准备 MTL-AQA 视频并抽帧。
- [ ] 验证官方 dataloader 能读取帧。
- [ ] 修复官方代码中 CPU/GPU device 管理问题，或在 GPU 环境运行。
- [ ] 跑通官方 C3D-AVG 1 epoch。
- [ ] 跑通官方 MSCADC 1 epoch。
- [ ] 与论文 C3D-AVG-MTL 90.44% rank correlation 做差距分析。

## P4：科研扩展任务

- [ ] 将 numpy MLP baseline 迁移到 PyTorch。
- [ ] 实现真正 trainable temporal encoder / TCN。
- [ ] 实现真正 trainable ST-GCN。
- [ ] 复现 Action Assessment by Joint Relation Graphs。
- [ ] 准备 FineDiving 第二阶段实验。
- [ ] 建立统一实验报告模板，包含 mean/std、多 seed、显著性分析。
