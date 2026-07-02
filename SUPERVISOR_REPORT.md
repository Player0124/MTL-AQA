# 当前项目进度汇报

## 1. 项目目标

本阶段项目围绕 skeleton-based action quality assessment / action recognition 展开。近期目标是基于 MTL-AQA 数据集建立一个最小可复现闭环：

```text
dataset -> feature/input -> model -> score prediction -> Spearman correlation / MSE
```

后续目标是在保持官方 split、final score 回归目标和评价指标不变的前提下，将输入从 RGB/video feature 替换为 skeleton sequence，并进一步接入 temporal encoder 或 ST-GCN 类模型。

## 2. 当前已完成工作

目前已经完成以下工作：

1. 克隆并保留 MTL-AQA 官方代码作为起点。
2. 审计了官方代码结构，包括 C3D-AVG、MSCADC、dataloader、官方 split 和 annotation 文件。
3. 确认官方 split 文件可用：
   - MTL-AQA 总样本数：1412
   - train：1059
   - test：353
4. 新增了一个最小 single-task score regression baseline。
5. 新增了统一的 `train.py` / `test.py` / `dataset.py` / `model.py` / `metrics.py`。
6. 实现了 final score 预测、MSE、Spearman rank correlation、checkpoint 保存、metrics.json 和 predictions.csv 输出。
7. 新增 skeleton input replacement 接口，支持 skeleton `.npy/.npz`，格式为 `[T,V,C]`。
8. 实现了两个 skeleton baseline 接口：
   - `temporal_mlp`
   - `stgcn` 风格固定图编码
9. 编写了复现文档和 skeleton baseline 文档：
   - `README_reproduction.md`
   - `README_skeleton_baseline.md`
   - `results_skeleton_baseline.md`

## 3. 当前代码与实验状态

当前核心代码位于项目根目录：

- `dataset.py`：负责读取官方 split、annotation、feature、skeleton。
- `model.py`：实现 numpy MLP regressor。
- `metrics.py`：实现 Spearman 和 MSE。
- `train.py`：训练入口，支持 CLI/config、checkpoint、日志、metrics、predictions。
- `test.py`：加载 checkpoint 并在 test split 上评估。
- `config.yaml`：保存默认配置。

当前代码支持两类输入：

1. `video_feature`
   - 当前没有真实 video feature，因此默认使用 annotation metadata fallback。
2. `skeleton`
   - 已实现数据接口，但当前仓库没有真实 skeleton 文件。
   - 目前只使用 synthetic skeleton 做了 smoke test。

当前运行环境：

- Python 3.12.13
- numpy 2.3.5
- scipy 1.18.0
- PyYAML 6.0.3
- torch 2.12.1+cpu
- torchvision 0.27.1+cpu
- CUDA 当前不可用：`torch.cuda.is_available() = False`

因此，当前新增的 numpy baseline 可以运行，PyTorch/torchvision/scipy 也已可 import。但由于当前安装的是 CPU 版 PyTorch，且官方代码中大量硬编码 `.cuda()`，官方 PyTorch/CUDA pipeline 仍不能直接运行。

## 4. 当前实验结果

当前可验证的主实验结果如下：

| Experiment | Input | Model | Split | Epochs | Spearman | MSE | 说明 |
|---|---|---|---:|---:|---:|---:|---|
| `runs/mtl_aqa_minimal` | annotation metadata fallback | MLP | 1059/353 | 20 | 0.3680757962 | 666.235943 | simplified single-task baseline；`spearman_p=9.1107e-13` |
| `runs/skeleton_temporal_mlp_smoke` | synthetic skeleton | temporal_mlp + MLP | 1059/353 | 2 | 0.0786188914 | 4651.189384 | 仅验证 skeleton 接口 |
| `runs/skeleton_stgcn_smoke` | synthetic skeleton | ST-GCN-style + MLP | 1059/353 | 2 | 0.1560733323 | 4641.265818 | 仅验证 skeleton 接口 |

其中 `runs/mtl_aqa_minimal` 是当前唯一较完整的 20 epoch 训练结果。该实验使用官方 train/test split，输出包括：

- `runs/mtl_aqa_minimal/checkpoints/best_model.pt`
- `runs/mtl_aqa_minimal/logs/train.log`
- `runs/mtl_aqa_minimal/outputs/metrics.json`
- `runs/mtl_aqa_minimal/outputs/predictions.csv`

需要特别说明：该结果不是官方 video-feature baseline，也不是 skeleton baseline。它使用 annotation metadata fallback，主要用于验证训练、评估和输出闭环。

## 5. 与原论文复现目标的差距

论文中 C3D-AVG-MTL 报告的 rank correlation 为 90.44%。当前主实验 Spearman 为 0.3681，差距较大。

主要原因是当前实验设置与论文不同：

1. 当前没有使用真实 RGB/video frame 或 C3D feature。
2. 当前没有运行官方 C3D-AVG-MTL 多任务模型。
3. 当前没有使用 dive classification 和 captioning 多任务损失。
4. 当前模型是 simplified single-task MLP baseline。
5. 当前 skeleton 实验还没有真实 skeleton 数据，只做了 synthetic smoke test。

因此，目前不能声称已经完成官方论文复现，只能说明最小 score regression pipeline 已建立。

## 6. 当前问题

当前主要问题如下：

1. 官方代码仍不能直接运行。
   - `opts.py` 中存在路径占位符。
   - 官方脚本硬编码 `.cuda()`。
   - 当前 PyTorch 为 CPU 版，CUDA 不可用。
   - 当前仓库缺少 extracted RGB frames。
   - 当前仓库缺少直接可加载的 C3D 权重。

2. 当前主结果不是 video-based AQA 结果。
   - `runs/mtl_aqa_minimal` 使用 annotation metadata fallback。
   - 该结果只能作为 pipeline sanity check。

3. skeleton baseline 尚未完成真实实验。
   - skeleton dataset 接口已实现。
   - temporal_mlp / stgcn-style 路径已 smoke test。
   - 但真实 skeleton `.npy` 文件尚未接入。

4. 当前项目缺少标准依赖文件。
   - 未找到 `requirements.txt`。
   - 未找到 `environment.yml`。

## 7. 下一步计划

近期计划：

- 准备最小真实 skeleton `.npy` 数据，验证真实 skeleton loader。
- 跑通真实 skeleton temporal MLP 的 1 epoch smoke test。
- 跑通真实 skeleton ST-GCN-style 的 1 epoch smoke test。
- 补充最小依赖文件和环境说明。
- 在日志中增加 input_dim、input_type、model 等关键信息。

中期计划：

- 准备完整 MTL-AQA skeleton feature。
- 完成完整 skeleton temporal MLP baseline。
- 完成完整 skeleton ST-GCN-style baseline。
- 准备真实 video feature baseline，与 skeleton baseline 对比。
- 尝试配置官方 PyTorch 环境，运行官方 C3D-AVG baseline 的 1 epoch smoke test。

后续计划：

- 实现真正 trainable ST-GCN 或 TCN encoder。
- 复现 Action Assessment by Joint Relation Graphs 的核心思想。
- 将 MTL-AQA video feature baseline 与 skeleton baseline 做系统对比。
- 准备 FineDiving 第二阶段实验。

## 8. 需要确认或支持的事项

当前需要进一步确认或准备：

1. 是否已有可用的 MTL-AQA 视频帧或离线 C3D/I3D feature。
2. 是否已有可用的 skeleton `.npy` 文件，或需要先做 pose extraction。
3. 后续是否优先复现官方 C3D-AVG-MTL，还是优先推进 skeleton baseline。
4. 是否需要迁移到 PyTorch，以便后续实现真正的 TCN/ST-GCN。
5. 是否需要建立统一实验记录表，包括多 seed、均值、方差和置信区间。

总体而言，当前项目已经完成了最小复现闭环和 skeleton 接口验证，但官方 baseline 与真实 skeleton 实验仍未完成。下一阶段重点应放在真实输入数据准备和可比较 baseline 实验上。
