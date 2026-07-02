# 当前项目进度汇报

## 1. 项目目标

本阶段项目目标是围绕 MTL-AQA 数据集建立 action quality assessment 的最小可复现闭环：

```text
dataset -> feature/input -> model -> score prediction -> Spearman correlation / MSE
```

后续目标是在保持官方 split、final score 回归目标和评价指标不变的前提下，将输入从 RGB/video feature 替换为 skeleton sequence，并接入 temporal encoder 或 ST-GCN 类模型。

## 2. 当前已完成工作

目前已完成：

1. 保留 MTL-AQA 官方代码目录 `MTL-AQA_code_release/` 作为参考实现。
2. 确认官方 MTL-AQA split 可读取：
   - annotations：1412
   - train：1059
   - test：353
3. 实现了一个最小 single-task score regression baseline：
   - `dataset.py`
   - `model.py`
   - `metrics.py`
   - `train.py`
   - `test.py`
   - `config.yaml`
4. 已完成 final score 预测、MSE、Spearman rank correlation、checkpoint、metrics.json、predictions.csv 输出。
5. 已接入 skeleton input replacement 接口，支持 skeleton `.npy/.npz`，格式为 `[T,V,C]`。
6. 已实现两个 skeleton 输入路径：
   - `temporal_mlp`
   - `stgcn` 风格固定图编码
7. 已清理不必要临时文件：
   - `get-pip.py` 已删除。
   - `runs/synthetic_skeleton_smoke/` 原始合成 `.npy` 已删除。
   - `__pycache__/` 当前已删除，并处于 git 删除状态。

## 3. 当前代码与实验状态

当前核心代码位于项目根目录：

| 文件 | 作用 |
|---|---|
| `dataset.py` | 读取官方 split、annotation/feature、skeleton |
| `model.py` | numpy MLP regressor |
| `metrics.py` | Spearman 和 MSE |
| `train.py` | 训练、日志、checkpoint、metrics、predictions |
| `test.py` | checkpoint 复评估 |
| `config.yaml` | 默认配置 |

当前配置默认为：

```text
input_type: video_feature
feature_mode: annotation
model: mlp
epochs: 20
batch_size: 64
lr: 0.001
seed: 0
output_dir: runs/mtl_aqa_minimal
```

需要说明：虽然参数名为 `video_feature`，当前由于没有真实 video feature，实际主实验使用的是 annotation metadata fallback。

## 4. 当前实验结果

### 4.1 主实验结果

当前唯一较完整的 20 epoch 主实验：

| Experiment | Input | Model | Epochs | Spearman | MSE | 说明 |
|---|---|---|---:|---:|---:|---|
| `runs/mtl_aqa_minimal` | annotation metadata fallback | MLP | 20 | 0.3680757962 | 666.235943 | simplified single-task baseline |

主实验日志最后一轮：

```text
epoch=020
train_loss=745.417859
train_spearman=0.33519831450036197
train_mse=693.854579
test_spearman=0.3680757961967368
test_mse=666.235943
```

主实验 `metrics.json`：

```text
spearman = 0.36807579619673686
spearman_p = 9.110719287371563e-13
mse = 666.2359429149961
```

主实验输出文件：

```text
runs/mtl_aqa_minimal/checkpoints/best_model.pt
runs/mtl_aqa_minimal/logs/train.log
runs/mtl_aqa_minimal/outputs/metrics.json
runs/mtl_aqa_minimal/outputs/predictions.csv
```

其中 `predictions.csv` 有 353 行，对应 test split 样本数。

### 4.2 其他 smoke test

| Experiment | Input | Model | Epochs | Spearman | MSE | 说明 |
|---|---|---|---:|---:|---:|---|
| `runs/smoke_test` | annotation metadata fallback | MLP | 1 | -0.1974734800 | 4365.446187 | 训练路径 smoke test |
| `runs/regression_video_smoke` | annotation fallback via `video_feature` | MLP | 1 | -0.1974734800 | 4365.446187 | video_feature 路径 smoke test |
| `runs/skeleton_temporal_mlp_smoke` | synthetic skeleton | temporal_mlp + MLP | 2 | 0.0786188914 | 4651.189384 | skeleton 接口验证，不是真实 skeleton 结果 |
| `runs/skeleton_stgcn_smoke` | synthetic skeleton | ST-GCN-style + MLP | 2 | 0.1560733323 | 4641.265818 | skeleton 接口验证，不是真实 skeleton 结果 |

注意：synthetic skeleton 原始输入目录已经清理，因此这两个 skeleton smoke test 当前不能直接复跑，除非重新生成 synthetic 数据或提供真实 skeleton 数据。

## 5. 与原论文复现目标的差距

当前项目尚未完成官方 baseline reproduction。

论文中 C3D-AVG-MTL 报告的 rank correlation 为 90.44%。当前主实验 Spearman 为 0.3681，但两者不能直接比较，原因是：

1. 当前输入不是 RGB frame 或 C3D feature，而是 annotation metadata fallback。
2. 当前模型是 simplified MLP single-task regressor。
3. 当前没有复现官方多任务结构，也没有使用 dive classification / captioning loss。
4. 官方 C3D-AVG / MSCADC pipeline 尚未跑通。
5. skeleton 实验目前只有 synthetic smoke test，没有真实 skeleton 输入结果。

因此当前应定位为：

```text
simplified single-task baseline / partial reproduction
```

不应表述为 official baseline reproduction。

## 6. 当前问题

当前主要问题如下：

1. 官方 pipeline 未跑通。
   - `MTL-AQA_code_release/opts.py` 中仍有 `'...'` 路径占位。
   - 官方代码大量硬编码 `.cuda()`。
   - 缺少 extracted RGB frames。
   - 缺少可直接加载的 `c3d.pickle` 或 `C3D_small_PyTorch_Trained_12.pth`。

2. 当前主结果不是 video-based AQA 性能。
   - `runs/mtl_aqa_minimal` 使用 annotation metadata fallback。
   - 该结果只能证明最小训练/评估闭环跑通。

3. skeleton 方向尚未完成真实实验。
   - skeleton dataset 与 encoder 接口已实现。
   - 真实 skeleton `.npy/.npz` 尚未找到。
   - 现有 skeleton 结果只是 synthetic smoke test。

4. 工程规范仍需补充。
   - 未找到 `requirements.txt`。
   - 未找到 `environment.yml`。
   - `train.py` 日志尚未显式记录 input_dim。

5. Git 当前还有缓存文件删除处于变更状态。
   - `__pycache__/*.pyc` 已删除。
   - 建议提交清理；`.gitignore` 已包含 `__pycache__/`。

## 7. 下一步计划

近期计划：

- 准备真实 skeleton `.npy`，格式 `[T,V,C]`，命名如 `18_67.npy`。
- 使用真实 skeleton 跑通 `temporal_mlp` 1 epoch smoke test。
- 使用真实 skeleton 跑通 `stgcn` 1 epoch smoke test。
- 增加 `requirements.txt` 或 `environment.yml`。
- 在训练日志中增加 input_dim、input_type、model、train/test size。
- 如需保留 synthetic skeleton 测试，应改为脚本生成，不再提交大量 `.npy`。

中期计划：

- 完整运行真实 skeleton temporal MLP baseline。
- 完整运行真实 skeleton ST-GCN-style baseline。
- 准备真实 video feature baseline，与 skeleton baseline 对比。
- 配置官方 `opts.py`，尝试 official C3D-AVG 1 epoch smoke test。

后续计划：

- 将 numpy baseline 迁移到 PyTorch。
- 实现真正 trainable TCN / ST-GCN。
- 复现 Joint Relation Graphs 相关方法。
- 准备 FineDiving 第二阶段实验。

## 8. 需要确认或支持的事项

需要进一步确认：

1. 是否已有 MTL-AQA 的视频帧或离线 C3D/I3D feature。
2. 是否已有真实 skeleton `.npy/.npz` 数据。
3. 后续优先目标是官方 C3D baseline 复现，还是优先推进 skeleton baseline。
4. 是否需要配置 CUDA 版 PyTorch 以运行官方 `.cuda()` 代码。
5. 是否需要建立统一实验记录表和多 seed 结果统计。

当前阶段的客观结论是：项目已经跑通 MTL-AQA 最小 score regression 闭环，但 official baseline 和真实 skeleton baseline 尚未完成。
