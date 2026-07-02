# 当前项目进度总结汇报

生成日期：2026-07-02  
项目目录：`F:\SkeletonRecognition\MTL-AQA`

## 1. 项目结构审计

### 1.1 当前目录结构摘要

当前项目根目录包含：

```text
F:\SkeletonRecognition\MTL-AQA
  .gitignore
  config.yaml
  dataset.py
  model.py
  metrics.py
  train.py
  test.py
  README.md
  docs/
  MTL-AQA_code_release/
  MTL-AQA_dataset_release/
  runs/
```

当前已清理：

- `get-pip.py`：未找到，已清理。
- `__pycache__/`：当前工作区未找到，但 git 暂存区仍记录其删除。
- `runs/synthetic_skeleton_smoke/`：未找到，合成 skeleton 原始 `.npy` 已清理。

### 1.2 关键文件说明

| 文件/目录 | 作用 | 当前状态 |
|---|---|---|
| `MTL-AQA_code_release/` | 官方 C3D-AVG / MSCADC 代码 | 存在，但官方 pipeline 未跑通 |
| `MTL-AQA_dataset_release/` | 官方 annotation、split、raw annotation | 存在 |
| `dataset.py` | 读取 MTL-AQA split、annotation/feature、skeleton 输入 | 核心实现 |
| `model.py` | numpy MLP regressor | 核心实现 |
| `metrics.py` | Spearman 和 MSE | 核心实现 |
| `train.py` | 训练入口 | 核心实现 |
| `test.py` | checkpoint 复评估入口 | 核心实现 |
| `config.yaml` | 默认配置 | 存在 |
| `docs/README.md` | 文档索引 | 存在 |
| `docs/README_reproduction.md` | MTL-AQA 最小复现说明 | 存在 |
| `docs/README_skeleton_baseline.md` | skeleton baseline 说明 | 存在 |
| `docs/PROJECT_PROGRESS_SUMMARY.md` | 当前项目审计与进度总结 | 存在 |
| `docs/SUPERVISOR_REPORT.md` | 导师汇报稿 | 存在 |
| `docs/TODO_NEXT_STEPS.md` | 下一步计划 | 存在 |
| `runs/` | 实验日志、metrics、predictions、checkpoints | 存在 |
| `.gitignore` | 忽略 pycache、get-pip、synthetic smoke 数据 | 存在 |

### 1.3 当前代码模块划分

| 模块 | 文件 | 当前能力 |
|---|---|---|
| 数据读取 | `dataset.py` | `MTLAQADataset` 支持 annotation fallback / feature；`MTLAQASkeletonDataset` 支持 skeleton `.npy/.npz` |
| skeleton 预处理 | `dataset.py` | 支持 padding、center truncation、坐标归一化、missing joints 置零 |
| skeleton 编码 | `dataset.py` | 支持 `temporal_mlp` 和 `stgcn` 风格固定图编码 |
| 模型 | `model.py` | `NumpyMLPRegressor`，一层 hidden MLP，Adam 更新 |
| 训练 | `train.py` | 支持 CLI/config、seed、日志、checkpoint、metrics、predictions |
| 评估 | `test.py` | 加载 checkpoint，重新评估 test split |
| 指标 | `metrics.py` | `scipy.stats.spearmanr` 优先；无 scipy 时有 fallback；MSE |

### 1.4 临时文件或实验产物

| 路径 | 类型 | 当前处理 |
|---|---|---|
| `runs/mtl_aqa_minimal/` | 主实验输出 | 保留 |
| `runs/smoke_test/` | 1 epoch smoke test | 保留为历史验证 |
| `runs/regression_video_smoke/` | video_feature 路径 smoke test | 保留为历史验证 |
| `runs/skeleton_temporal_mlp_smoke/` | synthetic skeleton smoke test 输出 | 保留日志/结果；原始 synthetic 输入已清理 |
| `runs/skeleton_stgcn_smoke/` | synthetic skeleton smoke test 输出 | 保留日志/结果；原始 synthetic 输入已清理 |
| `runs/synthetic_skeleton_smoke/` | 合成 skeleton 原始输入 | 未找到，已清理 |
| `__pycache__/` | Python 缓存 | 当前未找到，删除已暂存 |

## 2. 目标与当前实现对齐检查

| 模块 | 当前状态 | 证据文件/路径 | 问题 | 结论 |
|---|---|---|---|---|
| MTL-AQA 数据读取 | 已完成 | `dataset.py`; `MTL-AQA_dataset_release/Ready_2_Use/MTL-AQA_split_0_data/` | 仅 annotation/split 可用；无真实 RGB frames | 可读取官方 split 和 label |
| 官方 split | 已完成 | `train_split_0.pkl`, `test_split_0.pkl` | 未发现 split 修改 | train 1059 / test 353 |
| feature / annotation / label | 部分完成 | `MTLAQADataset` | 当前主实验使用 annotation fallback，不是真实 video feature | 最小闭环可用，真实 feature 未完成 |
| skeleton 数据接口 | 部分完成 | `MTLAQASkeletonDataset` | 真实 skeleton `.npy` 未找到；synthetic 输入已清理 | 接口完成，真实实验未完成 |
| 路径参数化 | 已完成 | `config.yaml`, `train.py` | 官方 `opts.py` 仍有 `'...'` 占位 | 新增 baseline 参数化，官方代码未配置 |
| 模型 | 部分完成 | `model.py`, `dataset.py` | MLP 是 simplified baseline；ST-GCN 只是固定编码风格 | 完成 minimal model，未完成 official/ST-GCN 正式模型 |
| official baseline reproduction | 未完成 | `MTL-AQA_code_release/train_test_C3DAVG.py`, `train_test_MSCADC.py` | 缺视频帧、缺可加载 C3D 权重、官方路径占位、官方代码硬编码 `.cuda()` | 尚未完成官方复现 |
| simplified single-task baseline | 已完成 | `runs/mtl_aqa_minimal/` | 输入为 annotation fallback | 已跑通最小闭环 |
| train.py | 已完成 | `train.py`; `runs/*/logs/train.log` | 当前实现为 CPU/numpy MLP | 可训练 |
| test.py | 已完成 | `test.py`; `runs/mtl_aqa_minimal/outputs/metrics.json` | 无单独 `eval.py`，由 `test.py` 承担 | 可评估 |
| Spearman | 已完成 | `metrics.py`; `metrics.json` | 历史 smoke test 的 `spearman_p` 为 null | 主实验已生成 p-value |
| MSE/L2 | 已完成 | `metrics.py`; `metrics.json` | 使用 raw final score MSE | 可用 |
| predictions.csv | 已完成 | `runs/*/outputs/predictions.csv` | 每个 predictions 均 353 行 | 与 test split 数量一致 |
| checkpoint | 已完成 | `runs/*/checkpoints/best_model.pt` | numpy pickle 风格 checkpoint | 可用于当前 test.py |
| train/test leakage | 未发现 | split pkl + `dataset.py` | 未做更深入统计验证 | 初步符合官方 split |

## 3. 数据与 split 审计

已读取：

```text
MTL-AQA_dataset_release/Ready_2_Use/MTL-AQA_split_0_data/final_annotations_dict.pkl
MTL-AQA_dataset_release/Ready_2_Use/MTL-AQA_split_0_data/train_split_0.pkl
MTL-AQA_dataset_release/Ready_2_Use/MTL-AQA_split_0_data/test_split_0.pkl
```

结果：

| 项目 | 数量 |
|---|---:|
| annotations | 1412 |
| train split | 1059 |
| test split | 353 |

示例：

- first train key：`(18, 67)`，`final_score = 94.35`
- first test key：`(26, 4)`，`final_score = 64.5`

## 4. 实验结果提取

### 4.1 主实验：`runs/mtl_aqa_minimal`

日志文件：

```text
runs/mtl_aqa_minimal/logs/train.log
```

配置：

| 项目 | 数值 |
|---|---|
| input | annotation metadata fallback |
| model | MLP |
| epochs | 20 |
| batch size | 64 |
| lr | 0.001 |
| seed | 0 |
| optimizer | `model.py` 中自实现 Adam |
| loss | MSE |
| device | `cpu-numpy` |
| split | official train/test split 1059/353 |

epoch 20 日志：

```text
train_loss=745.417859
train_spearman=0.33519831450036197
train_mse=693.854579
test_spearman=0.3680757961967368
test_mse=666.235943
```

`metrics.json`：

```json
{
  "spearman": 0.36807579619673686,
  "spearman_p": 9.110719287371563e-13,
  "mse": 666.2359429149961
}
```

输出文件：

| 文件 | 状态 |
|---|---|
| `runs/mtl_aqa_minimal/checkpoints/best_model.pt` | 存在，13163 bytes |
| `runs/mtl_aqa_minimal/logs/train.log` | 存在 |
| `runs/mtl_aqa_minimal/outputs/metrics.json` | 存在 |
| `runs/mtl_aqa_minimal/outputs/predictions.csv` | 存在，353 行 |

### 4.2 其他实验 / smoke test

| Experiment | Input | Model | Split | Epochs | Spearman | MSE/L2 | Checkpoint | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| `runs/mtl_aqa_minimal` | annotation metadata fallback | MLP | 1059/353 | 20 | 0.3680757962 | 666.235943 | `runs/mtl_aqa_minimal/checkpoints/best_model.pt` | simplified single-task baseline |
| `runs/smoke_test` | annotation metadata fallback | MLP | 1059/353 | 1 | -0.1974734800 | 4365.446187 | `runs/smoke_test/checkpoints/best_model.pt` | smoke test |
| `runs/regression_video_smoke` | annotation metadata fallback via `video_feature` path | MLP | 1059/353 | 1 | -0.1974734800 | 4365.446187 | `runs/regression_video_smoke/checkpoints/best_model.pt` | video_feature 路径 smoke test，不是真实 video feature |
| `runs/skeleton_temporal_mlp_smoke` | synthetic skeleton | temporal_mlp + MLP | 1059/353 | 2 | 0.0786188914 | 4651.189384 | `runs/skeleton_temporal_mlp_smoke/checkpoints/best_model.pt` | 仅接口验证；原始 synthetic 输入已清理 |
| `runs/skeleton_stgcn_smoke` | synthetic skeleton | ST-GCN-style + MLP | 1059/353 | 2 | 0.1560733323 | 4641.265818 | `runs/skeleton_stgcn_smoke/checkpoints/best_model.pt` | 仅接口验证；原始 synthetic 输入已清理 |

所有 `runs/*/outputs/predictions.csv` 均为 353 行，与 test split 数量一致。

## 5. 当前复现性质判断

| 判断项 | 当前结论 |
|---|---|
| official baseline reproduction | 未完成 |
| C3D-AVG-MTL reproduction | 未完成 |
| MSCADC reproduction | 未完成 |
| simplified single-task baseline | 已完成 |
| partial reproduction | 是 |
| 使用官方 split | 是 |
| 预测 final score | 是 |
| Spearman / MSE 评估 | 是 |
| 保存 predictions / metrics / checkpoint | 是 |
| 真实 video feature baseline | 尚未完成 |
| 真实 skeleton baseline | 尚未完成 |

当前最准确表述：

```text
项目已经跑通 MTL-AQA 最小 score regression 闭环，但目前属于 simplified single-task baseline / partial reproduction，不是官方 baseline reproduction。
```

## 6. 与原论文目标的差距

论文中 C3D-AVG-MTL rank correlation 报告值为 90.44%。当前主实验 Spearman 为 0.3680757962。

该差距不能直接解释为模型优劣，因为当前实验设置不同：

- 当前输入是 annotation metadata fallback，不是真实 RGB/video feature。
- 当前模型是 numpy MLP single-task regression，不是官方 C3D-AVG-MTL。
- 当前没有使用 dive classification 或 captioning 多任务损失。
- 当前没有使用官方视频帧 pipeline。
- 当前 skeleton 结果只是 synthetic smoke test，不是真实 skeleton AQA 实验。

## 7. 问题与风险清单

| 类型 | 问题 | 影响 | 优先级 |
|---|---|---|---|
| 阻塞 | 未找到 extracted RGB frames | 官方 C3D/MSCADC pipeline 无法运行 | 高 |
| 阻塞 | 未找到真实 video feature `.npy/.npz` | 当前 video_feature 结果实际为 annotation fallback | 高 |
| 阻塞 | 未找到真实 skeleton `.npy/.npz` | 不能报告真实 skeleton baseline | 高 |
| 阻塞 | 官方 `opts.py` 仍含 `'...'` 路径占位 | 官方代码不可直接运行 | 高 |
| 阻塞 | 官方代码大量硬编码 `.cuda()` | CPU 环境或 CPU 版 torch 无法直接运行 | 高 |
| 阻塞 | 未找到可直接加载的 `c3d.pickle` / `C3D_small_PyTorch_Trained_12.pth` | 官方模型无法加载预训练权重 | 高 |
| 非阻塞 | 当前无 `requirements.txt` / `environment.yml` | 环境复现不够规范 | 中 |
| 非阻塞 | `train.py` 日志未显式记录 input_dim | 后期排查维度不够直观 | 中 |
| 科研风险 | 主结果来自 annotation fallback | 不能作为真实 video/skeleton 性能 | 高 |
| 科研风险 | skeleton smoke test 原始输入已清理 | smoke test 不能直接复跑，除非重新生成 synthetic 或提供真实 skeleton | 中 |

## 8. 下一步计划

### 近期，1-3 天

- [ ] 准备少量真实 skeleton `.npy`，格式 `[T,V,C]`。
- [ ] 使用真实 skeleton 跑通 1 epoch `temporal_mlp` smoke test。
- [ ] 使用真实 skeleton 跑通 1 epoch `stgcn` smoke test。
- [ ] 新增 `requirements.txt` 或 `environment.yml`。
- [ ] 在日志中增加 `input_type`、`model`、`input_dim`、train/test size。
- [ ] 如需保留 synthetic smoke test，可写脚本重新生成 synthetic skeleton，而不是提交大量 `.npy`。

### 中期，1-2 周

- [ ] 准备完整 MTL-AQA skeleton feature。
- [ ] 完整运行 skeleton temporal MLP baseline。
- [ ] 完整运行 skeleton ST-GCN-style baseline。
- [ ] 准备真实 video feature baseline，与 skeleton baseline 对比。
- [ ] 配置官方 `opts.py`，尝试官方 C3D-AVG 1 epoch smoke test。

### 后续

- [ ] 将 numpy MLP baseline 迁移到 PyTorch。
- [ ] 实现真正 trainable TCN / ST-GCN。
- [ ] 复现 Joint Relation Graphs 思路。
- [ ] 准备 FineDiving 第二阶段实验。

## 9. Git 当前状态

当前 `git status --short` 显示：

```text
D  __pycache__/dataset.cpython-312.pyc
D  __pycache__/dataset.cpython-38.pyc
D  __pycache__/metrics.cpython-312.pyc
D  __pycache__/metrics.cpython-38.pyc
D  __pycache__/model.cpython-312.pyc
D  __pycache__/model.cpython-38.pyc
D  __pycache__/test.cpython-312.pyc
D  __pycache__/train.cpython-312.pyc
```

说明：当前只剩 Python 缓存文件删除处于 git 变更状态。建议提交该清理，后续 `.gitignore` 已包含 `__pycache__/`，可避免再次加入缓存文件。
