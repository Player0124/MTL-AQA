# 当前项目进度总结汇报

生成日期：2026-07-02  
项目目录：`F:\SkeletonRecognition\MTL-AQA`  
Git remote：`origin = https://github.com/Player0124/MTL-AQA.git`，`upstream = https://github.com/ParitoshParmar/MTL-AQA.git`  
Git 状态：初次审计时 `git status --short` 无输出，工作区干净；本次更新后新增/修改了汇报文档，并因复跑 `test.py` 更新了 `runs/mtl_aqa_minimal/outputs/metrics.json`。

## 1. 项目结构审计

### 1.1 当前目录结构摘要

```text
F:\SkeletonRecognition\MTL-AQA
  README.md
  README_reproduction.md
  README_skeleton_baseline.md
  reproduction_plan.md
  results_skeleton_baseline.md
  config.yaml
  dataset.py
  model.py
  metrics.py
  train.py
  test.py
  MTL-AQA_code_release/
  MTL-AQA_dataset_release/
  runs/
  __pycache__/
  .idea/
```

### 1.2 关键文件说明

| 文件/目录 | 作用 | 当前状态 |
|---|---|---|
| `README.md` | 官方仓库 README | 存在 |
| `reproduction_plan.md` | 第一阶段复现计划与阻塞记录 | 存在 |
| `README_reproduction.md` | MTL-AQA 最小 score regression baseline 文档 | 存在 |
| `README_skeleton_baseline.md` | skeleton input replacement baseline 文档 | 存在 |
| `results_skeleton_baseline.md` | video-feature baseline 与 skeleton smoke test 结果表 | 存在 |
| `config.yaml` | 当前训练默认配置 | 存在，使用相对路径 |
| `dataset.py` | MTL-AQA annotation/feature dataset 与 skeleton dataset | 核心实现 |
| `model.py` | numpy MLP score regressor | 核心实现 |
| `metrics.py` | MSE 与 Spearman rank correlation | 核心实现 |
| `train.py` | 统一训练入口 | 核心实现 |
| `test.py` | checkpoint 复评估入口 | 核心实现 |
| `MTL-AQA_code_release/` | 官方 C3D-AVG / MSCADC 代码 | 存在，但未跑通官方 pipeline |
| `MTL-AQA_dataset_release/` | 官方 annotation、split、raw annotation | 存在 |
| `runs/` | 实验产物：logs、metrics、predictions、checkpoint、synthetic skeleton smoke data | 实验产物 |
| `__pycache__/` | Python 字节码缓存 | 临时/缓存文件 |
| `.idea/` | IDE 配置 | 非核心文件 |

### 1.3 当前代码模块划分

| 模块 | 文件 | 说明 |
|---|---|---|
| 数据读取 | `dataset.py` | 读取官方 split、annotation、feature `.npy/.npz`、skeleton `.npy/.npz` |
| skeleton 预处理 | `dataset.py` | padding/truncation、坐标归一化、missing joints 处理 |
| skeleton 特征编码 | `dataset.py` | `temporal_mlp` 与 `stgcn` 风格编码 |
| 模型 | `model.py` | `NumpyMLPRegressor`，一层 hidden MLP，Adam 更新 |
| 指标 | `metrics.py` | MSE、Spearman，scipy 不可用时使用 fallback |
| 训练 | `train.py` | CLI/config、seed、训练循环、日志、checkpoint、metrics、predictions |
| 测试 | `test.py` | 加载 checkpoint、重新评估 test split |
| 官方参考实现 | `MTL-AQA_code_release/` | 作者原始 C3D-AVG、MSCADC、多任务代码 |

### 1.4 核心实现文件

- `dataset.py`
- `model.py`
- `metrics.py`
- `train.py`
- `test.py`
- `config.yaml`
- `README_reproduction.md`
- `README_skeleton_baseline.md`

### 1.5 临时文件或实验产物

| 路径 | 类型 | 说明 |
|---|---|---|
| `runs/mtl_aqa_minimal/` | 实验输出 | 20 epoch annotation fallback baseline |
| `runs/smoke_test/` | 实验输出 | 1 epoch smoke test |
| `runs/regression_video_smoke/` | 实验输出 | video_feature 路径 regression smoke test |
| `runs/skeleton_temporal_mlp_smoke/` | 实验输出 | synthetic skeleton temporal MLP smoke test |
| `runs/skeleton_stgcn_smoke/` | 实验输出 | synthetic skeleton ST-GCN-style smoke test |
| `runs/synthetic_skeleton_smoke/` | 合成测试数据 | 用于验证 skeleton loader，不是真实 skeleton 数据 |
| `__pycache__/` | 缓存 | Python 编译缓存 |
| `.idea/` | IDE 配置 | 与实验无直接关系 |

## 2. 目标与当前实现对齐检查

| 模块 | 当前状态 | 证据文件/路径 | 问题 | 结论 |
|---|---|---|---|---|
| 数据读取 | 部分完成 | `dataset.py`; `MTL-AQA_dataset_release/Ready_2_Use/MTL-AQA_split_0_data/` | 支持官方 split 与 annotation；无真实 RGB frame / C3D feature；真实 skeleton 未提供 | 最小闭环可用，真实模态数据不足 |
| 官方 split | 已完成 | `train_split_0.pkl`, `test_split_0.pkl`; 审计结果 1059/353 | 未发现 split 被修改 | 使用官方 split |
| feature/annotation/label | 部分完成 | `MTLAQADataset` | 当前主结果使用 annotation metadata fallback，不是真实 video feature | 可验证训练闭环，但不能代表 video baseline |
| 数据路径参数化 | 已完成 | `config.yaml`, `train.py` CLI | 官方 release 的 `opts.py` 仍含 `'...'` 占位 | 新增 baseline 路径参数化，官方代码未配置 |
| 硬编码路径 | 部分存在 | `MTL-AQA_code_release/opts.py`; `train_test_*.py` | 官方代码有占位路径、固定权重文件名、`.cuda()` | 新增代码基本相对路径，官方代码需整理 |
| 模型 | 部分完成 | `model.py`, `dataset.py` skeleton encoder | MLP 是 numpy 实现；`stgcn` 是特征编码风格，不是完整 trainable ST-GCN | simplified single-task baseline 完成 |
| official baseline | 未完成 | `MTL-AQA_code_release/train_test_C3DAVG.py`, `train_test_MSCADC.py` | 已安装 CPU 版 PyTorch/torchvision/scipy，但缺帧、缺完整可加载权重、官方代码硬编码 CUDA | 尚未复现官方 baseline |
| MLP regressor | 已完成 | `model.py` | 仅 CPU/numpy，无 GPU | 可用 |
| skeleton / temporal / ST-GCN | 部分完成 | `dataset.py`, `README_skeleton_baseline.md` | 真实 skeleton 数据未接入；ST-GCN 不是完整图卷积训练网络 | 接口完成，真实实验未完成 |
| 模型输入输出维度 | 部分清晰 | `dataset.py`, `model.py` | feature 编码后进入 MLP；未在日志中显式打印 input_dim | 代码可推导，建议增加日志 |
| train.py | 已完成 | `train.py` | 无 GPU 支持 | CPU/numpy 训练可用 |
| config/CLI | 已完成 | `config.yaml`, `train.py` | 无 PyYAML，使用自定义简单 parser | 当前够用 |
| seed 固定 | 已完成 | `train.py:set_seed`, `model.py:MLPConfig.seed` | 仅 numpy/random | 对当前实现有效 |
| CPU/GPU | 部分完成 | `train.py` 日志 `device: cpu-numpy` | 不支持 CUDA/GPU | 当前可 CPU 跑通 |
| train loss | 已完成 | `runs/*/logs/train.log` | 日志为纯文本，格式简单 | 可用 |
| checkpoint | 已完成 | `runs/*/checkpoints/best_model.pt` | numpy pickle 格式，扩展性有限 | 可复载 |
| test.py | 已完成 | `test.py` | 无单独 `eval.py`，但 `test.py` 覆盖评估 | 可用 |
| Spearman | 已完成 | `metrics.py` | scipy 已安装；历史 smoke test 的 metrics 可能仍保留旧的 null p-value，重新评估后可生成 p-value | 指标值可用 |
| MSE/L2 | 已完成 | `metrics.py` | 使用 raw final score MSE | 可用 |
| predictions.csv | 已完成 | `runs/*/outputs/predictions.csv` | 每个输出 353 行，对应 test split | 可用 |
| metrics.json | 已完成 | `runs/*/outputs/metrics.json` | 主实验已重新评估并生成 `spearman_p`；部分历史 smoke test 未重跑 | 可用 |
| train/test leakage | 未发现 | `dataset.py`, split files | 当前使用 train/test pkl；未发现混用训练集评估 | 初步可信 |
| 完整训练日志 | 部分完成 | `runs/mtl_aqa_minimal/logs/train.log` | 真实 skeleton 只有 2 epoch smoke test | annotation baseline 日志完整 |
| 复现实验命令 | 已完成 | `README_reproduction.md`, `README_skeleton_baseline.md`, logs args | 官方 pipeline 命令不可直接运行 | 新增 baseline 可复现 |

## 3. 实验结果提取

### 3.1 数据与 split

审计命令读取 `final_annotations_dict.pkl`、`train_split_0.pkl`、`test_split_0.pkl` 得到：

- annotation 总数：1412
- train split：1059
- test split：353
- 示例 train key：`(18, 67)`，final_score = `94.35`
- 示例 test key：`(26, 4)`，final_score = `64.5`

### 3.2 环境状态

当前可用 Python 与依赖状态：

```text
C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
Python 3.12.13
numpy 2.3.5
scipy 1.18.0
PyYAML 6.0.3
torch 2.12.1+cpu
torchvision 0.27.1+cpu
torch.cuda.is_available() = False
torch.cuda.device_count() = 0
```

未找到：

- `requirements.txt`
- `environment.yml`
- notebook 文件

### 3.3 实验结果表

| Experiment | Input | Model | Split | Epochs | Spearman | MSE/L2 | Checkpoint | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| `runs/mtl_aqa_minimal` | annotation metadata fallback | MLP | 1059/353 | 20 | 0.3680757962 | 666.235943 | `runs/mtl_aqa_minimal/checkpoints/best_model.pt` | simplified single-task baseline；不是真实 video feature；复评估后 `spearman_p=9.1107e-13` |
| `runs/smoke_test` | annotation metadata fallback | MLP | 1059/353 | 1 | -0.1974734800 | 4365.446187 | `runs/smoke_test/checkpoints/best_model.pt` | smoke test |
| `runs/regression_video_smoke` | annotation metadata fallback via `video_feature` path | MLP | 1059/353 | 1 | -0.1974734800 | 4365.446187 | `runs/regression_video_smoke/checkpoints/best_model.pt` | regression smoke test |
| `runs/skeleton_temporal_mlp_smoke` | synthetic skeleton | temporal_mlp + MLP | 1059/353 | 2 | 0.0786188914 | 4651.189384 | `runs/skeleton_temporal_mlp_smoke/checkpoints/best_model.pt` | synthetic smoke test only，不是真实 skeleton 结果 |
| `runs/skeleton_stgcn_smoke` | synthetic skeleton | ST-GCN-style encoder + MLP | 1059/353 | 2 | 0.1560733323 | 4641.265818 | `runs/skeleton_stgcn_smoke/checkpoints/best_model.pt` | synthetic smoke test only，不是真实 skeleton 结果 |

### 3.4 主实验日志摘要

`runs/mtl_aqa_minimal/logs/train.log`：

- epochs：20
- batch size：64
- lr：0.001
- seed：0
- optimizer：自实现 Adam，见 `model.py`
- loss：MSE
- device：`cpu-numpy`
- epoch 20：
  - train_loss = `745.417859`
  - train_spearman = `0.3351983145`
  - train_mse = `693.854579`
  - test_spearman = `0.3680757962`
  - test_spearman_p = `9.110719287371563e-13`，安装 scipy 后复评估生成
  - test_mse = `666.235943`

### 3.5 predictions 文件

所有 `runs/*/outputs/predictions.csv` 均为 353 行预测，对应 test split 数量。  
`runs/mtl_aqa_minimal/outputs/predictions.csv` 前几行：

```text
sample_id,prediction,ground_truth
26_4,46.959974,64.500000
06_28,70.709840,76.800000
01_79,43.753043,81.600000
10_36,37.255776,44.800000
17_21,40.557919,68.200000
```

### 3.6 是否使用官方 baseline / 简化 baseline

| 项目 | 结论 |
|---|---|
| 是否使用官方 split | 是 |
| 是否完成官方 C3D-AVG-MTL baseline | 否 |
| 是否完成官方 MSCADC baseline | 否 |
| 是否完成 simplified single-task baseline | 是 |
| 是否有真实 video feature baseline | 否，当前是 annotation metadata fallback |
| 是否有真实 skeleton baseline | 否，只有 synthetic skeleton smoke test |
| 是否有 prediction vs ground truth | 是 |
| 是否有 metrics.json | 是 |
| 是否有 checkpoint | 是 |

## 4. 当前完成度评估

| 维度 | 分数 | 依据 | 主要缺口 |
|---|---:|---|---|
| 代码结构完整度 | 72 | `dataset.py/model.py/metrics.py/train.py/test.py/config.yaml` 已形成闭环 | 仍是轻量脚本式结构，无 package/tests |
| 数据加载完成度 | 68 | 支持官方 split、annotation、feature、skeleton 文件格式 | 缺真实 video feature、真实 skeleton、RGB frames |
| baseline 模型完成度 | 62 | MLP regressor 可训练；skeleton encoder 已接入 | official baseline 未跑通；ST-GCN 不是完整 trainable 网络 |
| 训练流程完成度 | 70 | 支持 CLI/config/seed/log/checkpoint | 仅 CPU/numpy，无 GPU；缺验证集划分 |
| 评估流程完成度 | 80 | Spearman/MSE、predictions、metrics 均存在；主实验已生成 scipy p-value | 部分历史 smoke test 未重跑，仍可能保留旧格式 metrics |
| 实验结果可信度 | 54 | annotation fallback 主结果有 20 epoch 日志和输出 | 不是真实 video/skeleton 结果，不能和论文直接比较 |
| 可复现性 | 66 | 命令、配置、输出齐全；git 状态干净 | 依赖文件缺失；官方 pipeline 不可直接复现 |
| skeleton 方向对齐度 | 58 | skeleton dataset 与 temporal/ST-GCN-style 接口已实现 | 真实 skeleton 未接入，未完成真实 skeleton 实验 |
| 文档完整度 | 74 | reproduction/skeleton README 和 result table 存在 | 需要加入更系统的环境安装和真实数据准备说明 |

总体完成度估计：`66/100`。  
阶段判断：`51–70：已形成最小闭环，但结果或文档不完整`。

## 5. 问题与风险清单

| 类型 | 问题 | 影响 | 建议处理方式 | 优先级 |
|---|---|---|---|---|
| 阻塞性 | 未找到真实 RGB frames | 官方 C3D/MSCADC pipeline 无法运行 | 下载视频并用 `frame_extractor.sh` 抽帧，或准备离线 C3D/I3D feature | 高 |
| 阻塞性 | 未找到真实 video feature `.npy/.npz` | 当前 “video_feature” 结果实际是 annotation fallback | 准备每个 sample 的真实 feature 文件，使用 `--feature-root` | 高 |
| 阻塞性 | 未找到真实 skeleton `.npy` | skeleton baseline 只有 synthetic smoke test | 导出 `<video_id>_<sample_id>.npy`，shape `[T,V,C]` | 高 |
| 阻塞性 | 当前安装的是 CPU 版 PyTorch，CUDA 不可用 | 官方代码硬编码 `.cuda()`，仍无法直接运行官方 pipeline | 方案一：在 GPU/CUDA 环境运行；方案二：给官方代码增加 device 管理以支持 CPU dry-run | 高 |
| 阻塞性 | 官方 `opts.py` 仍是占位路径 | 官方 baseline 不可直接运行 | 配置 `dataset_dir/anno_n_splits_dir/dataset_frames_dir` | 高 |
| 阻塞性 | 官方代码硬编码 `.cuda()` | 无 GPU 或 CPU 环境无法 dry-run | 增加 device 管理或在 GPU 环境运行 | 高 |
| 阻塞性 | 官方 C3D 权重未找到为直接可加载形式 | `torch.load('c3d.pickle')` / `.pth` 无法验证 | 下载/解压对应权重并校验路径 | 高 |
| 非阻塞性 | 没有 `requirements.txt` / `environment.yml` | 环境复现不够规范 | 新增最小环境文件和官方环境说明；记录已安装 `scipy/PyYAML/torch/torchvision` | 中 |
| 非阻塞性 | 无自动化测试 | 后续修改可能破坏数据路径或指标 | 增加 small smoke test 脚本 | 中 |
| 非阻塞性 | ST-GCN 命名可能引起误解 | 当前是 ST-GCN-style 固定编码，不是完整模型 | 文档中持续明确，后续实现真 ST-GCN | 中 |
| 非阻塞性 | 日志格式较简单 | 后期实验管理困难 | 统一 JSONL/CSV logging | 低 |
| 非阻塞性 | checkpoint 是 pickle/numpy 格式 | 与 PyTorch 模型不通用 | 后续 PyTorch 化后保存 `.pt` state_dict | 中 |
| 科研风险 | 当前 Spearman 0.368 来自 annotation fallback | 不能证明 video/skeleton AQA 性能 | 只作为 pipeline sanity check 汇报 | 高 |
| 科研风险 | 与论文 C3D-AVG-MTL 90.44% 差距大 | 不能宣称复现官方结果 | 明确标注 partial reproduction | 高 |
| 科研风险 | skeleton 输入未使用真实姿态 | 不能得出 skeleton 方向结论 | 尽快接入真实 pose extraction 输出 | 高 |
| 科研风险 | 未复现 official multitask loss | 与论文设置不同 | 后续单独建立 official baseline 复现实验 | 中 |
| 科研风险 | 未建立验证集/多 seed | 指标稳定性未知 | 增加多 seed 和统计结果 | 中 |

## 6. 下一步计划

### 近期任务，1–3 天

- [ ] 新增 `requirements_minimal.txt`，记录当前 numpy baseline 依赖。
- [ ] 新增 `requirements_official.txt` 或 `environment_official.yml`，记录 PyTorch/scipy/torchvision 官方复现环境。
- [ ] 清理或标注 `runs/synthetic_skeleton_smoke/` 为 smoke data，避免误当真实 skeleton。
- [ ] 在 `train.py` 日志中打印 `input_dim`、`input_type`、`model`。
- [ ] 准备真实 skeleton `.npy` 小样本，至少覆盖 train/test 各若干样本，验证真实 skeleton loader。
- [ ] 跑真实 skeleton 的 1 epoch smoke test，生成 `predictions.csv` 与 `metrics.json`。
- [ ] 补充 `README_skeleton_baseline.md` 的真实 skeleton 数据准备示例。

### 中期任务，1–2 周

- [ ] 准备完整 MTL-AQA skeleton feature：`<video_id>_<sample_id>.npy`，shape `[T,V,C]`。
- [ ] 跑完整 skeleton temporal MLP baseline，并更新 result table。
- [ ] 跑完整 skeleton ST-GCN-style baseline，并更新 result table。
- [ ] 准备真实 C3D/I3D/video feature，替换 annotation fallback。
- [ ] 跑真实 video-feature MLP baseline，作为 skeleton 对照。
- [ ] 安装并验证官方 PyTorch 环境。
- [ ] 配置官方 `opts.py`，尝试 1 epoch C3D-AVG official smoke test。
- [ ] 对比 simplified baseline、video feature baseline、skeleton baseline 的 Spearman/MSE。

### 后续任务

- [ ] 实现真正 trainable ST-GCN 或轻量 TCN encoder。
- [ ] 复现 Action Assessment by Joint Relation Graphs 的核心骨架建模思想。
- [ ] 将 MTL-AQA video feature baseline 与 skeleton baseline 做系统对比。
- [ ] 准备 FineDiving 第二阶段实验。
- [ ] 增加多 seed、置信区间、实验表格自动汇总。
- [ ] 考虑从 numpy baseline 迁移到 PyTorch，以便后续模型扩展。

## 7. 当前结论

当前项目已经完成了一个可运行的 MTL-AQA 最小 score regression 闭环：官方 split、final score 回归、MSE、Spearman、checkpoint、metrics、predictions 均已具备。但当前主结果使用的是 annotation metadata fallback，不是真实 RGB/video feature，也不是官方 C3D-AVG-MTL baseline。skeleton 方向已经完成接口和 synthetic smoke test，但尚未接入真实 skeleton 数据，因此不能报告真实 skeleton AQA 性能。

该项目目前适合向导师汇报为：`最小复现闭环已建立，官方完整复现与真实 skeleton 实验仍在准备阶段`。
