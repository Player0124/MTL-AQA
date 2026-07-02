# 近期项目进度简报

## 1. 当前目标

本阶段围绕 MTL-AQA 数据集建立 action quality assessment 的最小实验闭环：

```text
official split -> input feature / skeleton -> model -> final score -> Spearman / MSE
```

后续目标是在保持官方 split、final score 回归目标和评价指标不变的前提下，将输入从 RGB/video feature 替换为 skeleton sequence。

## 2. 已完成工作

- 已基于官方 MTL-AQA 仓库整理项目结构，并保留官方 `MTL-AQA_code_release/` 作为参考实现。
- 已确认官方 split 可读取：共 1412 个样本，train 1059，test 353。
- 已实现最小 single-task score regression baseline：
  - `dataset.py`
  - `model.py`
  - `metrics.py`
  - `train.py`
  - `test.py`
  - `config.yaml`
- 已实现并验证输出：
  - checkpoint
  - train log
  - `metrics.json`
  - `predictions.csv`
- 已加入 skeleton input replacement 接口，支持 skeleton `.npy/.npz`，格式为 `[T,V,C]`。
- 已实现两个 skeleton 接入路径：
  - `temporal_mlp`
  - `stgcn` 风格固定图编码
- 已清理临时文件：
  - 删除 `get-pip.py`
  - 删除 `__pycache__/`
  - 删除合成 skeleton 原始 `.npy` smoke 数据
  - 新增 `.gitignore`

## 3. 当前实验结果

当前唯一较完整的主实验是 annotation metadata fallback baseline：

| Experiment | Input | Model | Epochs | Spearman | MSE | 说明 |
|---|---|---|---:|---:|---:|---|
| `runs/mtl_aqa_minimal` | annotation metadata fallback | MLP | 20 | 0.3680757962 | 666.235943 | 最小闭环验证 |

安装 scipy 后复评估得到：

```text
spearman_p = 9.1107e-13
```

Skeleton 方向目前只有 smoke-test 结果，用于验证代码路径，不作为真实性能结论：

| Experiment | Input | Model | Epochs | Spearman | MSE | 说明 |
|---|---|---|---:|---:|---:|---|
| `runs/skeleton_temporal_mlp_smoke` | synthetic skeleton | temporal_mlp + MLP | 2 | 0.0786188914 | 4651.189384 | 仅接口验证 |
| `runs/skeleton_stgcn_smoke` | synthetic skeleton | ST-GCN-style + MLP | 2 | 0.1560733323 | 4641.265818 | 仅接口验证 |

## 4. 当前环境状态

项目解释器为：

```text
C:\Users\User\AppData\Local\Programs\Python\Python38\python.exe
```

已安装：

```text
scipy 1.10.1
PyYAML 6.0.3
torch 2.4.1+cpu
torchvision 0.19.1+cpu
```

当前 PyTorch 仍是 CPU 版：

```text
cuda_available = False
```

如果要运行官方 `.cuda()` 代码，需要将 PyTorch 换成 CUDA 版，或修改官方代码以支持 CPU/device 自动选择。

## 5. 当前限制

- 当前主结果不是官方 C3D-AVG-MTL baseline。
- 当前主结果使用 annotation metadata fallback，不是真实 RGB/video feature。
- 尚未接入真实 skeleton `.npy` 数据。
- 官方代码仍存在路径占位符和硬编码 `.cuda()`。
- 缺少 extracted video frames 和可直接加载的完整 C3D 权重。

## 6. 下一步计划

- 准备真实 skeleton `.npy`，文件名对应官方 sample key，例如 `18_67.npy`。
- 跑通真实 skeleton temporal MLP 1 epoch smoke test。
- 跑通真实 skeleton ST-GCN-style 1 epoch smoke test。
- 将 PyTorch 从 CPU 版切换到 CUDA 版，验证 `torch.cuda.is_available()`。
- 配置官方 `opts.py`，尝试 official C3D-AVG 1 epoch smoke test。
- 后续补充真实 video feature baseline，与 skeleton baseline 做对比。

## 7. 简短结论

目前项目已经完成 MTL-AQA 最小 score regression 闭环，并完成 skeleton 输入接口验证；但官方 baseline 和真实 skeleton 实验尚未完成。当前工作适合作为后续 skeleton-based AQA 实验的工程基础。
