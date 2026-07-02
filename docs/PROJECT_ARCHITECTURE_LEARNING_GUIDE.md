# 项目架构学习总结

生成日期：2026-07-02  
适用对象：刚开始接触本项目的同学  
项目路径：`F:\SkeletonRecognition\MTL-AQA`

## 1. 一句话理解本项目

当前项目围绕 MTL-AQA 数据集做 action quality assessment。核心目标是建立一个最小可复现闭环：

```text
官方 split -> 输入特征 / skeleton -> 模型 -> final score 预测 -> Spearman / MSE 评估
```

当前项目不是完整官方论文复现，而是先建立一个可运行、可替换输入模态的 baseline。后续重点是将输入从 RGB/video feature 替换为 skeleton sequence。

## 2. 当前项目分成两层

### 2.1 官方代码层

路径：

```text
MTL-AQA_code_release/
```

这是论文作者发布的原始代码，包含 C3D-AVG、MSCADC、caption、多任务分类等内容。它更接近论文，但当前不能直接运行，原因包括：

- `opts.py` 中路径仍是 `'...'` 占位。
- 官方训练脚本硬编码 `.cuda()`。
- 当前 PyTorch 是 CPU 版，CUDA 不可用。
- 当前仓库没有 extracted RGB frames。
- 当前仓库没有直接可加载的完整 C3D 权重文件。

所以初学阶段建议先阅读但不要急着修改官方代码。

### 2.2 新增最小 baseline 层

路径在项目根目录：

```text
dataset.py
model.py
metrics.py
train.py
test.py
config.yaml
```

这一层是当前真正可运行的实验框架。它做的事情更简单：

```text
读取官方 split
读取 annotation / feature / skeleton
训练 MLP 回归 final_score
计算 Spearman 和 MSE
保存 checkpoint、metrics、predictions
```

## 3. 推荐阅读顺序

建议按下面顺序阅读代码：

1. `README_reproduction.md`  
   了解第一阶段为什么做最小 score regression baseline。

2. `README_skeleton_baseline.md`  
   了解 skeleton 输入格式、命令和结果表。

3. `config.yaml`  
   看默认数据路径、训练参数、模型选择。

4. `train.py`  
   看整个训练流程。

5. `dataset.py`  
   看数据如何从官方 split 变成模型输入。

6. `model.py`  
   看 MLP regressor 如何 forward/backward/Adam 更新。

7. `metrics.py`  
   看 Spearman 和 MSE 如何计算。

8. `test.py`  
   看 checkpoint 如何被重新加载并评估。

9. `MTL-AQA_code_release/`  
   最后再看官方复杂模型。

## 4. 数据区说明

### 4.1 官方 annotation 与 split

核心路径：

```text
MTL-AQA_dataset_release/Ready_2_Use/MTL-AQA_split_0_data/
```

主要文件：

| 文件 | 作用 |
|---|---|
| `final_annotations_dict.pkl` | 每个样本的标注，包括 `final_score` |
| `train_split_0.pkl` | 官方训练集 sample key |
| `test_split_0.pkl` | 官方测试集 sample key |
| `final_captions_dict.pkl` | caption 标注 |
| `vocab.json` | caption 词表 |

当前已确认：

```text
total samples = 1412
train = 1059
test = 353
```

sample key 形式：

```python
(18, 67)
```

在文件名中通常写成：

```text
18_67.npy
```

## 5. 核心文件详解

### 5.1 `config.yaml`

这是默认配置文件。当前内容包括：

```yaml
input_type: video_feature
data_root: MTL-AQA_dataset_release/Ready_2_Use
split_file: MTL-AQA_split_0_data
feature_root: null
feature_mode: annotation
skeleton_root: null
skeleton_length: 103
skeleton_dims: 3
model: mlp
epochs: 20
batch_size: 64
lr: 0.001
seed: 0
hidden_dim: 128
output_dir: runs/mtl_aqa_minimal
```

命令行参数会覆盖这里的默认值。

### 5.2 `dataset.py`

这是数据读取和输入模态替换的核心。

主要 class：

| 类 | 作用 |
|---|---|
| `MTLAQADataset` | 读取 annotation 或离线 feature |
| `MTLAQASkeletonDataset` | 读取 skeleton sequence |

#### `MTLAQADataset`

支持三种模式：

1. annotation fallback  
   当前主实验使用这种模式。它用 annotation metadata 作为输入特征，用来验证训练闭环。

2. feature file  
   如果传入 `--feature-root` 和 `--feature-mode feature`，会读取 `.npy/.npz` feature。

3. auto fallback  
   如果有 feature 就读 feature，没有就回退到 annotation。

#### `MTLAQASkeletonDataset`

要求 skeleton 文件为：

```text
<skeleton_root>/<video_id>_<sample_id>.npy
```

shape：

```text
[T, V, C]
```

含义：

| 维度 | 含义 |
|---|---|
| `T` | 帧数 |
| `V` | 关节点数量 |
| `C` | 坐标通道，2 或 3 |

`C=2`：

```text
x, y
```

`C=3`：

```text
x, y, confidence
```

这个 dataset 会做：

- padding 或 center truncation 到固定长度；
- 坐标归一化；
- missing joints 置零；
- skeleton 特征编码；
- 返回 `feature` 和 `final_score`。

#### skeleton encoder

当前有两个 encoder：

| encoder | 说明 |
|---|---|
| `temporal_mlp` | flatten 每帧关节，做时间 mean/std/motion pooling |
| `stgcn` | 固定 graph adjacency 平滑 + velocity/acceleration + pooling |

注意：当前 `stgcn` 是 ST-GCN-style 特征编码，不是完整 trainable ST-GCN。

### 5.3 `model.py`

实现 `NumpyMLPRegressor`。

结构：

```text
input feature
  -> Linear(input_dim, hidden_dim)
  -> ReLU
  -> Linear(hidden_dim, 1)
  -> predicted final_score
```

训练：

- loss：MSE
- optimizer：自实现 Adam
- backend：numpy
- device：CPU

为什么用 numpy：当前最小 baseline 最初是为了在没有 PyTorch 的情况下也能跑通。现在 PyTorch 已安装，但该 baseline 仍保持 numpy 实现。

### 5.4 `metrics.py`

实现两个指标：

| 函数 | 作用 |
|---|---|
| `mse()` | 计算 raw final score 的均方误差 |
| `spearmanr()` | 计算 Spearman rank correlation |

当前 `scipy` 已安装，因此优先使用：

```python
scipy.stats.spearmanr
```

如果 scipy 不可用，代码仍保留 fallback rank 实现。

### 5.5 `train.py`

这是训练入口。

常用参数：

```text
--input-type video_feature | skeleton
--feature-root
--feature-mode auto | feature | annotation
--skeleton-root
--model mlp | temporal_mlp | stgcn
--epochs
--batch-size
--lr
--seed
--output-dir
```

训练流程：

```text
1. 读取 config.yaml
2. 解析 CLI 参数
3. 固定 seed
4. 根据 input_type 创建 dataset
5. 标准化 train/test feature
6. 创建 MLP regressor
7. 每个 epoch 训练
8. 每个 epoch 在 test split 上评估
9. 保存 best checkpoint
10. 写 predictions.csv 和 metrics.json
```

### 5.6 `test.py`

这是评估入口。

功能：

```text
1. 读取 best_model.pt
2. 恢复训练时的参数
3. 重新读取 test split
4. 用保存的 feature mean/std 做标准化
5. 预测 final score
6. 写 metrics.json 和 predictions.csv
```

### 5.7 `runs/`

实验输出目录。

典型结构：

```text
runs/<experiment_name>/
  checkpoints/
    best_model.pt
  logs/
    train.log
  outputs/
    metrics.json
    predictions.csv
```

当前重要实验：

| 实验 | 说明 |
|---|---|
| `runs/mtl_aqa_minimal` | 20 epoch annotation fallback baseline |
| `runs/smoke_test` | 1 epoch smoke test |
| `runs/regression_video_smoke` | video_feature 路径 smoke test |
| `runs/skeleton_temporal_mlp_smoke` | synthetic skeleton temporal MLP smoke test |
| `runs/skeleton_stgcn_smoke` | synthetic skeleton ST-GCN-style smoke test |
| `runs/synthetic_skeleton_smoke` | 原始合成 `.npy` 已清理；该目录不应作为真实数据保留 |

## 6. 当前已知实验结果

| Experiment | Input | Model | Epochs | Spearman | MSE | 说明 |
|---|---|---|---:|---:|---:|---|
| `runs/mtl_aqa_minimal` | annotation metadata fallback | MLP | 20 | 0.3680757962 | 666.235943 | 最小闭环结果；复评估后 `spearman_p=9.1107e-13` |
| `runs/skeleton_temporal_mlp_smoke` | synthetic skeleton | temporal_mlp + MLP | 2 | 0.0786188914 | 4651.189384 | 仅验证接口 |
| `runs/skeleton_stgcn_smoke` | synthetic skeleton | ST-GCN-style + MLP | 2 | 0.1560733323 | 4641.265818 | 仅验证接口 |

不能误读：

- `runs/mtl_aqa_minimal` 不是官方 video baseline。
- skeleton smoke test 不是真实 skeleton 性能。
- 当前还不能与论文 90.44% 直接比较。

## 7. 常用命令

### 7.1 验证依赖

```powershell
& "C:\Users\User\AppData\Local\Programs\Python\Python38\python.exe" -c "import scipy, yaml, torch, torchvision; print(scipy.__version__, yaml.__version__, torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

### 7.2 跑最小 baseline

```powershell
& "C:\Users\User\AppData\Local\Programs\Python\Python38\python.exe" train.py --input-type video_feature --model mlp --epochs 20 --batch-size 64 --output-dir runs\mtl_aqa_minimal
```

### 7.3 跑真实 skeleton temporal MLP

```powershell
& "C:\Users\User\AppData\Local\Programs\Python\Python38\python.exe" train.py --input-type skeleton --skeleton-root <skeleton_root> --model temporal_mlp --epochs 20 --output-dir runs\skeleton_temporal_mlp
```

### 7.4 跑真实 skeleton ST-GCN-style

```powershell
& "C:\Users\User\AppData\Local\Programs\Python\Python38\python.exe" train.py --input-type skeleton --skeleton-root <skeleton_root> --model stgcn --epochs 20 --output-dir runs\skeleton_stgcn
```

### 7.5 重新评估 checkpoint

```powershell
& "C:\Users\User\AppData\Local\Programs\Python\Python38\python.exe" test.py --checkpoint runs\mtl_aqa_minimal\checkpoints\best_model.pt
```

## 8. 当前主要阻塞

| 阻塞 | 影响 |
|---|---|
| 没有真实 skeleton `.npy` | 不能报告真实 skeleton baseline |
| 没有真实 video feature | 当前 video_feature baseline 实际是 annotation fallback |
| 没有 extracted RGB frames | 官方 C3D/MSCADC pipeline 不能运行 |
| 官方代码硬编码 `.cuda()` | 当前 CPU PyTorch 不能直接运行官方代码 |
| 官方 `opts.py` 未配置路径 | 官方 pipeline 不能直接运行 |
| 缺少完整 C3D 权重 | 官方模型无法加载预训练 backbone |

## 9. 初学者修改建议

如果你要继续开发，建议从小到大改：

1. 先不要动 `MTL-AQA_code_release/`。
2. 先准备少量真实 skeleton `.npy`。
3. 用 `--input-type skeleton` 跑 1 epoch。
4. 检查 `predictions.csv` 是否有 353 行。
5. 检查 `metrics.json` 是否生成。
6. 再跑完整 20 epoch。
7. 最后考虑把 numpy MLP 迁移到 PyTorch。

## 10. 当前项目定位

当前项目最准确的定位是：

```text
MTL-AQA 最小 score regression 闭环已完成；
skeleton 输入接口已完成 smoke test；
真实 video/skeleton baseline 与官方论文复现仍未完成。
```

这份定位对科研汇报很重要，避免把 smoke test 当作最终实验结果。
