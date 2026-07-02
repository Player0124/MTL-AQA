# MTL-AQA Minimal Reproduction

## Project Goal

Build a minimal, reproducible MTL-AQA action quality assessment loop:

```text
official split -> input feature -> score regressor -> final score prediction -> Spearman/MSE
```

This repository starts from the official MTL-AQA code release and adds a small single-task baseline for environments where the original video pipeline cannot run immediately.

## Paper Information

- Title: "What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment"
- Venue: CVPR 2019
- Official repository: https://github.com/ParitoshParmar/MTL-AQA
- Reported paper reference: C3D-AVG-MTL rank correlation = 90.44%.

## Dataset Information

MTL-AQA contains 1412 diving samples. The official split in this checkout is:

- Train: 1059 samples
- Test: 353 samples

Default split directory:

```text
MTL-AQA_dataset_release/Ready_2_Use/MTL-AQA_split_0_data/
  final_annotations_dict.pkl
  final_captions_dict.pkl
  train_split_0.pkl
  test_split_0.pkl
  vocab.json
```

The official training scripts also require extracted RGB frames, which are not included in the current checkout.

## Setup

The current runnable baseline uses only numpy and Python standard library. On this machine, use the bundled Python:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' --version
```

For the original official pipeline, create a separate environment with PyTorch, torchvision, scipy, Pillow, and CUDA support if available.

## Data Preparation

For the minimal baseline, no extra preparation is required beyond the official annotation files already in this repository.

For feature-based training, prepare one `.npy` or `.npz` file per official sample key. Supported names include:

```text
<feature_root>/18_67.npy
<feature_root>/18_67.npz
<feature_root>/18/67.npy
<feature_root>/18/67.npz
<feature_root>/18-67.npy
```

If a feature array has more than one dimension, the loader mean-pools over the first axis and flattens the remaining dimensions.

## Training

Smoke test:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' train.py --epochs 1 --batch-size 128 --output-dir runs\smoke_test
```

Default minimal baseline:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' train.py --epochs 20 --batch-size 64 --output-dir runs\mtl_aqa_minimal
```

Feature-based run:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' train.py --feature-root <feature_root> --feature-mode feature --epochs 20 --batch-size 64 --output-dir runs\mtl_aqa_features
```

Supported training arguments:

```text
--data-root
--split-file
--feature-root
--feature-mode
--epochs
--batch-size
--lr
--seed
--output-dir
```

## Evaluation

Reload a saved checkpoint and evaluate on the official test split:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' test.py --checkpoint runs\mtl_aqa_minimal\checkpoints\best_model.pt
```

Metrics:

- Spearman rank correlation via `scipy.stats.spearmanr` when scipy is installed.
- Built-in rank-correlation fallback when scipy is unavailable.
- MSE / L2 loss on raw final scores.

## Expected Outputs

```text
runs/mtl_aqa_minimal/
  checkpoints/
    best_model.pt
  logs/
    train.log
  outputs/
    metrics.json
    predictions.csv
```

`predictions.csv` columns:

```text
sample_id,prediction,ground_truth
```

## Current Result

| Run | Type | Input | Epochs | Test Spearman | Test MSE |
| --- | --- | --- | ---: | ---: | ---: |
| `runs/mtl_aqa_minimal` | simplified single-task baseline | annotation metadata dry-run | 20 | 0.3680757962 | 666.235943 |

Classification of this reproduction:

- Not an official baseline reproduction yet.
- Simplified single-task baseline.
- Partial reproduction of the score-regression evaluation loop using official split files.

Gap to paper:

- Paper C3D-AVG-MTL: 90.44% rank correlation.
- Current baseline: 36.81% rank correlation.
- Difference: about 53.63 percentage points.
- The gap is expected because this run does not use RGB frames, learned C3D features, captioning loss, dive classification loss, or the official multitask architecture.

## Known Issues

- Official `opts.py` contains placeholder paths.
- Official scripts assume CUDA via direct `.cuda()` calls.
- Current repository checkout does not include extracted frames.
- Current repository checkout does not include directly loadable C3D pretrained weights.
- Current machine's bundled Python does not include PyTorch, torchvision, or scipy.
- The runnable default baseline uses annotation metadata only, so it should be treated as a dry-run baseline rather than a video AQA model.

## Official Pipeline Notes

Official entry points:

```text
MTL-AQA_code_release/train_test_C3DAVG.py
MTL-AQA_code_release/train_test_MSCADC.py
```

Before running them, configure:

```text
MTL-AQA_code_release/opts.py
```

Set:

- `anno_n_splits_dir`
- `dataset_frames_dir`
- model-specific image size and resize options
- pretrained weight path in the training script or working directory

## Next Step: Replace RGB/Video Feature With Skeleton Sequence

The clean next step for skeleton-based AQA is to export one feature file per official sample key using the same split:

```text
<feature_root>/<video_id>_<sample_id>.npy
```

Then train:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' train.py --feature-root <skeleton_feature_root> --feature-mode feature --output-dir runs\mtl_aqa_skeleton_baseline
```

This preserves the dataset split, score target, metric code, output files, and documentation while replacing only the input representation.
