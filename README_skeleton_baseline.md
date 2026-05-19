# MTL-AQA Skeleton Input Baseline

## Goal

Replace the previous video/RGB-feature input with pre-extracted 2D skeleton sequences while keeping the same MTL-AQA split, final-score regression target, metrics, and output files.

This is a minimal input-replacement experiment. It does not include pose extraction, FineDiving, Joint Relation Graphs, or a SOTA architecture.

## Data Format

Place one skeleton file per official MTL-AQA sample:

```text
<skeleton_root>/<sample_id>.npy
```

Supported sample id formats:

```text
18_67.npy
18-67.npy
18/67.npy
```

Preferred canonical format:

```text
18_67.npy
```

Each file must contain:

```text
shape = [T, V, C]
```

- `T`: number of frames
- `V`: number of joints
- `C`: 2 or 3 channels
- `C=2`: `x, y`
- `C=3`: `x, y, confidence`

The loader pads or center-truncates to `--skeleton-length`, normalizes coordinates per sample, and sets missing joints to zero. A joint is treated as missing when coordinates are non-finite, both x/y are zero, or confidence is not above the configured threshold.

## Models

Two minimal skeleton baselines are available through the same training script:

| Model | Description |
| --- | --- |
| `temporal_mlp` | Flattens joints per frame, pools temporal mean/std/motion magnitude, then uses the existing MLP score regressor. |
| `stgcn` | Applies a fixed skeleton adjacency smoothing step, adds temporal velocity/acceleration features, globally pools, then uses the existing MLP score regressor. |

The implementation is numpy-only so it can run in the current environment without PyTorch. The `stgcn` model is an ST-GCN-style baseline, not a full trainable ST-GCN stack.

## Training Commands

Temporal MLP skeleton baseline:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' train.py --input-type skeleton --skeleton-root <skeleton_root> --model temporal_mlp --skeleton-length 103 --batch-size 64 --epochs 20 --output-dir runs\skeleton_temporal_mlp
```

ST-GCN-style skeleton baseline:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' train.py --input-type skeleton --skeleton-root <skeleton_root> --model stgcn --skeleton-length 103 --batch-size 64 --epochs 20 --output-dir runs\skeleton_stgcn
```

Evaluation:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' test.py --checkpoint runs\skeleton_stgcn\checkpoints\best_model.pt
```

The official split remains `MTL-AQA_split_0_data`; the target remains `final_score`; metrics remain Spearman rank correlation and MSE.

## Outputs

Each run writes:

```text
<output_dir>/
  checkpoints/best_model.pt
  logs/train.log
  outputs/metrics.json
  outputs/predictions.csv
```

`predictions.csv` keeps the same schema:

```text
sample_id,prediction,ground_truth
```

## Result Table

Current comparable video-feature baseline from the previous phase:

| Input | Model | Spearman | MSE | Notes |
| --- | --- | ---: | ---: | --- |
| Video feature | MLP | 0.3680757962 | 666.235943 | Previous Phase 1 dry-run baseline using annotation metadata fallback. |
| Skeleton | Temporal MLP | pending | pending | Run with real skeleton `.npy` files using the command above. |
| Skeleton | ST-GCN-style | pending | pending | Run with real skeleton `.npy` files using the command above. |

## Smoke-Test Result

The repository currently does not include real skeleton `.npy` files. To verify the new data/model paths, a synthetic skeleton directory was generated under `runs/synthetic_skeleton_smoke`. These numbers only validate the code path and are not real skeleton AQA results.

| Input | Model | Spearman | MSE | Notes |
| --- | --- | ---: | ---: | --- |
| Synthetic skeleton | Temporal MLP | 0.0786188914 | 4651.189384 | Interface smoke test only. |
| Synthetic skeleton | ST-GCN-style | 0.1560733323 | 4641.265818 | Interface smoke test only. |

## Reproducible Smoke Commands

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' train.py --input-type skeleton --skeleton-root runs\synthetic_skeleton_smoke --model temporal_mlp --epochs 2 --batch-size 128 --output-dir runs\skeleton_temporal_mlp_smoke
```

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' train.py --input-type skeleton --skeleton-root runs\synthetic_skeleton_smoke --model stgcn --epochs 2 --batch-size 128 --output-dir runs\skeleton_stgcn_smoke
```

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' test.py --checkpoint runs\skeleton_stgcn_smoke\checkpoints\best_model.pt
```

## Next Step

Export real pose sequences into `<skeleton_root>/<video_id>_<sample_id>.npy`, then rerun the two skeleton commands and fill the pending rows in the result table.
