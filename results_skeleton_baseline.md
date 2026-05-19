# Skeleton Baseline Results

## Real Experiment Table

| Input | Model | Spearman | MSE | Notes |
| --- | --- | ---: | ---: | --- |
| Video feature | MLP | 0.3680757962 | 666.235943 | Previous Phase 1 dry-run baseline using annotation metadata fallback. |
| Skeleton | Temporal MLP | pending | pending | Requires real skeleton `.npy` files. |
| Skeleton | ST-GCN-style | pending | pending | Requires real skeleton `.npy` files. |

## Smoke-Test Table

Synthetic skeleton files under `runs/synthetic_skeleton_smoke` were used only to verify the new skeleton input path.

| Input | Model | Spearman | MSE | Notes |
| --- | --- | ---: | ---: | --- |
| Synthetic skeleton | Temporal MLP | 0.0786188914 | 4651.189384 | Interface smoke test only. |
| Synthetic skeleton | ST-GCN-style | 0.1560733323 | 4641.265818 | Interface smoke test only. |
