# MTL-AQA Reproduction Plan

## Repository

- Official GitHub repository used as the starting point: https://github.com/ParitoshParmar/MTL-AQA
- Paper: "What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment", CVPR 2019.
- Current local checkout contains:
  - `MTL-AQA_code_release/`: official C3D-AVG and MSCADC training code.
  - `MTL-AQA_dataset_release/`: annotation files, official split files, vocabulary, and raw annotation spreadsheets.

## Environment Dependencies

Official code appears to target an older Python/PyTorch stack:

- Python 3.x, likely Python 3.6/3.7 era.
- PyTorch with CUDA; official scripts call `.cuda()` directly and do not support CPU without edits.
- torchvision, PIL/Pillow, numpy, scipy.
- C3D pretrained weights:
  - `c3d.pickle` for C3D-AVG, linked in the official code readme.
  - `C3D_small_PyTorch_Trained_12.pth` for MSCADC.

Current machine state:

- System `python` is not on PATH.
- Bundled Codex Python is available at `C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`.
- Bundled Python has numpy and pandas, but no torch, torchvision, scipy, or PyYAML.
- The minimal baseline added here intentionally depends only on Python stdlib plus numpy.

## Official Data Directory Expected By Release Code

The official dataloaders expect:

```text
<data_root>/
  Ready_2_Use/
    MTL-AQA_split_0_data/
      final_annotations_dict.pkl
      final_captions_dict.pkl
      train_split_0.pkl
      test_split_0.pkl
      vocab.json
  frames/
    01/
      *.jpg
    02/
      *.jpg
    ...
```

The current repository includes annotation and split files, but not extracted video frames.

Official split inspected:

- `final_annotations_dict.pkl`: 1412 samples.
- `train_split_0.pkl`: 1059 samples.
- `test_split_0.pkl`: 353 samples.
- Sample key format: `(video_id, sample_id)`, for example `(18, 67)`.
- Annotation fields include `start_frame`, `end_frame`, dive class labels, `difficulty`, and `final_score`.

## Official Baseline Commands

Official files:

- C3D-AVG: `MTL-AQA_code_release/train_test_C3DAVG.py`
- MSCADC: `MTL-AQA_code_release/train_test_MSCADC.py`
- Options: `MTL-AQA_code_release/opts.py`

Expected official workflow after manually editing `opts.py`:

```powershell
cd MTL-AQA_code_release
python train_test_C3DAVG.py
python train_test_MSCADC.py
```

Required edits before those scripts can run:

- Replace placeholder paths in `opts.py`:
  - `dataset_dir`
  - `anno_n_splits_dir`
  - `dataset_frames_dir`
- Set C3D-AVG dimensions to `C, H, W = 3,112,112` and `input_resize = 171,128` if using C3D-AVG.
- Set MSCADC dimensions to `C, H, W = 3,180,180` and `input_resize = 640,360` if using MSCADC.
- Provide the correct pretrained weight file in the script working directory.
- Provide extracted frames named by video serial number.

## Official Pipeline Blockers

- `opts.py` ships with placeholder paths (`'...'`).
- Official scripts hard-code CUDA via `.cuda()` and cannot do CPU dry-run as written.
- Current checkout does not include extracted frames.
- Current checkout does not include complete pretrained C3D weight files in directly loadable `.pth` / `.pickle` form.
- Current Python environment lacks PyTorch/torchvision/scipy.
- `dataloader_MSCADC.py` has a likely bug in `load_image`: `image.resize(...)` is called without assigning the result.

Because of these blockers, the current deliverable is a simplified single-task baseline rather than an official full reproduction.

## Minimal Baseline Implemented

Added files:

- `dataset.py`: loads official annotation/split pickle files; can load external `.npy`/`.npz` features when available.
- `model.py`: one-hidden-layer numpy MLP regressor.
- `metrics.py`: MSE and Spearman rank correlation; uses scipy if installed, otherwise uses an internal rank fallback.
- `train.py`: train/evaluate loop, checkpointing, logging, predictions, metrics.
- `test.py`: reloads checkpoint and writes test predictions/metrics.
- `config.yaml`: default paths and hyperparameters.
- `README_reproduction.md`: full reproduction notes.

Default baseline mode:

- Uses official train/test split.
- Predicts `final_score`.
- Uses MSE training loss.
- Reports train loss, test Spearman, and test MSE every epoch.
- Saves:
  - `runs/mtl_aqa_minimal/checkpoints/best_model.pt`
  - `runs/mtl_aqa_minimal/outputs/predictions.csv`
  - `runs/mtl_aqa_minimal/outputs/metrics.json`
  - `runs/mtl_aqa_minimal/logs/train.log`

Important limitation:

- Since current checkout has no video frames or C3D/I3D features, the runnable default uses annotation metadata as a dry-run input. This validates the score-regression loop and official split handling, but it is not comparable to the paper's video-based C3D-AVG-MTL result.

## Commands Run Successfully

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' train.py --epochs 1 --batch-size 128 --output-dir runs\smoke_test
```

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' train.py --epochs 20 --batch-size 64 --output-dir runs\mtl_aqa_minimal
```

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' test.py --checkpoint runs\mtl_aqa_minimal\checkpoints\best_model.pt
```

## Current Result

| Run | Input | Epochs | Test Spearman | Test MSE | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `runs/mtl_aqa_minimal` | annotation metadata dry-run | 20 | 0.3680757962 | 666.235943 | simplified single-task baseline |

Paper reference:

- C3D-AVG-MTL rank correlation: 90.44%.
- Current gap: about 53.63 Spearman percentage points.
- Main reason: this run is not using the paper's video/C3D feature pipeline or multitask losses.

## Next Steps

1. Install a compatible PyTorch + torchvision environment.
2. Download or reconstruct the C3D pretrained weights expected by official scripts.
3. Download source videos from `Video_List.xlsx` and extract frames with `frame_extractor.sh`.
4. Edit `opts.py` paths and run a 1-epoch official C3D-AVG or MSCADC smoke test.
5. If using external video/skeleton features, export one `.npy` or `.npz` feature per official sample key and train with `--feature-root <feature_root> --feature-mode feature`.
