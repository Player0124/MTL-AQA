import csv
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


SampleKey = Tuple[int, int]


ANNOTATION_FEATURES = (
    "primary_view",
    "start_frame",
    "end_frame",
    "position",
    "difficulty",
    "armstand",
    "rotation_type",
    "ss_no",
    "tw_no",
)


class MTLAQADataset:
    def __init__(
        self,
        data_root: str,
        split_file: str,
        mode: str,
        feature_root: Optional[str] = None,
        feature_mode: str = "auto",
    ):
        if mode not in {"train", "test"}:
            raise ValueError("mode must be 'train' or 'test'")
        self.data_root = Path(data_root)
        self.split_dir = _resolve_split_dir(self.data_root, split_file)
        self.mode = mode
        self.feature_root = Path(feature_root) if feature_root else None
        self.feature_mode = feature_mode

        self.annotations: Dict[SampleKey, Dict[str, float]] = _load_pickle(
            self.split_dir / "final_annotations_dict.pkl"
        )
        self.keys: List[SampleKey] = _load_split_keys(self.split_dir, mode)
        self.samples = [self._make_sample(key) for key in self.keys]

    def _make_sample(self, key: SampleKey) -> Dict[str, object]:
        annotation = self.annotations[key]
        feature = self._load_feature(key, annotation)
        return {
            "key": key,
            "feature": feature.astype(np.float64),
            "target": float(annotation["final_score"]),
        }

    def _load_feature(self, key: SampleKey, annotation: Dict[str, float]) -> np.ndarray:
        if self.feature_mode == "annotation":
            return _annotation_feature(annotation)
        if self.feature_root:
            loaded = _try_load_feature_file(self.feature_root, key)
            if loaded is not None:
                return loaded
            if self.feature_mode == "feature":
                raise FileNotFoundError(
                    f"No feature file found for key {key} under {self.feature_root}"
                )
        return _annotation_feature(annotation)

    @property
    def x(self) -> np.ndarray:
        return np.stack([sample["feature"] for sample in self.samples], axis=0)

    @property
    def y(self) -> np.ndarray:
        return np.asarray([sample["target"] for sample in self.samples], dtype=np.float64)

    def sample_ids(self) -> List[str]:
        return [format_key(sample["key"]) for sample in self.samples]


def _resolve_split_dir(data_root: Path, split_file: str) -> Path:
    split_path = Path(split_file)
    candidates = [split_path]
    if not split_path.is_absolute():
        candidates.append(data_root / split_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.parent
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not resolve split path: {split_file}")


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_split_keys(split_dir: Path, mode: str) -> List[SampleKey]:
    preferred = split_dir / f"{mode}_split_0.pkl"
    if preferred.exists():
        return _load_pickle(preferred)
    matches = sorted(split_dir.glob(f"{mode}_split_*.pkl"))
    if not matches:
        raise FileNotFoundError(f"No {mode}_split_*.pkl found in {split_dir}")
    return _load_pickle(matches[0])


def _annotation_feature(annotation: Dict[str, float]) -> np.ndarray:
    values = []
    duration = float(annotation["end_frame"]) - float(annotation["start_frame"])
    for name in ANNOTATION_FEATURES:
        values.append(float(annotation[name]))
    values.append(duration)
    return np.asarray(values, dtype=np.float64)


def _try_load_feature_file(feature_root: Path, key: SampleKey) -> Optional[np.ndarray]:
    video_id, sample_id = key
    stems = [
        f"{video_id:02d}_{sample_id}",
        f"{video_id}_{sample_id}",
        f"{video_id:02d}-{sample_id}",
        f"{video_id}-{sample_id}",
    ]
    paths = []
    for stem in stems:
        paths.extend([feature_root / f"{stem}.npy", feature_root / f"{stem}.npz"])
    paths.extend(
        [
            feature_root / f"{video_id:02d}" / f"{sample_id}.npy",
            feature_root / f"{video_id:02d}" / f"{sample_id}.npz",
            feature_root / str(video_id) / f"{sample_id}.npy",
            feature_root / str(video_id) / f"{sample_id}.npz",
        ]
    )
    for path in paths:
        if path.exists():
            array = _load_array(path)
            return _pool_feature(array)
    return None


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        npz = np.load(path)
        for name in ("feature", "features", "arr_0"):
            if name in npz:
                return np.asarray(npz[name], dtype=np.float64)
        first_key = list(npz.keys())[0]
        return np.asarray(npz[first_key], dtype=np.float64)
    return np.asarray(np.load(path), dtype=np.float64)


def _pool_feature(array: np.ndarray) -> np.ndarray:
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    return array.reshape(array.shape[0], -1).mean(axis=0)


def format_key(key: SampleKey) -> str:
    return f"{key[0]:02d}_{key[1]}"


def standardize_train_test(train_x: np.ndarray, test_x: np.ndarray):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


def write_predictions(path: str, sample_ids: Iterable[str], predictions, targets) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "prediction", "ground_truth"])
        for sample_id, pred, target in zip(sample_ids, predictions, targets):
            writer.writerow([sample_id, f"{float(pred):.6f}", f"{float(target):.6f}"])
