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


class MTLAQASkeletonDataset:
    """MTL-AQA dataset wrapper for pre-extracted 2D skeleton sequences.

    Expected skeleton file shape is [T, V, C], where C is 2 or 3:
    x/y coordinates and optional confidence. Files are resolved by the same
    official split sample keys used by MTLAQADataset.
    """

    def __init__(
        self,
        data_root: str,
        split_file: str,
        mode: str,
        skeleton_root: str,
        skeleton_length: int = 103,
        skeleton_dims: int = 3,
        skeleton_encoding: str = "temporal_mlp",
        confidence_threshold: float = 0.0,
    ):
        if mode not in {"train", "test"}:
            raise ValueError("mode must be 'train' or 'test'")
        if skeleton_encoding not in {"temporal_mlp", "stgcn"}:
            raise ValueError("skeleton_encoding must be 'temporal_mlp' or 'stgcn'")
        self.data_root = Path(data_root)
        self.split_dir = _resolve_split_dir(self.data_root, split_file)
        self.mode = mode
        self.skeleton_root = Path(skeleton_root)
        self.skeleton_length = skeleton_length
        self.skeleton_dims = skeleton_dims
        self.skeleton_encoding = skeleton_encoding
        self.confidence_threshold = confidence_threshold

        self.annotations: Dict[SampleKey, Dict[str, float]] = _load_pickle(
            self.split_dir / "final_annotations_dict.pkl"
        )
        self.keys: List[SampleKey] = _load_split_keys(self.split_dir, mode)
        self.samples = [self._make_sample(key) for key in self.keys]

    def _make_sample(self, key: SampleKey) -> Dict[str, object]:
        skeleton = _load_skeleton_file(self.skeleton_root, key)
        skeleton = _prepare_skeleton_sequence(
            skeleton,
            target_length=self.skeleton_length,
            output_dims=self.skeleton_dims,
            confidence_threshold=self.confidence_threshold,
        )
        feature = encode_skeleton(skeleton, self.skeleton_encoding)
        return {
            "key": key,
            "feature": feature.astype(np.float64),
            "target": float(self.annotations[key]["final_score"]),
        }

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


def _candidate_sample_paths(root: Path, key: SampleKey) -> List[Path]:
    video_id, sample_id = key
    stems = [
        f"{video_id:02d}_{sample_id}",
        f"{video_id}_{sample_id}",
        f"{video_id:02d}-{sample_id}",
        f"{video_id}-{sample_id}",
    ]
    paths = []
    for stem in stems:
        paths.extend([root / f"{stem}.npy", root / f"{stem}.npz"])
    paths.extend(
        [
            root / f"{video_id:02d}" / f"{sample_id}.npy",
            root / f"{video_id:02d}" / f"{sample_id}.npz",
            root / str(video_id) / f"{sample_id}.npy",
            root / str(video_id) / f"{sample_id}.npz",
        ]
    )
    return paths


def _load_skeleton_file(skeleton_root: Path, key: SampleKey) -> np.ndarray:
    for path in _candidate_sample_paths(skeleton_root, key):
        if path.exists():
            array = _load_array(path)
            if array.ndim != 3:
                raise ValueError(f"Skeleton file {path} must have shape [T, V, C], got {array.shape}")
            if array.shape[2] < 2:
                raise ValueError(f"Skeleton file {path} must have at least x/y channels")
            return array
    raise FileNotFoundError(f"No skeleton .npy/.npz found for sample {format_key(key)} under {skeleton_root}")


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


def _prepare_skeleton_sequence(
    skeleton: np.ndarray,
    target_length: int,
    output_dims: int,
    confidence_threshold: float,
) -> np.ndarray:
    if output_dims not in {2, 3}:
        raise ValueError("skeleton_dims must be 2 or 3")
    skeleton = np.asarray(skeleton, dtype=np.float64)
    skeleton = _pad_or_truncate_time(skeleton, target_length)
    xy = skeleton[:, :, :2]
    confidence = _extract_confidence(skeleton, confidence_threshold)
    finite = np.isfinite(xy).all(axis=2)
    non_zero = np.abs(xy).sum(axis=2) > 1e-12
    valid = finite & non_zero & (confidence > confidence_threshold)

    normalized_xy = np.zeros_like(xy)
    if valid.any():
        valid_xy = xy[valid]
        center = valid_xy.mean(axis=0)
        centered = xy - center
        centered[~np.isfinite(centered)] = 0.0
        scale = np.nanstd(centered[valid])
        if not np.isfinite(scale) or scale < 1e-6:
            scale = np.nanmax(np.abs(centered[valid]))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        normalized_xy = centered / scale
        normalized_xy[~valid] = 0.0

    if output_dims == 2:
        return normalized_xy
    return np.concatenate([normalized_xy, valid[:, :, None].astype(np.float64)], axis=2)


def _pad_or_truncate_time(sequence: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    frames = sequence.shape[0]
    if frames == target_length:
        return sequence
    if frames > target_length:
        start = (frames - target_length) // 2
        return sequence[start : start + target_length]
    padded = np.zeros((target_length, sequence.shape[1], sequence.shape[2]), dtype=sequence.dtype)
    padded[:frames] = sequence
    return padded


def _extract_confidence(skeleton: np.ndarray, confidence_threshold: float) -> np.ndarray:
    if skeleton.shape[2] >= 3:
        confidence = skeleton[:, :, 2]
        confidence = np.where(np.isfinite(confidence), confidence, 0.0)
        return confidence
    return np.ones(skeleton.shape[:2], dtype=np.float64) * max(confidence_threshold + 1.0, 1.0)


def encode_skeleton(skeleton: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == "temporal_mlp":
        return _encode_skeleton_temporal_mlp(skeleton)
    if encoding == "stgcn":
        return _encode_skeleton_stgcn(skeleton)
    raise ValueError(f"Unsupported skeleton encoding: {encoding}")


def _encode_skeleton_temporal_mlp(skeleton: np.ndarray) -> np.ndarray:
    frame_features = skeleton.reshape(skeleton.shape[0], -1)
    velocity = np.diff(frame_features, axis=0, prepend=frame_features[:1])
    return np.concatenate(
        [
            frame_features.mean(axis=0),
            frame_features.std(axis=0),
            np.abs(velocity).mean(axis=0),
        ],
        axis=0,
    )


def _encode_skeleton_stgcn(skeleton: np.ndarray) -> np.ndarray:
    adjacency = _skeleton_adjacency(skeleton.shape[1])
    spatial = np.einsum("vw,twc->tvc", adjacency, skeleton)
    temporal = np.diff(spatial, axis=0, prepend=spatial[:1])
    acceleration = np.diff(temporal, axis=0, prepend=temporal[:1])
    graph_features = np.concatenate([spatial, temporal, acceleration], axis=2)
    per_joint = graph_features.mean(axis=0)
    per_joint_std = graph_features.std(axis=0)
    global_pool = graph_features.mean(axis=(0, 1))
    return np.concatenate([per_joint.reshape(-1), per_joint_std.reshape(-1), global_pool], axis=0)


def _skeleton_adjacency(num_joints: int) -> np.ndarray:
    adjacency = np.eye(num_joints, dtype=np.float64)
    for left, right in _default_edges(num_joints):
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    degree = adjacency.sum(axis=1, keepdims=True)
    degree[degree == 0.0] = 1.0
    return adjacency / degree


def _default_edges(num_joints: int) -> List[Tuple[int, int]]:
    coco17_edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]
    if num_joints >= 17:
        return coco17_edges
    return [(i, i + 1) for i in range(max(num_joints - 1, 0))]


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
