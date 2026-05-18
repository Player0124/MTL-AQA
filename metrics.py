import math
from typing import Iterable, Tuple

import numpy as np


def mse(predictions: Iterable[float], targets: Iterable[float]) -> float:
    preds = np.asarray(list(predictions), dtype=np.float64)
    gts = np.asarray(list(targets), dtype=np.float64)
    return float(np.mean((preds - gts) ** 2))


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, compatible with scipy.stats.spearmanr semantics."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearmanr(predictions: Iterable[float], targets: Iterable[float]) -> Tuple[float, float]:
    preds = np.asarray(list(predictions), dtype=np.float64)
    gts = np.asarray(list(targets), dtype=np.float64)
    try:
        from scipy.stats import spearmanr as scipy_spearmanr

        rho, p_value = scipy_spearmanr(preds, gts)
        return float(rho), float(p_value)
    except Exception:
        if len(preds) < 2:
            return float("nan"), float("nan")
        pred_ranks = _rankdata(preds)
        gt_ranks = _rankdata(gts)
        pred_std = pred_ranks.std()
        gt_std = gt_ranks.std()
        if pred_std == 0.0 or gt_std == 0.0:
            return float("nan"), float("nan")
        rho = float(np.corrcoef(pred_ranks, gt_ranks)[0, 1])
        return rho, float("nan")


def safe_float(value: float) -> float:
    return None if isinstance(value, float) and math.isnan(value) else float(value)
