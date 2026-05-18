import argparse
import json
import os
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np

from dataset import MTLAQADataset, standardize_train_test, write_predictions
from metrics import mse, safe_float, spearmanr
from model import MLPConfig, NumpyMLPRegressor


def parse_args():
    config = load_config("config.yaml")
    parser = argparse.ArgumentParser(description="Minimal MTL-AQA score regression baseline")
    parser.add_argument("--data-root", default=config.get("data_root", "MTL-AQA_dataset_release/Ready_2_Use"))
    parser.add_argument(
        "--split-file",
        default=config.get("split_file", "MTL-AQA_split_0_data"),
        help="Split directory or a train/test split pickle inside that directory.",
    )
    parser.add_argument("--feature-root", default=config.get("feature_root"))
    parser.add_argument(
        "--feature-mode",
        choices=["auto", "feature", "annotation"],
        default=config.get("feature_mode", "auto"),
    )
    parser.add_argument("--epochs", type=int, default=int(config.get("epochs", 20)))
    parser.add_argument("--batch-size", type=int, default=int(config.get("batch_size", 64)))
    parser.add_argument("--lr", type=float, default=float(config.get("lr", 1e-3)))
    parser.add_argument("--seed", type=int, default=int(config.get("seed", 0)))
    parser.add_argument("--hidden-dim", type=int, default=int(config.get("hidden_dim", 128)))
    parser.add_argument("--output-dir", default=config.get("output_dir", "runs/mtl_aqa_minimal"))
    return parser.parse_args()


def load_config(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {}
    values: Dict[str, object] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = parse_scalar(value.strip())
    return values


def parse_scalar(value: str):
    if value in {"", "null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def evaluate(model: NumpyMLPRegressor, x: np.ndarray, y: np.ndarray):
    preds = model.predict(x)
    rho, p_value = spearmanr(preds, y)
    return preds, {"spearman": safe_float(rho), "spearman_p": safe_float(p_value), "mse": mse(preds, y)}


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    outputs_dir = output_dir / "outputs"
    for directory in (ckpt_dir, logs_dir, outputs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"started_at: {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"device: cpu-numpy\n")
        log.write(f"args: {vars(args)}\n")

    train_data = MTLAQADataset(
        args.data_root, args.split_file, "train", args.feature_root, args.feature_mode
    )
    test_data = MTLAQADataset(
        args.data_root, args.split_file, "test", args.feature_root, args.feature_mode
    )
    train_x, test_x, feature_mean, feature_std = standardize_train_test(train_data.x, test_data.x)
    train_y = train_data.y
    test_y = test_data.y

    model = NumpyMLPRegressor(
        MLPConfig(input_dim=train_x.shape[1], hidden_dim=args.hidden_dim, seed=args.seed)
    )
    best_spearman = -np.inf
    best_metrics = None
    best_preds = None
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        indices = rng.permutation(len(train_x))
        losses = []
        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            losses.append(model.train_batch(train_x[batch_idx], train_y[batch_idx], args.lr))
        train_preds, train_metrics = evaluate(model, train_x, train_y)
        test_preds, test_metrics = evaluate(model, test_x, test_y)
        epoch_line = (
            f"epoch={epoch:03d} train_loss={float(np.mean(losses)):.6f} "
            f"train_spearman={train_metrics['spearman']} train_mse={train_metrics['mse']:.6f} "
            f"test_spearman={test_metrics['spearman']} test_mse={test_metrics['mse']:.6f}"
        )
        print(epoch_line)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(epoch_line + "\n")

        current_spearman = test_metrics["spearman"]
        if current_spearman is not None and current_spearman > best_spearman:
            best_spearman = current_spearman
            best_metrics = test_metrics
            best_preds = test_preds.copy()
            save_checkpoint(
                ckpt_dir / "best_model.pt",
                model,
                args,
                feature_mean,
                feature_std,
                train_x.shape[1],
                best_metrics,
            )

    if best_preds is None:
        best_preds, best_metrics = evaluate(model, test_x, test_y)
    predictions_path = outputs_dir / "predictions.csv"
    metrics_path = outputs_dir / "metrics.json"
    write_predictions(str(predictions_path), test_data.sample_ids(), best_preds, test_y)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(best_metrics, handle, indent=2)

    print(f"best_test_spearman={best_metrics['spearman']}")
    print(f"best_test_mse={best_metrics['mse']:.6f}")
    print(f"checkpoint={ckpt_dir / 'best_model.pt'}")
    print(f"predictions={predictions_path}")
    print(f"metrics={metrics_path}")


def save_checkpoint(path, model, args, feature_mean, feature_std, input_dim, metrics):
    state = {
        "model_state": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "args": vars(args),
        "metrics": metrics,
    }
    with open(path, "wb") as handle:
        pickle.dump(state, handle)


if __name__ == "__main__":
    main()
