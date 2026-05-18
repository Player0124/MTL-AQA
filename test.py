import argparse
import json
import os
import pickle
from pathlib import Path

from dataset import MTLAQADataset, write_predictions
from metrics import mse, safe_float, spearmanr
from model import MLPConfig, NumpyMLPRegressor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate minimal MTL-AQA score regression baseline")
    parser.add_argument("--checkpoint", default="runs/mtl_aqa_minimal/checkpoints/best_model.pt")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--feature-root", default=None)
    parser.add_argument("--feature-mode", choices=["auto", "feature", "annotation"], default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.checkpoint, "rb") as handle:
        checkpoint = pickle.load(handle)
    saved_args = checkpoint["args"]
    data_root = args.data_root or saved_args["data_root"]
    split_file = args.split_file or saved_args["split_file"]
    feature_root = args.feature_root if args.feature_root is not None else saved_args.get("feature_root")
    feature_mode = args.feature_mode or saved_args.get("feature_mode", "auto")
    output_dir = Path(args.output_dir or saved_args["output_dir"]) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MTLAQADataset(data_root, split_file, "test", feature_root, feature_mode)
    x = (dataset.x - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    y = dataset.y
    model = NumpyMLPRegressor(
        MLPConfig(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            seed=checkpoint["seed"],
        )
    )
    model.load_state_dict(checkpoint["model_state"])
    preds = model.predict(x)
    rho, p_value = spearmanr(preds, y)
    metrics = {"spearman": safe_float(rho), "spearman_p": safe_float(p_value), "mse": mse(preds, y)}

    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    write_predictions(str(predictions_path), dataset.sample_ids(), preds, y)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"test_spearman={metrics['spearman']}")
    print(f"test_mse={metrics['mse']:.6f}")
    print(f"predictions={predictions_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
