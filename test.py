import argparse
import json
import os
import pickle
from pathlib import Path

from dataset import MTLAQADataset, MTLAQASkeletonDataset, write_predictions
from metrics import mse, safe_float, spearmanr
from model import MLPConfig, NumpyMLPRegressor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate minimal MTL-AQA score regression baseline")
    parser.add_argument("--checkpoint", default="runs/mtl_aqa_minimal/checkpoints/best_model.pt")
    parser.add_argument("--input-type", choices=["video_feature", "skeleton"], default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--feature-root", default=None)
    parser.add_argument("--feature-mode", choices=["auto", "feature", "annotation"], default=None)
    parser.add_argument("--skeleton-root", default=None)
    parser.add_argument("--skeleton-length", type=int, default=None)
    parser.add_argument("--skeleton-dims", type=int, choices=[2, 3], default=None)
    parser.add_argument("--model", choices=["mlp", "temporal_mlp", "stgcn"], default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.checkpoint, "rb") as handle:
        checkpoint = pickle.load(handle)
    saved_args = checkpoint["args"]
    data_root = args.data_root or saved_args["data_root"]
    split_file = args.split_file or saved_args["split_file"]
    input_type = args.input_type or saved_args.get("input_type", "video_feature")
    feature_root = args.feature_root if args.feature_root is not None else saved_args.get("feature_root")
    feature_mode = args.feature_mode or saved_args.get("feature_mode", "auto")
    skeleton_root = args.skeleton_root if args.skeleton_root is not None else saved_args.get("skeleton_root")
    skeleton_length = args.skeleton_length or saved_args.get("skeleton_length", 103)
    skeleton_dims = args.skeleton_dims or saved_args.get("skeleton_dims", 3)
    model_name = args.model or saved_args.get("model", "mlp")
    output_dir = Path(args.output_dir or saved_args["output_dir"]) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_type == "skeleton":
        if not skeleton_root:
            raise ValueError("--skeleton-root is required when evaluating a skeleton checkpoint")
        skeleton_encoding = model_name
        if skeleton_encoding == "mlp":
            skeleton_encoding = "temporal_mlp"
        dataset = MTLAQASkeletonDataset(
            data_root,
            split_file,
            "test",
            skeleton_root,
            skeleton_length=skeleton_length,
            skeleton_dims=skeleton_dims,
            skeleton_encoding=skeleton_encoding,
        )
    else:
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
