from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw1.data import load_fashion_mnist
from hw1.model import OneHiddenLayerMLP
from hw1.train import TrainerConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for Fashion-MNIST MLP hyperparameters")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-path", type=str, default="artifacts/search_results.json")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not auto-generate per-activation heatmaps after search finishes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_fashion_mnist(data_dir=args.data_dir)

    grid = {
        "learning_rate": [0.01, 0.05],
        "hidden_dim1": [128, 256],
        "weight_decay": [ 1e-4, 5e-4],
        "activation": ["sigmoid", "relu", "tanh"],
    }

    results = []
    best = None

    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        hp = dict(zip(keys, values))
        print(f"Running config: {hp}")

        model = OneHiddenLayerMLP(
            hidden_dim1=hp["hidden_dim1"],
            activation=hp["activation"],
            seed=args.seed,
        )
        cfg = TrainerConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=hp["learning_rate"],
            lr_decay=0.95,
            weight_decay=hp["weight_decay"],
            seed=args.seed,
            save_path="checkpoints/grid_tmp_best.npz",
        )

        history = train_model(
            model,
            data["x_train"],
            data["y_train"],
            data["x_val"],
            data["y_val"],
            cfg,
        )
        score = max(history["val_acc"])
        item = {"config": hp, "best_val_acc": score}
        results.append(item)

        if best is None or score > best["best_val_acc"]:
            best = item

    Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_path, "w", encoding="utf-8") as f:
        json.dump({"best": best, "results": results}, f, ensure_ascii=False, indent=2)

    print("Best config:")
    print(best)
    print(f"Saved search results to: {args.results_path}")

    if not args.no_plot:
        plot_script = ROOT / "scripts" / "plot_search_results.py"
        print("Generating per-activation heatmaps...")
        try:
            subprocess.run(
                [sys.executable, str(plot_script), str(args.results_path)],
                check=True,
            )
            print("Done. Saved: artifacts/search_heatmap_<activation>.png")
        except Exception as e:
            print(f"[WARN] Plot generation failed: {e}")


if __name__ == "__main__":
    main()
