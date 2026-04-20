from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from hw1.data import load_fashion_mnist
from hw1.model import OneHiddenLayerMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save misclassified examples for error analysis")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-path", type=str, default="checkpoints/best_model.npz")
    parser.add_argument("--hidden-dim1", type=int, default=256)
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--num-errors", type=int, default=16)
    parser.add_argument("--out-path", type=str, default="artifacts/error_cases.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_fashion_mnist(data_dir=args.data_dir)
    classes = data["classes"]

    model = OneHiddenLayerMLP(
        hidden_dim1=args.hidden_dim1,
        activation=args.activation,
    )
    model.load(args.model_path)

    y_pred = model.predict(data["x_test"])
    y_true = data["y_test"]
    wrong_idx = np.where(y_pred != y_true)[0]

    n = min(args.num_errors, len(wrong_idx))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            idx = wrong_idx[i]
            ax.imshow(data["x_test_images"][idx], cmap="gray")
            ax.set_title(
                f"T:{classes[y_true[idx]]}\nP:{classes[y_pred[idx]]}",
                fontsize=9,
            )

    plt.tight_layout()
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_path, dpi=150)
    plt.close(fig)
    print(f"Saved error cases to: {args.out_path}")


if __name__ == "__main__":
    main()
