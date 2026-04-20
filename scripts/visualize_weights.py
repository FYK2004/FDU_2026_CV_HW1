from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt

from hw1.model import OneHiddenLayerMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize first layer weights as image patterns")
    parser.add_argument("--model-path", type=str, default="checkpoints/best_model.npz")
    parser.add_argument("--hidden-dim1", type=int, default=256)
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--num-filters", type=int, default=64)
    parser.add_argument("--out-path", type=str, default="artifacts/first_layer_weights.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = OneHiddenLayerMLP(
        hidden_dim1=args.hidden_dim1,
        activation=args.activation,
    )
    model.load(args.model_path)

    w = model.W1.T
    n = min(args.num_filters, w.shape[0])
    cols = 8
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.ravel()

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            img = w[i].reshape(28, 28)
            ax.imshow(img, cmap="gray")

    plt.suptitle("First Layer Learned Spatial Patterns")
    plt.tight_layout()
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to: {args.out_path}")


if __name__ == "__main__":
    main()
