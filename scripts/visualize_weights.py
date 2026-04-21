from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from hw1.model import OneHiddenLayerMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize first layer weights as image patterns")
    parser.add_argument("--model-path", type=str, default="checkpoints/best_model.npz")
    parser.add_argument("--hidden-dim1", type=int, default=256)
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--num-filters", type=int, default=64)
    # 只保留 per_class 选择方式
    parser.add_argument(
        "--selection",
        choices=["per_class"],
        default="per_class",
        help="Visualize most discriminative hidden units for each class (per_class).",
    )
    parser.add_argument("--normalize", choices=["none", "per_filter"], default="per_filter")
    parser.add_argument("--clip", type=float, default=2.0, help="Used when normalize=per_filter.")
    parser.add_argument("--cmap", type=str, default="gray")
    parser.add_argument("--out-path", type=str, default="artifacts/first_layer_weights.png")
    return parser.parse_args()

def _per_filter_normalize(img: np.ndarray, clip: float) -> tuple[np.ndarray, float, float]:
    img = img.astype(np.float32, copy=False)
    img = img - float(img.mean())
    std = float(img.std()) + 1e-8
    img = img / std
    lim = float(clip)
    img = np.clip(img, -lim, lim)
    return img, -lim, lim


def main() -> None:
    args = parse_args()
    model = OneHiddenLayerMLP(
        hidden_dim1=args.hidden_dim1,
        activation=args.activation,
    )
    model.load(args.model_path)

    w1 = model.W1.T  # (H, 784)
    n = min(args.num_filters, w1.shape[0])

    # 只保留 per_class 选择方式
    w2 = model.W2  # (H, C)
    chosen = []
    for c in range(w2.shape[1]):
        # 对每个类别，找绝对值最大的隐藏单元
        j = np.argmax(np.abs(w2[:, c]))
        if j not in chosen:
            chosen.append(j)
    # 若不足n个，补充贡献度高的
    if len(chosen) < n:
        norms = np.linalg.norm(w1, axis=1)
        max_abs_w2 = np.max(np.abs(w2), axis=1)
        scores = max_abs_w2 * norms
        extra = [j for j in np.argsort(-scores) if j not in chosen]
        chosen += extra[: n - len(chosen)]
    idx = chosen[:n]
    w = w1[idx]

    cols = 8
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.ravel()

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            img = w[i].reshape(28, 28)
            if args.normalize == "per_filter":
                img, vmin, vmax = _per_filter_normalize(img, clip=args.clip)
            else:
                vmin, vmax = None, None
            ax.imshow(img, cmap=args.cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            # 将下标放到图片下方
            ax.set_title(str(i + 1), fontsize=10, pad=4)

    plt.suptitle("First Layer Learned Spatial Patterns")
    plt.tight_layout()
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to: {args.out_path}")


if __name__ == "__main__":
    main()
