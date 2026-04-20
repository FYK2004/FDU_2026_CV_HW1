from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np


from hw1.data import load_fashion_mnist
from hw1.evaluate import confusion_matrix
from hw1.model import OneHiddenLayerMLP
from plot_utils import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained MLP on Fashion-MNIST test set")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-path", type=str, default="checkpoints/best_model.npz")
    parser.add_argument("--hidden-dim1", type=int, default=256)
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_fashion_mnist(data_dir=args.data_dir)

    model = OneHiddenLayerMLP(
        hidden_dim1=args.hidden_dim1,
        activation=args.activation,
    )
    model.load(args.model_path)

    y_pred = model.predict(data["x_test"])
    y_true = data["y_test"]

    acc = float((y_pred == y_true).mean())
    cm = confusion_matrix(y_true, y_pred, num_classes=10)

    np.set_printoptions(linewidth=160)
    print(f"Test Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # 输出混淆矩阵图片
    class_names = [str(i) for i in range(10)]
    save_path = "artifacts/confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, save_path)
    print(f"Confusion matrix image saved to {save_path}")


if __name__ == "__main__":
    main()
