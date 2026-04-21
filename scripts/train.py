from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt

from hw1.data import load_fashion_mnist
from hw1.model import OneHiddenLayerMLP
from hw1.train import TrainerConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one-hidden-layer MLP on Fashion-MNIST")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--lr-schedule", choices=["exp", "cosine"], default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim1", type=int, default=256)
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="checkpoints/best_model.npz")
    parser.add_argument("--history-path", type=str, default="artifacts/history.json")
    parser.add_argument("--curve-path", type=str, default="artifacts/curves.png")
    return parser.parse_args()


def plot_curves(history: dict[str, list[float]], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], color="tab:green", label="Val Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_fashion_mnist(data_dir=args.data_dir)

    model = OneHiddenLayerMLP(
        input_dim=784,
        hidden_dim1=args.hidden_dim1,
        activation=args.activation,
        seed=args.seed,
    )

    cfg = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        lr_schedule=args.lr_schedule,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        save_path=args.save_path,
    )

    history = train_model(
        model,
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
        cfg,
    )

    Path(args.history_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    plot_curves(history, args.curve_path)
    print(f"Saved best model to: {args.save_path}")
    print(f"Saved history to: {args.history_path}")
    print(f"Saved curves to: {args.curve_path}")


if __name__ == "__main__":
    main()
