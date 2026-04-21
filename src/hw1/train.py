from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evaluate import evaluate_model
from .model import OneHiddenLayerMLP

def _lr_exp(base_lr: float, lr_decay: float, epoch: int, min_lr: float) -> float:
    return max(base_lr * (lr_decay**epoch), min_lr)


def _lr_cosine(base_lr: float, epoch: int, total_epochs: int, min_lr: float) -> float:
    # 余弦退火：从初始学习率平滑降到最小学习率
    if total_epochs <= 1:
        return max(base_lr, min_lr)
    t = epoch / float(total_epochs - 1)  # 0 -> 1
    return float(min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * t)))


@dataclass
class TrainerConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 0.05
    lr_decay: float = 0.95
    lr_schedule: str = "cosine"  # 指数衰减 / 余弦退火
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    seed: int = 42
    save_path: str = "checkpoints/best_model.npz"

def _iterate_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
):
    idx = np.arange(len(x))
    rng.shuffle(idx)
    for start in range(0, len(x), batch_size):
        batch_idx = idx[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]

def train_model(
    model: OneHiddenLayerMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainerConfig,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(cfg.seed)
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = -1.0
    save_path = Path(cfg.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        if cfg.lr_schedule == "exp":
            lr = _lr_exp(cfg.learning_rate, cfg.lr_decay, epoch, cfg.min_lr)
        elif cfg.lr_schedule == "cosine":
            lr = _lr_cosine(cfg.learning_rate, epoch, cfg.epochs, cfg.min_lr)
        else:
            raise ValueError(f"Unsupported lr_schedule: {cfg.lr_schedule}")
        losses = []

        for xb, yb in _iterate_minibatches(x_train, y_train, cfg.batch_size, rng):
            cache = model.forward(xb)
            loss = model.backward(cache, yb, weight_decay=cfg.weight_decay)
            model.step(lr)
            losses.append(loss)

        train_loss = float(np.mean(losses))
        val_loss, val_acc = evaluate_model(model, x_val, y_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(str(save_path))

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == cfg.epochs:
            print(
                f"Epoch {epoch + 1:02d}/{cfg.epochs} | "
                f"lr={lr:.5f} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )

    return history
