from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evaluate import evaluate_model
from .model import OneHiddenLayerMLP

# 训练器配置，包含训练超参数
@dataclass
class TrainerConfig:
    epochs: int = 20           # 训练轮数
    batch_size: int = 128      # 每批样本数
    learning_rate: float = 0.05
    lr_decay: float = 0.95     # 学习率衰减
    weight_decay: float = 1e-4 # L2正则
    seed: int = 42
    save_path: str = "checkpoints/best_model.npz"  # 最优模型保存路径

# 小批量数据生成器
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

# 训练主流程
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

    min_lr = 1e-5
    for epoch in range(cfg.epochs):
        lr = max(cfg.learning_rate * (cfg.lr_decay ** epoch), min_lr)  # 学习率衰减，设下限
        losses = []

        # 遍历所有小批量
        for xb, yb in _iterate_minibatches(x_train, y_train, cfg.batch_size, rng):
            cache = model.forward(xb)  # 前向传播
            loss = model.backward(cache, yb, weight_decay=cfg.weight_decay)  # 反向传播
            model.step(lr)  # 参数更新
            losses.append(loss)

        train_loss = float(np.mean(losses))
        val_loss, val_acc = evaluate_model(model, x_val, y_val)  # 验证集评估

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(str(save_path))

        # 只在每10个epoch、首尾epoch输出
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == cfg.epochs:
            print(
                f"Epoch {epoch + 1:02d}/{cfg.epochs} | "
                f"lr={lr:.5f} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )

    return history
