from __future__ import annotations

import numpy as np

from .model import OneHiddenLayerMLP

def evaluate_model(model: OneHiddenLayerMLP, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    cache = model.forward(x)
    # 归一化前先做平移，避免指数函数溢出
    shifted = cache.logits - np.max(cache.logits, axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()

    preds = np.argmax(cache.logits, axis=1)
    acc = float((preds == y).mean())
    return float(loss), acc

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm
