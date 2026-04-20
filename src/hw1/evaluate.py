from __future__ import annotations

import numpy as np

from .model import OneHiddenLayerMLP

# 评估模型在给定数据上的损失和准确率
def evaluate_model(model: OneHiddenLayerMLP, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # 前向传播，得到各类别的 logits
    cache = model.forward(x)
    # softmax 归一化，防止数值溢出
    shifted = cache.logits - np.max(cache.logits, axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=1, keepdims=True)
    # 交叉熵损失
    loss = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()

    preds = np.argmax(cache.logits, axis=1)  # 预测类别
    acc = float((preds == y).mean())         # 计算准确率
    return float(loss), acc

# 计算混淆矩阵
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm
