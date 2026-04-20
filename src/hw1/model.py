from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 权重随机初始化（Xavier初始化）
def _xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)

# 激活函数及其反向传播
def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)  # relu激活

def _relu_backward(grad: np.ndarray, x: np.ndarray) -> np.ndarray:
    return grad * (x > 0)      # relu反向传播

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _sigmoid_backward(grad: np.ndarray, x: np.ndarray) -> np.ndarray:
    s = _sigmoid(x)
    return grad * s * (1.0 - s)

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def _tanh_backward(grad: np.ndarray, x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return grad * (1.0 - t * t)

# 激活函数字典
ACTIVATIONS = {
    "relu": (_relu, _relu_backward),
    "sigmoid": (_sigmoid, _sigmoid_backward),
    "tanh": (_tanh, _tanh_backward),
}

# 前向传播中间变量缓存
@dataclass
class ForwardCache:
    x: np.ndarray      # 输入
    z1: np.ndarray     # 第一层线性输出
    a1: np.ndarray     # 第一层激活输出
    logits: np.ndarray # 最后一层输出（未softmax）

# 单隐藏层感知机模型
class OneHiddenLayerMLP:
    def __init__(
        self,
        input_dim: int = 784,      # 输入维度
        hidden_dim1: int = 256,    # 第一隐藏层维度
        num_classes: int = 10,     # 输出类别数
        activation: str = "relu", # 激活函数
        seed: int = 42,
    ) -> None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}")

        self.activation_name = activation
        self.act, self.act_backward = ACTIVATIONS[activation]
        rng = np.random.default_rng(seed)

        # 权重和偏置初始化
        self.W1 = _xavier_init(input_dim, hidden_dim1, rng)
        self.b1 = np.zeros((1, hidden_dim1), dtype=np.float32)
        self.W2 = _xavier_init(hidden_dim1, num_classes, rng)
        self.b2 = np.zeros((1, num_classes), dtype=np.float32)

        self.grads: dict[str, np.ndarray] = {}  # 存储梯度

    def forward(self, x: np.ndarray) -> ForwardCache:
        z1 = x @ self.W1 + self.b1
        a1 = self.act(z1)
        logits = a1 @ self.W2 + self.b2
        return ForwardCache(x=x, z1=z1, a1=a1, logits=logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        cache = self.forward(x)
        return np.argmax(cache.logits, axis=1)

    @staticmethod
    def _cross_entropy_with_logits(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        n = logits.shape[0]
        loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()

        dlogits = probs
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        return float(loss), dlogits

    def backward(
        self,
        cache: ForwardCache,
        y: np.ndarray,
        weight_decay: float = 0.0,
    ) -> float:
        ce_loss, dlogits = self._cross_entropy_with_logits(cache.logits, y)
        reg = 0.5 * weight_decay * (
            np.sum(self.W1 * self.W1)
            + np.sum(self.W2 * self.W2)
        )

        da1 = dlogits @ self.W2.T
        dW2 = cache.a1.T @ dlogits + weight_decay * self.W2
        db2 = np.sum(dlogits, axis=0, keepdims=True)

        dz1 = self.act_backward(da1, cache.z1)
        dW1 = cache.x.T @ dz1 + weight_decay * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }
        return float(ce_loss + reg)

    def step(self, lr: float) -> None:
        self.W1 -= lr * self.grads["W1"]
        self.b1 -= lr * self.grads["b1"]
        self.W2 -= lr * self.grads["W2"]
        self.b2 -= lr * self.grads["b2"]

    def save(self, path: str) -> None:
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            activation=self.activation_name,
        )

    def load(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
