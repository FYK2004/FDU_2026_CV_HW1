from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np

DATASET_URLS = {
    "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "test_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}

def _download(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return
    urllib.request.urlretrieve(url, target_path)

def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image file: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)

def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label file: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num)

def load_fashion_mnist(
    data_dir: str | os.PathLike = "data",
    validation_split: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    data_dir = Path(data_dir)
    paths = {
        key: data_dir / Path(url).name for key, url in DATASET_URLS.items()
    }

    for key, url in DATASET_URLS.items():
        _download(url, paths[key])

    train_images = _read_images(paths["train_images"]).astype(np.float32) / 255.0
    train_labels = _read_labels(paths["train_labels"]).astype(np.int64)
    test_images = _read_images(paths["test_images"]).astype(np.float32) / 255.0
    test_labels = _read_labels(paths["test_labels"]).astype(np.int64)

    rng = np.random.default_rng(seed)
    n_train = train_images.shape[0]
    indices = np.arange(n_train)
    rng.shuffle(indices)

    n_val = int(n_train * validation_split)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    x_train = train_images[train_idx].reshape(len(train_idx), -1)
    y_train = train_labels[train_idx]
    x_val = train_images[val_idx].reshape(len(val_idx), -1)
    y_val = train_labels[val_idx]

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": test_images.reshape(test_images.shape[0], -1),
        "y_test": test_labels,
        "x_test_images": test_images,
        "classes": np.array(
            [
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ]
        ),
    }
