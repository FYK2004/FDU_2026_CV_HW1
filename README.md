
# FDU 2026 春季 计算机视觉 HW1（Fashion-MNIST / NumPy）

本仓库用 **纯 NumPy** 从零实现 **单隐藏层 MLP**（含前向、反向、SGD 更新），完成 PDF 要求的训练、评估、超参搜索与可视化产物导出。

---

## 快速开始（复现推荐配置）

1) 安装依赖

```powershell
pip install -r requirements.txt
```

2) 训练并保存验证集最优模型

```powershell
python scripts/train.py --epochs 100 --batch-size 128 --learning-rate 0.05 --lr-schedule cosine --min-lr 1e-5 --weight-decay 1e-4 --hidden-dim1 256 --activation relu
```

3) 测试集评估（并导出混淆矩阵）

```powershell
python scripts/test.py --data-dir data --model-path checkpoints/best_model.npz --hidden-dim1 256 --activation relu
```

4) 导出错例图（用于报告分析）

```powershell
python scripts/error_analysis.py --data-dir data --model-path checkpoints/best_model.npz --hidden-dim1 256 --activation relu --num-errors 4 --out-path artifacts/error_cases.png
```

首次运行会自动下载 Fashion-MNIST 到 `data/`。

---

## 项目结构

```text
.
├── scripts/                # 可直接运行的命令行脚本（对应 PDF 步骤）
│   ├── train.py
│   ├── test.py
│   ├── search.py
│   ├── visualize_weights.py
│   └── error_analysis.py
├── src/hw1/                # 纯 NumPy 实现：数据/模型/训练/评估
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── artifacts/              # 训练曲线/混淆矩阵/热力图/权重可视化/错例等输出
└── requirements.txt
```

---

## 产出文件（报告可直接引用/截图）

- **最优模型权重**：`checkpoints/best_model.npz`
- **训练历史（json）**：`artifacts/history*.json`
- **训练曲线**：`artifacts/curves*.png`
- **测试集混淆矩阵**：`artifacts/confusion_matrix.png`
- **超参搜索结果**：`artifacts/search_results.json`
- **超参热力图**：`artifacts/search_heatmap_{relu,tanh,sigmoid}.png`
- **第一层权重可视化**：`artifacts/first_layer_weights.png`
- **错例图**：`artifacts/error_cases.png`

---

## 脚本说明（参数要点）

### `scripts/train.py` 训练

关键参数：
- `--hidden-dim1`：隐藏层宽度
- `--activation`：`relu` / `tanh` / `sigmoid`
- `--learning-rate`、`--lr-schedule`（`exp`/`cosine`）、`--min-lr`
- `--weight-decay`：L2 正则强度

输出：
- `checkpoints/best_model.npz`
- `artifacts/history*.json`、`artifacts/curves*.png`

### `scripts/search.py` 超参搜索

```powershell
python scripts/search.py --data-dir data --epochs 100 --batch-size 128 --seed 42 --results-path artifacts/search_results.json
```

超参网格在脚本内 `grid` 变量中设置；默认会额外导出 `artifacts/search_heatmap_*.png`（可用 `--no-plot` 关闭）。

### `scripts/visualize_weights.py` 第一层权重可视化

```powershell
python scripts/visualize_weights.py --model-path checkpoints/best_model.npz --hidden-dim1 256 --activation relu --num-filters 16
```

---

## 错例为何会分错（写报告时可参考）

Fashion-MNIST 是 \(28\times 28\) 的灰度图，很多类别差异本就很细；而单隐藏层 MLP 只把图像展平成 784 维向量，**不会显式利用空间结构（局部纹理/边缘连贯性/平移不变性）**，因此容易在“外观相近 + 信息缺失”的样本上出错。

典型混淆：
- **Sneaker / Sandal / Ankle boot**：鞋底轮廓在低分辨率下非常相近；如果鞋帮（boot 的高帮）部分被裁切/较暗，MLP 可能只看到“鞋底+前掌”的形状，从而偏向 Sandal/Sneaker。
- **Coat / Pullover**：两者都是上衣轮廓；当衣领/门襟等细节不明显、或纹理噪声较大时，整体外形更像 Pullover 的“实心块”，导致 Coat 被错分。

建议在报告里结合 `artifacts/error_cases.png` 的具体样本，描述“缺失了哪些关键像素线索”（例如鞋帮高度、开口/带子、衣领/开衩/门襟等）。

---

## 注意事项

- 训练、测试、可视化等脚本参数需与训练时一致（尤其是 `--hidden-dim1`、`--activation`），否则权重无法正确加载。
- 本实现不依赖 PyTorch / TensorFlow / JAX，核心训练逻辑均在 `src/hw1/` 中。
