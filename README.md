
# FDU 2026 春季 计算机视觉 HW1

本仓库实现了**单隐藏层 MLP**（纯 NumPy）用于 Fashion-MNIST 分类，覆盖 PDF 所有任务：

- 数据加载与预处理
- 模型定义与反向传播
- 训练循环（SGD + 学习率衰减 + L2 正则 + 最优权重保存）
- 测试评估（Accuracy + Confusion Matrix）
- 超参数搜索
- 第一层权重可视化
- 错例分析

---

## 目录结构

```text
.
├── scripts/
│   ├── train.py
│   ├── test.py
│   ├── search.py
│   ├── visualize_weights.py
│   └── error_analysis.py
├── src/hw1/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
└── requirements.txt
```


## 环境准备

1. 安装依赖
	```powershell
	pip install -r requirements.txt
	```
2. 首次运行会自动下载 Fashion-MNIST 数据到 `data/`

---


## 按 PDF 任务逐步操作（强烈建议严格按顺序执行）


### 第 1 步（PDF Step 1）：训练单隐藏层 MLP 并保存最佳模型

```powershell
python scripts/train.py --epochs 100 --batch-size 128 --learning-rate 0.05 --lr-decay 0.95 --weight-decay 1e-4 --hidden-dim1 256 --activation relu
```
**可调超参数：**
- `--epochs`：训练轮数
- `--batch-size`：每批样本数
- `--learning-rate`：初始学习率
- `--lr-decay`：学习率衰减系数
- `--weight-decay`：L2正则强度
- `--hidden-dim1`：隐藏层神经元数
- `--activation`：激活函数（relu/sigmoid/tanh）
- `--seed`：随机种子
- `--save-path`：模型保存路径
- `--history-path`：训练历史保存路径
- `--curve-path`：训练曲线图片保存路径
输出：
- `checkpoints/best_model.npz`
- `artifacts/history.json`
- `artifacts/curves.png`（训练/验证 Loss 曲线 + 验证集 Accuracy 曲线）


### 第 2 步（PDF Step 2）：测试集评估（准确率 + 混淆矩阵）

```powershell
python scripts/test.py --model-path checkpoints/best_model.npz --hidden-dim1 256 --activation relu
```
**可调超参数：**
- `--data-dir`：数据集目录
- `--model-path`：模型权重路径
- `--hidden-dim1`：隐藏层神经元数（需与训练一致）
- `--activation`：激活函数（需与训练一致）
终端会打印：
- Test Accuracy
- Confusion Matrix
- 并保存混淆矩阵图片到 `artifacts/confusion_matrix.png`


### 第 3 步（PDF Step 3）：超参数查找（网格搜索）

```powershell
python scripts/search.py --epochs 100
```
建议在 scripts/search.py 的 grid 变量中设置如下合理参数范围：
- learning_rate: [0.01, 0.02, 0.05]
- hidden_dim1: [128, 256, 512]
- weight_decay: [1e-5, 1e-4, 5e-4]
- activation: ["relu", "tanh"]
**可调超参数：**
- `--data-dir`：数据集目录
- `--epochs`：每组参数训练轮数（默认100）
- `--batch-size`：每批样本数
- `--seed`：随机种子
- `--results-path`：搜索结果保存路径
（具体搜索哪些参数可在脚本内 grid 变量自定义）
输出：
- `artifacts/search_results.json`（包含每组参数性能和最优配置）
- 默认会自动生成每种激活函数一张热力图到 `artifacts/`：
  - `search_heatmap_relu.png`
  - `search_heatmap_tanh.png`
  - `search_heatmap_sigmoid.png`
- 若不想自动绘图可加 `--no-plot`
- 若想额外生成更多对比图，可运行 `python scripts/plot_search_results.py artifacts/search_results.json --all`


### 第 4 步（PDF Step 4）：第一层权重可视化

```powershell
python scripts/visualize_weights.py --model-path checkpoints/best_model.npz --hidden-dim1 256 --activation relu
```
**可调超参数：**
- `--model-path`：模型权重路径
- `--hidden-dim1`：隐藏层神经元数（需与训练一致）
- `--activation`：激活函数（需与训练一致）
- `--num-filters`：可视化的权重个数
- `--out-path`：图片保存路径
输出：
- `artifacts/first_layer_weights.png`


### 第 5 步（PDF Step 5）：错例分析图像导出

```powershell
python scripts/error_analysis.py --model-path checkpoints/best_model.npz --hidden-dim1 256 --activation relu --num-errors 16
```
**可调超参数：**
- `--data-dir`：数据集目录
- `--model-path`：模型权重路径
- `--hidden-dim1`：隐藏层神经元数（需与训练一致）
- `--activation`：激活函数（需与训练一致）
- `--num-errors`：导出的错例数量
- `--out-path`：图片保存路径
输出：
- `artifacts/error_cases.png`

---


## 作业报告内容与代码产出对应关系（建议直接引用/截图）

- **模型结构与训练配置**：引用 `src/hw1/model.py` 及训练命令（第1步）
- **训练曲线**：`artifacts/curves.png`（第1步输出）
- **混淆矩阵与测试准确率**：`scripts/test.py` 终端输出（第2步）
- **权重空间模式观察**：`artifacts/first_layer_weights.png`（第4步输出）
- **错例分析**：`artifacts/error_cases.png`（第5步输出）

---


## 注意事项 & 常见问题

- 所有脚本参数需与训练时保持一致（如 hidden-dim1、activation 等），否则模型无法正确加载。
- 训练、测试、可视化等均基于 NumPy 手工实现，未用 PyTorch / TensorFlow / JAX。
- 若遇到数据下载失败，请检查网络或手动下载 Fashion-MNIST 至 `data/` 目录。

---
