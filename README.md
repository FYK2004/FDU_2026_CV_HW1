
# FDU 2026 春季 计算机视觉 HW1（Fashion-MNIST / NumPy）

## 项目结构

- **`src/hw1/`**：核心实现（数据读取、模型、训练与评估逻辑）
  - `data.py`：Fashion-MNIST 下载/加载与 DataLoader
  - `model.py`：两层 MLP（含激活函数与前向/反向）
  - `train.py`：训练循环（与脚本入口解耦的训练实现）
  - `evaluate.py`：Accuracy / 混淆矩阵等评估工具
- **`scripts/`**：可直接运行的实验入口脚本
  - `train.py`：训练并保存验证集最优模型
  - `search.py`：网格搜索超参并保存最优模型与结果
  - `test.py`：加载最优模型做测试集评估
  - `error_analysis.py`：导出错分样本图
  - `plot_search_results.py`：将搜索结果可视化为热力图
  - `visualize_weights.py`：可视化第一层权重
- **仓库根目录其他文件**
  - `.gitignore`：Git 忽略规则
  - `requirements.txt`：依赖列表
  - `CV_HW1.pdf`：作业报告

## 环境依赖

```powershell
pip install -r requirements.txt
```

首次运行脚本会自动下载 Fashion-MNIST 到 `data/`。

## 实验流程（推荐按顺序执行）

### 1) 构建并训练模型（在验证集上保存最优权重）

```powershell
python scripts/train.py --epochs 100 --batch-size 128 --learning-rate 0.05 --lr-schedule cosine --min-lr 1e-5 --weight-decay 1e-4 --hidden-dim1 256 --activation relu
```

参数说明（`scripts/train.py`）：
- `--data-dir`：数据目录（默认 `data`）
- `--epochs`：训练轮数（默认 `100`）
- `--batch-size`：mini-batch 大小（默认 `128`）
- `--learning-rate`：初始学习率（默认 `0.05`）
- `--lr-decay`：指数衰减系数（仅 `--lr-schedule exp` 时生效，默认 `0.95`）
- `--lr-schedule`：学习率策略（`exp`/`cosine`，默认 `cosine`）
- `--min-lr`：学习率下限（默认 `1e-5`）
- `--weight-decay`：L2 正则系数（默认 `1e-4`）
- `--hidden-dim1`：隐藏层宽度（默认 `256`）
- `--activation`：激活函数（`relu`/`sigmoid`/`tanh`，默认 `relu`）
- `--seed`：随机种子（默认 `42`）
- `--save-path`：保存验证集最优模型的路径（默认 `checkpoints/best_model.npz`）
- `--history-path`：训练过程记录 json 路径（默认 `artifacts/history.json`）
- `--curve-path`：训练曲线图片路径（默认 `artifacts/curves.png`）

关键输出（默认）：
- `checkpoints/best_model.npz`

### 2) 参数搜索（得到最优配置）

```powershell
python scripts/search.py --data-dir data --epochs 100 --batch-size 128 --seed 42 --results-path artifacts/search_results.json
```

参数说明（`scripts/search.py`）：
- `--data-dir`：数据目录（默认 `data`）
- `--epochs`：每组超参训练轮数（默认 `100`）
- `--batch-size`：mini-batch 大小（默认 `128`）
- `--seed`：随机种子（默认 `42`）
- `--results-path`：搜索结果保存路径（默认 `artifacts/search_results.json`）
- `--no-plot`：不生成热力图（默认不启用；加上该开关即启用）

关键输出（默认）：
- `artifacts/search_results.json`
- `checkpoints/best_model.npz`（搜索过程中会保存当前最优模型，结束后固定复制到该路径）

### 3) 用最优模型进行测试集评估（Accuracy + Confusion Matrix）

```powershell
python scripts/test.py --data-dir data --model-path checkpoints/best_model.npz --hidden-dim1 256 --activation relu
```

参数说明（`scripts/test.py`）：
- `--data-dir`：数据目录（默认 `data`）
- `--model-path`：模型权重路径（默认 `checkpoints/best_model.npz`）
- `--hidden-dim1`：隐藏层宽度（默认 `256`，需与模型一致）
- `--activation`：激活函数（`relu`/`sigmoid`/`tanh`，默认 `relu`，需与模型一致）

关键输出（默认）：
- 终端打印 Test Accuracy
- `artifacts/confusion_matrix.png`

注意：`--hidden-dim1` 与 `--activation` 需与“最优模型”的训练配置保持一致。

### 4) 错例分析（导出错分样本图）

```powershell
python scripts/error_analysis.py --data-dir data --model-path checkpoints/best_model.npz --hidden-dim1 256 --activation relu --num-errors 4 --out-path artifacts/error_cases.png
```

参数说明（`scripts/error_analysis.py`）：
- `--data-dir`：数据目录（默认 `data`）
- `--model-path`：模型权重路径（默认 `checkpoints/best_model.npz`）
- `--hidden-dim1`：隐藏层宽度（默认 `256`，需与模型一致）
- `--activation`：激活函数（`relu`/`sigmoid`/`tanh`，默认 `relu`，需与模型一致）
- `--num-errors`：导出错例数量（默认 `16`）
- `--out-path`：错例图保存路径（默认 `artifacts/error_cases.png`）

关键输出（默认）：
- `artifacts/error_cases.png`
