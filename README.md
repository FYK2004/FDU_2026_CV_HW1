
# FDU 2026 春季 计算机视觉 HW1（Fashion-MNIST / NumPy）

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
