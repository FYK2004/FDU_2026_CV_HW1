import argparse
import json
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot grid search results")
    parser.add_argument("results_path", type=str, help="Path to artifacts/search_results.json")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate extra plots (topk/scatter/summary). Default: only per-activation heatmaps.",
    )
    return parser.parse_args()

def _sorted_unique(values):
    return sorted(set(values))


def _config_key(cfg: Dict) -> Tuple:
    return (
        cfg.get("activation"),
        cfg.get("hidden_dim1"),
        cfg.get("learning_rate"),
        cfg.get("weight_decay"),
    )


def _format_cfg(cfg: Dict) -> str:
    return (
        f"act={cfg.get('activation')}, h={cfg.get('hidden_dim1')}, "
        f"lr={cfg.get('learning_rate')}, wd={cfg.get('weight_decay')}"
    )


def plot_search_results(results_path: str, *, all_plots: bool = False) -> None:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]

    activations = _sorted_unique(item["config"]["activation"] for item in results)
    hidden_dims = _sorted_unique(item["config"]["hidden_dim1"] for item in results)
    learning_rates = _sorted_unique(item["config"]["learning_rate"] for item in results)
    weight_decays = _sorted_unique(item["config"]["weight_decay"] for item in results)

    acc_map: Dict[Tuple, float] = {}
    for item in results:
        cfg = item["config"]
        acc_map[_config_key(cfg)] = float(item["best_val_acc"])

    out_dir = Path(results_path).parent

    vmin = min(acc_map.values()) if acc_map else 0.0
    vmax = max(acc_map.values()) if acc_map else 1.0

    for act in activations:
        fig_a, axes_a = plt.subplots(
            nrows=1,
            ncols=len(hidden_dims),
            figsize=(4.6 * len(hidden_dims) + 1.4, 4.2),
            squeeze=False,
            constrained_layout=True,
        )
        im_a = None
        for c, h in enumerate(hidden_dims):
            ax = axes_a[0][c]
            grid: List[List[float]] = []
            for lr in learning_rates:
                row = []
                for wd in weight_decays:
                    row.append(acc_map.get((act, h, lr, wd), float("nan")))
                grid.append(row)

            im_a = ax.imshow(grid, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(
                list(range(len(weight_decays))),
                [str(wd) for wd in weight_decays],
                rotation=30,
                ha="right",
            )
            ax.set_yticks(list(range(len(learning_rates))), [str(lr) for lr in learning_rates])
            ax.set_xlabel("weight_decay")
            ax.set_ylabel("learning_rate")
            ax.set_title(f"hidden_dim1={h}")

            for i in range(len(learning_rates)):
                for j in range(len(weight_decays)):
                    val = grid[i][j]
                    # 数值为“非数值”时，会出现“自身不等于自身”
                    if val == val:
                        ax.text(
                            j,
                            i,
                            f"{val:.3f}",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="black",
                        )

        if im_a is not None:
            cbar_a = fig_a.colorbar(im_a, ax=axes_a.ravel().tolist(), shrink=0.95, pad=0.02)
            cbar_a.set_label("best_val_acc")
        fig_a.suptitle(f"Grid Search Heatmaps — activation={act} (lr × weight_decay)", y=1.03)

        out_path_a = out_dir / f"search_heatmap_{act}.png"
        fig_a.savefig(out_path_a, dpi=200, bbox_inches="tight")
        plt.close(fig_a)

    if not all_plots:
        print(f"Saved per-activation heatmaps to: {out_dir}")
        return

    ranked = sorted(results, key=lambda x: float(x["best_val_acc"]), reverse=True)
    top_k = min(15, len(ranked))
    top_items = ranked[:top_k]
    labels = [_format_cfg(it["config"]) for it in top_items]
    accs = [float(it["best_val_acc"]) for it in top_items]

    fig2, ax2 = plt.subplots(figsize=(10.5, 0.55 * top_k + 2.0))
    y = list(range(top_k))[::-1]
    ax2.barh(y, accs[::-1])
    ax2.set_yticks(y, labels[::-1])
    ax2.set_xlabel("best_val_acc")
    ax2.set_title(f"Top-{top_k} Configurations by Validation Accuracy")
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    fig2.tight_layout()
    out_path2 = out_dir / "search_topk.png"
    fig2.savefig(out_path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # 一个更直观的总览图：学习率 × 权重衰减 的散点（颜色=验证准确率，大小=隐藏层宽度，形状=激活函数）
    marker_map = {
        "relu": "o",
        "tanh": "s",
        "sigmoid": "^",
    }
    # 遇到未知激活函数时用备用点形
    default_markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]

    fig25, ax25 = plt.subplots(figsize=(9.5, 6.2))

    # 点大小做个缩放，避免看起来差不多
    hmin = min(hidden_dims) if hidden_dims else 1
    hmax = max(hidden_dims) if hidden_dims else 1
    def _size(h):
        if hmax == hmin:
            return 220
        return 120 + 320 * (h - hmin) / (hmax - hmin)

    vmin2 = min(acc_map.values()) if acc_map else 0.0
    vmax2 = max(acc_map.values()) if acc_map else 1.0

    # 每种激活函数单独画一层，图例更清楚
    used_markers = {}
    for idx, act in enumerate(activations):
        mk = marker_map.get(act)
        if mk is None:
            mk = default_markers[idx % len(default_markers)]
        used_markers[act] = mk

        xs, ys, cs, ss = [], [], [], []
        for item in results:
            cfg = item["config"]
            if cfg["activation"] != act:
                continue
            xs.append(float(cfg["learning_rate"]))
            ys.append(float(cfg["weight_decay"]))
            cs.append(float(item["best_val_acc"]))
            ss.append(_size(int(cfg["hidden_dim1"])))

        sc = ax25.scatter(
            xs,
            ys,
            c=cs,
            s=ss,
            marker=mk,
            cmap="viridis",
            vmin=vmin2,
            vmax=vmax2,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.9,
            label=f"activation={act}",
        )

    ax25.set_xscale("log")
    ax25.set_yscale("log")
    ax25.set_xlabel("learning_rate (log)")
    ax25.set_ylabel("weight_decay (log)")
    ax25.set_title("Grid Search Overview (color=val_acc, size=hidden_dim1, marker=activation)")
    ax25.grid(True, which="both", linestyle="--", alpha=0.25)
    cbar25 = fig25.colorbar(sc, ax=ax25, pad=0.02)
    cbar25.set_label("best_val_acc")
    ax25.legend(loc="best", frameon=True)

    # 隐藏层宽度的大小图例（最多放 3 个参照）
    ref_dims = hidden_dims if len(hidden_dims) <= 3 else [hidden_dims[0], hidden_dims[len(hidden_dims)//2], hidden_dims[-1]]
    handles = [
        ax25.scatter([], [], s=_size(h), c="none", edgecolors="black", marker="o", linewidths=0.6)
        for h in ref_dims
    ]
    labels = [f"hidden_dim1={h}" for h in ref_dims]
    size_legend = ax25.legend(handles, labels, title="Point size", loc="lower left", frameon=True)
    ax25.add_artist(size_legend)

    fig25.tight_layout()
    out_path25 = out_dir / "search_scatter.png"
    fig25.savefig(out_path25, dpi=200, bbox_inches="tight")
    plt.close(fig25)

    # 每种激活函数单独一张散点图
    for idx, act in enumerate(activations):
        mk = used_markers.get(act, default_markers[idx % len(default_markers)])

        xs, ys, cs, ss = [], [], [], []
        for item in results:
            cfg = item["config"]
            if cfg["activation"] != act:
                continue
            xs.append(float(cfg["learning_rate"]))
            ys.append(float(cfg["weight_decay"]))
            cs.append(float(item["best_val_acc"]))
            ss.append(_size(int(cfg["hidden_dim1"])))

        fig26, ax26 = plt.subplots(figsize=(8.6, 5.8))
        sc26 = ax26.scatter(
            xs,
            ys,
            c=cs,
            s=ss,
            marker=mk,
            cmap="viridis",
            vmin=vmin2,
            vmax=vmax2,
            edgecolors="black",
            linewidths=0.7,
            alpha=0.95,
        )
        ax26.set_xscale("log")
        ax26.set_yscale("log")
        ax26.set_xlabel("learning_rate (log)")
        ax26.set_ylabel("weight_decay (log)")
        ax26.set_title(f"Grid Search Overview — activation={act} (color=val_acc, size=hidden_dim1)")
        ax26.grid(True, which="both", linestyle="--", alpha=0.25)
        cbar26 = fig26.colorbar(sc26, ax=ax26, pad=0.02)
        cbar26.set_label("best_val_acc")

        # 隐藏层宽度的大小图例
        handles = [
            ax26.scatter([], [], s=_size(h), c="none", edgecolors="black", marker="o", linewidths=0.7)
            for h in ref_dims
        ]
        labels = [f"hidden_dim1={h}" for h in ref_dims]
        ax26.legend(handles, labels, title="Point size", loc="lower left", frameon=True)

        fig26.tight_layout()
        out_path26 = out_dir / f"search_scatter_{act}.png"
        fig26.savefig(out_path26, dpi=200, bbox_inches="tight")
        plt.close(fig26)

    # 隐藏层宽度的简单对比：对每种激活函数，取该宽度下的最优验证准确率
    fig3, ax3 = plt.subplots()
    for act in activations:
        accs_hd = []
        for h in hidden_dims:
            acc = max(
                (
                    float(item["best_val_acc"])
                    for item in results
                    if item["config"]["hidden_dim1"] == h and item["config"]["activation"] == act
                ),
                default=float("nan"),
            )
            accs_hd.append(acc)
        ax3.plot(hidden_dims, accs_hd, marker="o", label=f"activation={act}")
    ax3.set_xlabel("Hidden Dimension")
    ax3.set_ylabel("Best Validation Accuracy (max over lr, wd)")
    ax3.set_title("Hidden Dim vs Best Val Acc (per activation)")
    ax3.legend()
    fig3.tight_layout()
    out_path3 = out_dir / "search_performance.png"
    fig3.savefig(out_path3, dpi=200, bbox_inches="tight")
    plt.close(fig3)

    print(f"Saved top-k ranking plot to {out_path2}")
    print(f"Saved scatter overview plot to {out_path25}")
    print(f"Saved hidden-dim summary plot to {out_path3}")

if __name__ == "__main__":
    args = parse_args()
    plot_search_results(args.results_path, all_plots=args.all)
