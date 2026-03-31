"""
使用方式：
    python plot_metrics.py --name <name>

範例：
    python plot_metrics.py --name run_2extraLayer2

程式會自動讀取：  ./outputs/<name>/metrics.json
並將圖片輸出至：  ./outputs/<name>/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


# ── 解析命令列參數 ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="繪製訓練指標圖（loss / accuracy）",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument(
    "--name",
    required=True,
    type=str,
    help="輸出資料夾名稱，程式將讀取 ./outputs/<name>/metrics.json",
)
args = parser.parse_args()

# ── 路徑設定 ───────────────────────────────────────────────────────────────────
base_dir  = Path.cwd() / "outputs" / args.name
json_path = base_dir / "metrics.json"

if not json_path.exists():
    print(f"[錯誤] 找不到檔案：{json_path}", file=sys.stderr)
    sys.exit(1)

base_dir.mkdir(parents=True, exist_ok=True)

# ── 載入資料 ───────────────────────────────────────────────────────────────────
with open(json_path, "r") as f:
    data = json.load(f)

history       = data["history"]
test_accuracy = data["test_accuracy"]

epochs     = [h["epoch"] for h in history]
train_loss = [h["train_loss"] for h in history]
val_loss   = [h["val_loss"] for h in history]
train_acc  = [h["train_accuracy"] for h in history]
val_acc    = [h["val_accuracy"] for h in history]
test_acc   = [test_accuracy] * len(epochs)

# ── 共用樣式 ───────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4e",
    "axes.labelcolor":  "#c8ccd8",
    "axes.titlecolor":  "#e8ecf5",
    "xtick.color":      "#7a7e8e",
    "ytick.color":      "#7a7e8e",
    "grid.color":       "#2a2d3e",
    "grid.linewidth":   0.8,
    "text.color":       "#c8ccd8",
    "font.family":      "DejaVu Sans",
}

COLORS = {
    "train_loss": "#ff6b6b",
    "val_loss":   "#4ecdc4",
    "train_acc":  "#ffe66d",
    "val_acc":    "#a29bfe",
    "test_acc":   "#fd79a8",
}

MARKER = dict(markersize=5, markeredgewidth=1.2)


def styled_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=6)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(framealpha=0.15, edgecolor="#3a3d4e", fontsize=9)
    ax.set_xticks(epochs)


# ── 四張獨立圖 ─────────────────────────────────────────────────────────────────
single_plots = [
    ("Train Loss",      epochs, train_loss, "Loss",     COLORS["train_loss"], "train_loss.png"),
    ("Validation Loss", epochs, val_loss,   "Loss",     COLORS["val_loss"],   "val_loss.png"),
    ("Train Accuracy",  epochs, train_acc,  "Accuracy", COLORS["train_acc"],  "train_acc.png"),
    ("Val Accuracy",    epochs, val_acc,    "Accuracy", COLORS["val_acc"],    "val_acc.png"),
]

with plt.rc_context(STYLE):
    for title, x, y, ylabel, color, fname in single_plots:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, y, color=color, linewidth=2, marker="o", label=title, **MARKER)
        styled_ax(ax, title, "Epoch", ylabel)
        fig.tight_layout()
        out = base_dir / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"已儲存：{out}")

# ── 第五張：三條 Accuracy 對比圖 ───────────────────────────────────────────────
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, train_acc, color=COLORS["train_acc"], linewidth=2,
            marker="o", label="Train Accuracy", **MARKER)
    ax.plot(epochs, val_acc,   color=COLORS["val_acc"],   linewidth=2,
            marker="s", label="Val Accuracy", **MARKER)
    ax.plot(epochs, test_acc,  color=COLORS["test_acc"],  linewidth=2,
            linestyle="--", label=f"Test Accuracy ({test_accuracy:.4f})")

    best_epoch = epochs[val_acc.index(max(val_acc))]
    best_val   = max(val_acc)
    ax.annotate(
        f"Best Val\n{best_val:.4f}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 0.6, best_val - 0.08),
        fontsize=8, color=COLORS["val_acc"],
        arrowprops=dict(arrowstyle="->", color=COLORS["val_acc"], lw=1),
    )

    styled_ax(ax, "Accuracy Comparison (Train / Val / Test)", "Epoch", "Accuracy")
    ax.set_ylim(0, 1.08)

    fig.tight_layout()
    out = base_dir / "accuracy_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已儲存：{out}")

print("全部完成！")
