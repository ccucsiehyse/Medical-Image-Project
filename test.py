"""
test.py — 評估已訓練好的 best_model.pt（不重新訓練）。

功能：
- 載入指定的 checkpoint（best_model.pt）
- 使用指定的測試集資料夾（類別子資料夾結構）進行推論
- 輸出 Test loss / Test accuracy
- 額外輸出：confusion matrix、每類別 precision/recall/F1、macro/weighted F1

執行範例（建議在 sampleCode 目錄下）：
    python test.py --checkpoint outputs/run_xxx/best_model.pt --test-dir "D:/path/to/test"
或使用 split-root：
    python test.py --checkpoint outputs/run_xxx/best_model.pt --split-root "D:/path/to/split_root"
    （會自動使用 split-root/test）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

# 直接重用訓練腳本的模型與資料處理，確保前處理/架構一致
from train_alzheimer import (  # type: ignore
    AlzheimerCNN,
    BasicImageTransform,
    AlzheimerDataset,
    collect_samples,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate best_model.pt on a test dataset folder.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to best_model.pt (checkpoint).",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Test dataset root. Must contain class subfolders.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help="If provided, use split-root/test as the test-dir.",
    )
    parser.add_argument("--image-size", type=int, default=None, help="Override image size (default: use checkpoint args or 224).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "auto", "cpu", "cuda"],
        help="Override device (default: use checkpoint args or auto).",
    )
    parser.add_argument(
        "--cm-output",
        type=Path,
        default=None,
        help="If set, save confusion matrix figure to this path (e.g. cm.png).",
    )
    parser.add_argument(
        "--show-cm",
        action="store_true",
        help="Show confusion matrix figure in an interactive window.",
    )
    parser.add_argument(
        "--cm-normalize",
        action="store_true",
        help="Normalize confusion matrix by row (true label) when plotting.",
    )
    return parser.parse_args()


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    num_classes = len(class_names)
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)  # rows=true, cols=pred

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        bs = labels.size(0)

        total_loss += float(loss.item()) * bs
        total_correct += int((preds == labels).sum().item())
        total_samples += int(bs)

        for t, p in zip(labels.view(-1), preds.view(-1), strict=False):
            conf[int(t), int(p)] += 1

    if total_samples == 0:
        raise RuntimeError("No samples were evaluated. Check your test-dir path and folder structure.")

    # per-class metrics from confusion matrix
    tp = conf.diag().to(torch.float64)
    support = conf.sum(dim=1).to(torch.float64)
    pred_count = conf.sum(dim=0).to(torch.float64)
    fp = pred_count - tp
    fn = support - tp

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    macro_f1 = float(f1.mean().item())
    weighted_f1 = float(((f1 * support) / (support.sum() + eps)).sum().item())

    return {
        "test_loss": total_loss / total_samples,
        "test_accuracy": total_correct / total_samples,
        "confusion_matrix": conf.tolist(),
        "per_class": [
            {
                "class": class_names[i],
                "support": int(support[i].item()),
                "precision": float(precision[i].item()),
                "recall": float(recall[i].item()),
                "f1": float(f1[i].item()),
            }
            for i in range(num_classes)
        ],
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def plot_confusion_matrix(
    *,
    conf_mat: list[list[int]],
    class_names: list[str],
    output_path: Path | None,
    show: bool,
    normalize: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting confusion matrix. "
            "Install it by running: pip install matplotlib"
        ) from exc

    mat = np.asarray(conf_mat, dtype=np.float64)
    if normalize:
        row_sum = mat.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        mat = mat / row_sum

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix" + (" (Normalized)" if normalize else ""),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = mat.max() / 2.0 if mat.size else 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            text = f"{mat[i, j]:.2f}" if normalize else f"{int(mat[i, j])}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if mat[i, j] > threshold else "black",
            )

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Confusion matrix figure saved to: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()

    ckpt = _load_checkpoint(args.checkpoint)
    ckpt_args = ckpt.get("args") or {}
    class_to_idx = ckpt.get("class_to_idx") or {}

    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise RuntimeError("Checkpoint does not contain 'class_to_idx'. Please use the checkpoint saved by train_alzheimer.py.")

    # 以 checkpoint 的 label 對應順序為準（避免 test-dir 類別排序不同而對錯 label）
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: int(kv[1]))]

    if args.split_root is not None:
        test_dir = args.split_root / "test"
    else:
        test_dir = args.test_dir

    if test_dir is None:
        raise ValueError("Please provide --test-dir or --split-root.")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory does not exist: {test_dir}")

    # 驗證測試集類別資料夾是否齊全
    found = {p.name for p in test_dir.iterdir() if p.is_dir()}
    expected = set(class_names)
    if found != expected:
        raise ValueError(
            "Test directory class folders do not match checkpoint classes. "
            f"expected={sorted(expected)}, found={sorted(found)}"
        )

    # device / image_size：預設沿用 checkpoint 設定（若 checkpoint 無則 fallback）
    device_name = args.device if args.device is not None else str(ckpt_args.get("device", "auto"))
    device = resolve_device(device_name)

    image_size = args.image_size if args.image_size is not None else int(ckpt_args.get("image_size", 224))

    # 建立 samples（使用 checkpoint 的 class_to_idx）
    samples = collect_samples(test_dir, class_to_idx)
    dataset = AlzheimerDataset(samples=samples, transform=BasicImageTransform(image_size=image_size, train=False))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = AlzheimerCNN(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Test dir: {test_dir}")
    print(f"Test images: {len(samples)}")
    print(f"Image size: {image_size}")

    metrics = evaluate(model=model, dataloader=dataloader, device=device, class_names=class_names)

    print(f"Test loss: {metrics['test_loss']:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("")
    print("Per-class metrics:")
    for row in metrics["per_class"]:
        print(
            f"- {row['class']}: support={row['support']} "
            f"precision={row['precision']:.4f} recall={row['recall']:.4f} f1={row['f1']:.4f}"
        )
    print("")
    print("Confusion matrix (rows=true, cols=pred):")
    for r in metrics["confusion_matrix"]:
        print(" ".join(str(x) for x in r))

    if args.cm_output is not None or args.show_cm:
        plot_confusion_matrix(
            conf_mat=metrics["confusion_matrix"],
            class_names=class_names,
            output_path=args.cm_output,
            show=args.show_cm,
            normalize=args.cm_normalize,
        )


if __name__ == "__main__":
    main()

