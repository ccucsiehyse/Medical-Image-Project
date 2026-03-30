"""
eval_alzheimer.py — 重現訓練時終端機輸出（供補截圖／報告用）

目的：
    train_alzheimer.py 訓練結束後會將結果寫入某個「輸出目錄」（--output-dir），
    其中包含：
        - metrics.json      ：各 epoch 的 loss/acc、最佳驗證準確率、測試集指標
        - split_summary.json：資料切分筆數與各類別分佈
        - best_model.pt     ：最佳驗證準確率時儲存的權重（可選，用於還原 Device 等）

    本腳本**不重新訓練**，而是讀取上述 JSON（與可選的 checkpoint），
    依當初 train_alzheimer 的 print 格式把內容再印一次，方便你補上當初沒截到的畫面。

與 train_alzheimer.py 的對應關係：
    - 「Device: …」：若有 best_model.pt，從 checkpoint["args"]["device"] 經 resolve_device
      轉成與訓練時相同的 cuda/cpu 顯示；若無 .pt 則無法還原，會印提示句。
    - 「Classes: …」：來自 metrics.json 的 class_names。
    - 「Train / Val / Test: total=… | 類別=筆數」：來自 split_summary.json 的
      train_count、val_distribution 等，列印順序依類別索引 0..C-1（與訓練時 summarize_split 一致）。
    - 「Epoch i/N | …」：來自 metrics.json 的 history；N 優先取 checkpoint.args.epochs（與訓練設定的總 epoch 一致），
      否則退而求其次用 history 長度。若 history 內含 lr，會一併重播當下 learning rate。
    - 最後三行 Best val / Test loss / Test accuracy：直接來自 metrics.json。

執行範例（於 sampleCode 目錄下）：
    python eval_alzheimer.py --run-dir outputs/run_baseline

    若不指定 --run-dir，預設為 outputs/alzheimer_run。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    """讀取 UTF-8 JSON 檔並解析為 dict（metrics 或 split_summary）。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _load_checkpoint(path: Path) -> dict:
    """
    載入 PyTorch checkpoint（與 train_alzheimer 儲存格式相容）。

    - map_location="cpu"：僅供讀取 meta 與權重，避免在無 GPU 機器上出錯。
    - weights_only=False：checkpoint 內含 args、class_to_idx 等，非純權重張量；
      舊版 PyTorch 無 weights_only 參數時改用最簡單的 torch.load。
    """
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _print_split_line(split_name: str, total: int, distribution: dict, class_names: list[str]) -> None:
    """
    印出單一分割（Train / Val / Test）的一行摘要，格式對齊 train_alzheimer.summarize_split。

    參數：
        split_name   ：顯示名稱，例如 "Train"。
        total        ：該分割總樣本數（與 split_summary 的 *_count 一致）。
        distribution ：類別名稱 -> 該類在此分割的筆數（JSON 物件）。
        class_names  ：類別順序列表，須與訓練時相同；這裡用來決定印出順序（依索引 0,1,… 排序）。
    """
    idx_to_class = {i: n for i, n in enumerate(class_names)}
    parts = [f"{idx_to_class[idx]}={int(distribution[idx_to_class[idx]])}" for idx in sorted(idx_to_class)]
    print(f"{split_name}: total={total} | {', '.join(parts)}")


def replay_terminal(
    *,
    metrics_path: Path,
    split_summary_path: Path,
    checkpoint_path: Path | None = None,
) -> None:
    """
    核心流程：讀 JSON →（可選）載入 checkpoint → 依序 print，模擬訓練腳本終端輸出。

    參數：
        metrics_path        ：metrics.json 路徑。
        split_summary_path  ：split_summary.json 路徑。
        checkpoint_path     ：best_model.pt；若為 None 或檔案不存在，則跳過 Device 還原與權重載入，
                             其餘仍可依 JSON 重播 epoch 與最終指標。

    權重載入（僅驗證可載入，不做 forward）：
        與訓練相同建立 AlzheimerCNN(num_classes)，再 load_state_dict；
        若檔案損毀或類別數不符會在此拋錯，可及早發現 checkpoint 與 metrics 不一致。
    """
    metrics = _load_json(metrics_path)
    split_summary = _load_json(split_summary_path)

    class_names: list[str] = metrics["class_names"]
    history: list = metrics.get("history") or []

    ckpt: dict | None = None
    if checkpoint_path is not None and checkpoint_path.is_file():
        ckpt = _load_checkpoint(checkpoint_path)
        from train_alzheimer import AlzheimerCNN, resolve_device

        # 訓練時 serialize_args 會把 device 存成字串（如 "auto" / "cuda" / "cpu"）
        args_dict = ckpt.get("args") or {}
        dev = resolve_device(str(args_dict.get("device", "auto")))
        print(f"Device: {dev}")
        # 確認 state_dict 與目前 metrics 中的類別數一致（結構與 train 存檔相同）
        model = AlzheimerCNN(num_classes=len(class_names))
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # 無 .pt 時無法還原當時實際使用的 cuda/cpu，僅提示使用者
        print("Device: (未指定 checkpoint，無法重現；訓練時終端機會顯示 cuda 或 cpu)")

    print(f"Classes: {class_names}")

    _print_split_line("Train", split_summary["train_count"], split_summary["train_distribution"], class_names)
    _print_split_line("Val", split_summary["val_count"], split_summary["val_distribution"], class_names)
    _print_split_line("Test", split_summary["test_count"], split_summary["test_distribution"], class_names)

    # 畫面上的「Epoch i/N」：N 應與訓練設定的總 epoch 相同（例如 10），故優先讀 checkpoint.args.epochs
    epochs_total = len(history)
    if ckpt is not None:
        ckpt_epochs = (ckpt.get("args") or {}).get("epochs")
        if isinstance(ckpt_epochs, int) and ckpt_epochs > 0:
            epochs_total = ckpt_epochs

    for row in history:
        ep = int(row["epoch"])
        base_line = (
            f"Epoch {ep}/{epochs_total} | "
            f"train_loss={float(row['train_loss']):.4f} train_acc={float(row['train_accuracy']):.4f} | "
            f"val_loss={float(row['val_loss']):.4f} val_acc={float(row['val_accuracy']):.4f}"
        )
        # 相容舊版 metrics：若沒有 lr 欄位則沿用舊輸出格式
        if "lr" in row and row["lr"] is not None:
            base_line += f" | lr={float(row['lr']):.6g}"
        print(base_line)

    # 與 train_alzheimer 訓練結束後最後三行 print 一致（數值四位小數）
    print(f"Best val accuracy: {float(metrics['best_val_accuracy']):.4f}")
    print(f"Test loss: {float(metrics['test_loss']):.4f}")
    print(f"Test accuracy: {float(metrics['test_accuracy']):.4f}")


def parse_args() -> argparse.Namespace:
    """解析命令列；目前僅需指定訓練輸出目錄 --run-dir。"""
    parser = argparse.ArgumentParser(
        description="從某次訓練輸出目錄讀取 metrics.json / split_summary.json，重現終端輸出。"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/alzheimer_run"),
        help="訓練輸出目錄（內含 metrics.json、split_summary.json；可選 best_model.pt）。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dir = args.run_dir
    # 與 train 相同：預設檔名 best_model.pt；不存在則僅重播 JSON，不強制要求權重檔
    candidate = run_dir / "best_model.pt"
    ckpt_path = candidate if candidate.is_file() else None

    replay_terminal(
        metrics_path=run_dir / "metrics.json",
        split_summary_path=run_dir / "split_summary.json",
        checkpoint_path=ckpt_path,
    )
