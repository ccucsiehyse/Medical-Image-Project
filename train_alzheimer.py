"""
阿茲海默症（Alzheimer）腦部影像分類訓練腳本。

功能概要：
- **stratified 模式**：從單一 `--data-dir` 讀取類別子資料夾，依比例做**分層（stratified）**切分為 train/val/test。
- **folders 模式**：直接使用你事先切好的 `--split-root/train|val|test`（或分別指定三個路徑），不再於程式內切分。
- 使用自訂小型 CNN（AlzheimerCNN）與加權交叉熵，處理類別不平衡。
- 依驗證集準確率儲存最佳權重，最後在測試集上評估並寫出 metrics.json、split_summary.json。

執行方式：以命令列參數設定路徑與超參數（見 parse_args）。
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# 此腳本會掃描的影像副檔名（小寫比對）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    """解析命令列參數，並依 split-mode 檢查路徑與比例。"""
    parser = argparse.ArgumentParser(
        description="Train an Alzheimer image classifier from a single folder split."
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="stratified",
        choices=["stratified", "folders"],
        help=(
            "stratified：僅使用 --data-dir，在程式內依 val/test 比例切分；"
            "folders：使用既有 train/val/test 目錄（--split-root 或三個 --*-dir），不再切分。"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Alzheimer_s Dataset/train"),
        help="（stratified）含類別子資料夾的根目錄，程式會由此切出 train/val/test。folders 模式下可忽略。",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help="（folders）若設定，則使用 split-root/train、split-root/val、split-root/test。",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="（folders）訓練集根目錄（類別子資料夾下為影像）。與 --split-root 二擇一。",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=None,
        help="（folders）驗證集根目錄。",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="（folders）測試集根目錄。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/alzheimer_run"),
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    args = parser.parse_args()

    if args.split_mode == "stratified":
        # 驗證與測試比例須在 [0,1)，且兩者之和須小於 1（否則沒有訓練集）
        if not 0.0 <= args.val_ratio < 1.0:
            raise ValueError("--val-ratio must be in [0, 1).")
        if not 0.0 <= args.test_ratio < 1.0:
            raise ValueError("--test-ratio must be in [0, 1).")
        if args.val_ratio + args.test_ratio >= 1.0:
            raise ValueError("--val-ratio + --test-ratio must be < 1.")
    else:
        # folders：必須有 split-root，或三者路徑皆指定
        has_root = args.split_root is not None
        has_three = (
            args.train_dir is not None and args.val_dir is not None and args.test_dir is not None
        )
        if has_root and has_three:
            raise ValueError("請只指定 --split-root，或只指定 --train-dir / --val-dir / --test-dir，不要同時兩組都給。")
        if not has_root and not has_three:
            raise ValueError(
                "split-mode=folders 時請指定 --split-root，或同時指定 --train-dir、--val-dir、--test-dir。"
            )
    return args


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch（含 CUDA）隨機種子，使實驗可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """
    將字串 'auto' / 'cpu' / 'cuda' 轉成 torch.device。
    'auto' 時：有 GPU 則 cuda，否則 cpu。
    """
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_class_names(data_dir: Path) -> list[str]:
    """列出 data_dir 下一層所有子資料夾名稱，排序後作為類別名稱（標籤順序由此決定）。"""
    classes = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class folders found under: {data_dir}")
    return classes


def resolve_pre_split_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """folders 模式：由 --split-root 或三個 --*-dir 得到 train/val/test 路徑。"""
    if args.split_root is not None:
        root = args.split_root
        return root / "train", root / "val", root / "test"
    assert args.train_dir is not None and args.val_dir is not None and args.test_dir is not None
    return args.train_dir, args.val_dir, args.test_dir


def assert_matching_class_dirs(train_dir: Path, val_dir: Path, test_dir: Path) -> list[str]:
    """確保 train/val/test 下一層類別子資料夾名稱集合相同（順序以 train 為準）。"""
    class_names = list_class_names(train_dir)
    expected = set(class_names)
    for split_label, path in (("val", val_dir), ("test", test_dir)):
        found = {p.name for p in path.iterdir() if p.is_dir()}
        if found != expected:
            raise ValueError(
                f"{split_label} 的類別子資料夾須與 train 一致。"
                f" train={sorted(expected)}, {split_label}={sorted(found)}"
            )
    return class_names


def collect_samples(data_dir: Path, class_to_idx: dict[str, int]) -> list[tuple[Path, int]]:
    """
    遞迴掃描每個類別資料夾，收集 (影像路徑, 整數標籤) 清單。
    class_to_idx：類別名稱 -> 0..C-1 的對應表。
    """
    samples: list[tuple[Path, int]] = []
    for class_name, label in class_to_idx.items():
        class_dir = data_dir / class_name
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))
    if not samples:
        raise FileNotFoundError(f"No image files found under: {data_dir}")
    return samples


def compute_split_counts(total_count: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    """
    依總樣本數與比例計算驗證、測試各要取幾張（單一類別內使用）。

    - 先以四捨五入得到 val_count、test_count。
    - 若比例 > 0 但四捨五入後為 0，且該類至少 3 筆，則強制給 1 筆，避免完全沒有 val/test。
    - 若 val+test 加總 >= total（會吃掉全部樣本），則遞減直到能留下訓練集。
    """
    val_count = int(round(total_count * val_ratio))
    test_count = int(round(total_count * test_ratio))

    if val_ratio > 0 and val_count == 0 and total_count >= 3:
        val_count = 1
    if test_ratio > 0 and test_count == 0 and total_count >= 3:
        test_count = 1

    while val_count + test_count >= total_count and (val_count > 0 or test_count > 0):
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1

    return val_count, test_count


def stratified_split(
    samples: list[tuple[Path, int]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """
    分層切分：每個類別各自打亂後，依比例切出 val / test，剩餘為 train。
    最後再分別打亂三個集合的順序（不影響類別比例，只影響迭代順序）。
    """
    grouped: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for sample in samples:
        grouped[sample[1]].append(sample)

    rng = random.Random(seed)
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    test_samples: list[tuple[Path, int]] = []

    for label, class_samples in grouped.items():
        shuffled = class_samples[:]
        rng.shuffle(shuffled)
        val_count, test_count = compute_split_counts(len(shuffled), val_ratio, test_ratio)
        val_samples.extend(shuffled[:val_count])
        test_samples.extend(shuffled[val_count : val_count + test_count])
        train_samples.extend(shuffled[val_count + test_count :])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    if not train_samples:
        raise RuntimeError("Split failed because no training samples were generated.")
    return train_samples, val_samples, test_samples


class BasicImageTransform:
    """
    簡單影像前處理：RGB、resize、訓練時隨機水平翻轉，再正規化到 [0,1] 並轉成 CHW Tensor。
    """

    def __init__(self, image_size: int, train: bool) -> None:
        self.image_size = image_size
        self.train = train

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        # 訓練階段：50% 機率左右翻轉，做輕量資料增強
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        # PIL 為 HWC，PyTorch 慣例為 CHW
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        return image_tensor


class AlzheimerDataset(Dataset):
    """PyTorch Dataset：依 samples 清單載入影像並套用 transform。"""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform: BasicImageTransform | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            if self.transform is not None:
                image_tensor = self.transform(image)
            else:
                image_tensor = BasicImageTransform(image_size=224, train=False)(image)
        return image_tensor, label


class AlzheimerCNN(nn.Module):
    """
    小型卷積網路：多層 Conv-BN-ReLU-Pool 抽取特徵，AdaptiveAvgPool2d(1,1) 後接 MLP 分類。
    輸入假設為 batch×3×H×W（與 BasicImageTransform 輸出一致）。
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # inplace=True 代表直接覆蓋輸入資料以節省記憶體，減少中間暫存空間
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 將任意空間尺寸壓成 1×1，方便接全連接層
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 128),  # 全連接層，將前一層通道數 256 映射成 128 維的特徵向量，準備進入分類層
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def create_dataloader(
    samples: list[tuple[Path, int]],
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    train: bool,
) -> DataLoader:
    """建立 DataLoader；若有 CUDA 則 pin_memory 加速 CPU→GPU 傳輸。"""
    dataset = AlzheimerDataset(samples=samples, transform=BasicImageTransform(image_size, train=train))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_class_weights(samples: list[tuple[Path, int]], num_classes: int) -> torch.Tensor:
    """
    依訓練集各類樣本數計算 CrossEntropyLoss 的類別權重。
    公式：weight[c] = N / (C * n_c)，樣本少的類別權重較大（常見的 inverse-frequency 變形）。
    """
    counts = Counter(label for _, label in samples)
    total = sum(counts.values())
    weights = []
    for class_index in range(num_classes):
        class_count = counts[class_index]
        weights.append(total / (num_classes * class_count))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """
    執行一個 epoch：optimizer 為 None 時為評估模式（不反向傳播、不更新權重）。
    回傳整個 dataloader 的加權平均 loss 與整體 accuracy。
    max_batches 可用於快速 smoke test。
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_index, (images, labels) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            predictions = logits.argmax(dim=1)
            # 以樣本數加權累加 loss，使回傳值為「全資料平均 loss」
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("No samples were processed. Check split ratios or debug batch limits.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def summarize_split(split_name: str, samples: list[tuple[Path, int]], idx_to_class: dict[int, str]) -> None:
    """印出某一分割的總數與各類別筆數。"""
    counts = Counter(label for _, label in samples)
    details = ", ".join(f"{idx_to_class[idx]}={counts[idx]}" for idx in sorted(idx_to_class))
    print(f"{split_name}: total={len(samples)} | {details}")


def save_json(path: Path, payload: dict) -> None:
    """將 dict 以縮排 JSON 寫入檔案（ASCII，避免部分環境編碼問題）。"""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def serialize_args(args: argparse.Namespace) -> dict:
    """將 argparse 結果轉成可 JSON 序列化的 dict（Path 轉成字串）。"""
    serialized = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def main() -> None:
    """主流程：讀資料、（可選）切分、建模型、訓練、存最佳權重、測試集評估、寫出結果 JSON。"""
    args = parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.split_mode == "stratified":
        if not args.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")
        # 類別名稱排序後對應到 0..C-1，之後模型輸出維度與權重順序皆依此固定
        class_names = list_class_names(args.data_dir)
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
        all_samples = collect_samples(args.data_dir, class_to_idx)
        train_samples, val_samples, test_samples = stratified_split(
            samples=all_samples,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        train_path, val_path, test_path = resolve_pre_split_dirs(args)
        for label, p in (("train", train_path), ("val", val_path), ("test", test_path)):
            if not p.exists():
                raise FileNotFoundError(f"folders 模式：{label} 目錄不存在: {p}")
        class_names = assert_matching_class_dirs(train_path, val_path, test_path)
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
        train_samples = collect_samples(train_path, class_to_idx)
        val_samples = collect_samples(val_path, class_to_idx)
        test_samples = collect_samples(test_path, class_to_idx)
        if not train_samples:
            raise RuntimeError("folders 模式：訓練集為空。")
        if not val_samples:
            raise RuntimeError("folders 模式：驗證集為空。")
        if not test_samples:
            raise RuntimeError("folders 模式：測試集為空。")

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    summarize_split("Train", train_samples, idx_to_class)
    summarize_split("Val", val_samples, idx_to_class)
    summarize_split("Test", test_samples, idx_to_class)

    train_loader = create_dataloader(
        samples=train_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        train=True,
    )
    val_loader = create_dataloader(
        samples=val_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        train=False,
    )
    test_loader = create_dataloader(
        samples=test_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        train=False,
    )

    model = AlzheimerCNN(num_classes=len(class_names)).to(device)
    class_weights = compute_class_weights(train_samples, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    best_checkpoint_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            max_batches=args.max_eval_batches,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

        # 驗證準確率創新高時儲存 checkpoint（含權重與類別對照，方便推論還原標籤）
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "args": serialize_args(args),
                    "best_val_accuracy": best_val_accuracy,
                },
                best_checkpoint_path,
            )

    # 載入驗證集最佳模型，在測試集上只做評估（不再更新權重）
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        max_batches=args.max_eval_batches,
    )

    print(f"Best val accuracy: {best_val_accuracy:.4f}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

    save_json(
        args.output_dir / "metrics.json",
        {
            "class_names": class_names,
            "best_val_accuracy": best_val_accuracy,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "history": history,
        },
    )
    split_summary_payload: dict = {
        "split_mode": args.split_mode,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "train_distribution": {
            idx_to_class[idx]: Counter(label for _, label in train_samples)[idx]
            for idx in sorted(idx_to_class)
        },
        "val_distribution": {
            idx_to_class[idx]: Counter(label for _, label in val_samples)[idx]
            for idx in sorted(idx_to_class)
        },
        "test_distribution": {
            idx_to_class[idx]: Counter(label for _, label in test_samples)[idx]
            for idx in sorted(idx_to_class)
        },
    }
    if args.split_mode == "stratified":
        split_summary_payload["data_dir"] = str(args.data_dir)
        split_summary_payload["val_ratio"] = args.val_ratio
        split_summary_payload["test_ratio"] = args.test_ratio
    else:
        tr, va, te = resolve_pre_split_dirs(args)
        split_summary_payload["train_dir"] = str(tr)
        split_summary_payload["val_dir"] = str(va)
        split_summary_payload["test_dir"] = str(te)
    save_json(args.output_dir / "split_summary.json", split_summary_payload)


if __name__ == "__main__":
    main()
