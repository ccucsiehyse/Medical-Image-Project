"""
阿茲海默症（Alzheimer）腦部影像分類訓練腳本。

功能概要：
- 支援載入 ResNet18。
- stratified 模式與 folders 模式資料讀取。

# 執行範例：
python train_resnet18.py \
  --split-root "split" \
  --split-mode "folders" \
  --lr 0.001 \
  --batch-size 32 \
  --epochs 20 \
  --device cuda \
  --output-dir "outputs/resnet18" \
  --lr-scheduler plateau

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
import torchvision.models as models  # 新增引入
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an Alzheimer image classifier from a single folder split."
    )
    # 遷移學習新增參數
    parser.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="預訓練權重的路徑 (.pth 或 .pt)。若不提供則從頭訓練。",
    )
    parser.add_argument(
        "--freeze-base",
        action="store_true",
        help="是否凍結 ResNet18 的基礎卷積層，僅訓練最後的全連接層。",
    )
    
    parser.add_argument("--split-mode", type=str, default="stratified", choices=["stratified", "folders"])
    parser.add_argument("--data-dir", type=Path, default=Path("Alzheimer_s Dataset/train"))
    parser.add_argument("--split-root", type=Path, default=Path("split_aug"))
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/alzheimer_run"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", type=str, default="none", choices=["none", "step", "plateau"])
    parser.add_argument("--lr-step-size", type=int, default=7)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    
    args = parser.parse_args()

    if args.split_mode == "stratified":
        if not 0.0 <= args.val_ratio < 1.0:
            raise ValueError("--val-ratio must be in [0, 1).")
        if not 0.0 <= args.test_ratio < 1.0:
            raise ValueError("--test-ratio must be in [0, 1).")
        if args.val_ratio + args.test_ratio >= 1.0:
            raise ValueError("--val-ratio + --test-ratio must be < 1.")
    else:
        has_root = args.split_root is not None
        has_three = args.train_dir is not None and args.val_dir is not None and args.test_dir is not None
        if has_root and has_three:
            raise ValueError("請只指定 --split-root，或只指定 --train-dir / --val-dir / --test-dir，不要同時兩組都給。")
        if not has_root and not has_three:
            raise ValueError("split-mode=folders 時請指定 --split-root，或同時指定 --train-dir、--val-dir、--test-dir。")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_class_names(data_dir: Path) -> list[str]:
    classes = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class folders found under: {data_dir}")
    return classes


def resolve_pre_split_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.split_root is not None:
        root = args.split_root
        return root / "train", root / "val", root / "test"
    assert args.train_dir is not None and args.val_dir is not None and args.test_dir is not None
    return args.train_dir, args.val_dir, args.test_dir


def assert_matching_class_dirs(train_dir: Path, val_dir: Path, test_dir: Path) -> list[str]:
    class_names = list_class_names(train_dir)
    expected = set(class_names)
    for split_label, path in (("val", val_dir), ("test", test_dir)):
        found = {p.name for p in path.iterdir() if p.is_dir()}
        if found != expected:
            raise ValueError(f"{split_label} 的類別子資料夾須與 train 一致。")
    return class_names


def collect_samples(data_dir: Path, class_to_idx: dict[str, int]) -> list[tuple[Path, int]]:
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
    return train_samples, val_samples, test_samples


class BasicImageTransform:
    def __init__(self, image_size: int, train: bool) -> None:
        self.image_size = image_size
        self.train = train

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        # ResNet18 標準化參數 (ImageNet/RadImageNet 通用常態化設定)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std
        
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        return image_tensor


class AlzheimerDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform: BasicImageTransform | None = None) -> None:
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


def create_dataloader(
    samples: list[tuple[Path, int]], image_size: int, batch_size: int, num_workers: int, shuffle: bool, train: bool
) -> DataLoader:
    dataset = AlzheimerDataset(samples=samples, transform=BasicImageTransform(image_size, train=train))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )


def compute_class_weights(samples: list[tuple[Path, int]], num_classes: int) -> torch.Tensor:
    counts = Counter(label for _, label in samples)
    total = sum(counts.values())
    weights = []
    for class_index in range(num_classes):
        class_count = counts[class_index]
        weights.append(total / (num_classes * class_count))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None, max_batches: int | None = None
) -> dict[str, float]:
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
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples}


def summarize_split(split_name: str, samples: list[tuple[Path, int]], idx_to_class: dict[int, str]) -> None:
    counts = Counter(label for _, label in samples)
    details = ", ".join(f"{idx_to_class[idx]}={counts[idx]}" for idx in sorted(idx_to_class))
    print(f"{split_name}: total={len(samples)} | {details}")


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def serialize_args(args: argparse.Namespace) -> dict:
    serialized = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.split_mode == "stratified":
        class_names = list_class_names(args.data_dir)
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
        all_samples = collect_samples(args.data_dir, class_to_idx)
        train_samples, val_samples, test_samples = stratified_split(
            samples=all_samples, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
        )
    else:
        train_path, val_path, test_path = resolve_pre_split_dirs(args)
        class_names = assert_matching_class_dirs(train_path, val_path, test_path)
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
        train_samples = collect_samples(train_path, class_to_idx)
        val_samples = collect_samples(val_path, class_to_idx)
        test_samples = collect_samples(test_path, class_to_idx)

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    summarize_split("Train", train_samples, idx_to_class)
    summarize_split("Val", val_samples, idx_to_class)
    summarize_split("Test", test_samples, idx_to_class)

    train_loader = create_dataloader(train_samples, args.image_size, args.batch_size, args.num_workers, True, True)
    val_loader = create_dataloader(val_samples, args.image_size, args.batch_size, args.num_workers, False, False)
    test_loader = create_dataloader(test_samples, args.image_size, args.batch_size, args.num_workers, False, False)

    # 建立 ResNet18 並處理預訓練權重
    model = models.resnet18(weights=None)
    
    if args.weights_path:
        print(f"Loading pretrained weights from {args.weights_path}")
        state_dict = torch.load(args.weights_path, map_location="cpu")
        # 若權重來自 DataParallel，可能帶有 'module.' 前綴，予以移除
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # 排除全連接層權重，避免與目前類別數不符
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        model.load_state_dict(state_dict, strict=False)

    # 若設定凍結基礎層，則關閉梯度計算
    if args.freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # 替換最後的全連接層以符合當前分類任務
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    class_weights = compute_class_weights(train_samples, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 若有凍結層，只將需要梯度的參數傳入優化器
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params_to_update, lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = None
    if args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.lr_min)

    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    best_checkpoint_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, args.max_train_batches)
        val_metrics = run_epoch(model, val_loader, criterion, device, None, args.max_eval_batches)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6g}"
        )

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

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(model, test_loader, criterion, device, None, args.max_eval_batches)

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
    }
    save_json(args.output_dir / "split_summary.json", split_summary_payload)


if __name__ == "__main__":
    main()