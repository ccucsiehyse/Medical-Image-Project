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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an Alzheimer image classifier from a single folder split."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Alzheimer_s Dataset/train"),
        help="Directory that contains class subfolders. Only this folder is used for train/val/test splitting.",
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

    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1).")
    if not 0.0 <= args.test_ratio < 1.0:
        raise ValueError("--test-ratio must be in [0, 1).")
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("--val-ratio + --test-ratio must be < 1.")
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
    if not train_samples:
        raise RuntimeError("Split failed because no training samples were generated.")
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
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        return image_tensor


class AlzheimerDataset(Dataset):
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
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
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
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
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
    dataset = AlzheimerDataset(samples=samples, transform=BasicImageTransform(image_size, train=train))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
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
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
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

    if total_samples == 0:
        raise RuntimeError("No samples were processed. Check split ratios or debug batch limits.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


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
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")

    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    save_json(
        args.output_dir / "split_summary.json",
        {
            "data_dir": str(args.data_dir),
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
        },
    )


if __name__ == "__main__":
    main()
