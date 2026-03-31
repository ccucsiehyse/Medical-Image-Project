"""
offline_augment.py

將指定類別的影像離線擴充，生成帶些微差異的版本存成新檔案。
預設針對 ModerateDemented，每張原圖生成 N 張變體。

執行：
    python offline_augment.py \
        --src-dir "Alzheimer_s Dataset/train/ModerateDemented" \
        --dst-dir "Alzheimer_s Dataset/train/ModerateDemented_aug" \
        --target-count 520 \
        --seed 42
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir",      type=Path, required=True,
                   help="原始類別資料夾（例如 .../ModerateDemented）")
    p.add_argument("--dst-dir",      type=Path, required=True,
                   help="增強後的輸出資料夾（不存在則自動建立）")
    p.add_argument("--target-count", type=int, default=520,
                   help="目標總張數（含原圖）")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--save-format",  type=str, default="jpg",
                   choices=["png", "jpg"])
    return p.parse_args()


def augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    """
    對單張影像套用一組隨機但保守的增強。
    每次呼叫的參數都由 rng 獨立取樣，確保每張變體都不同。
    """
    img = image.copy().convert("RGB")

    # 1. 水平翻轉（50%）
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 2. 隨機旋轉 ±15°
    angle = rng.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

    # 3. 輕微縮放 + 裁切回原尺寸
    w, h = img.size
    scale = rng.uniform(0.92, 1.08)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    # 裁切或補邊回原尺寸
    if scale >= 1.0:
        left = (new_w - w) // 2
        top  = (new_h - h) // 2
        img  = img.crop((left, top, left + w, top + h))
    else:
        new_img = Image.new("RGB", (w, h), (0, 0, 0))
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2
        new_img.paste(img, (offset_x, offset_y))
        img = new_img

    # 4. 亮度微調（CT 影像保守範圍）
    brightness_factor = rng.uniform(0.85, 1.15)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # 5. 對比度微調
    contrast_factor = rng.uniform(0.85, 1.15)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # 6. 輕微高斯雜訊（模擬 CT 掃描雜訊）
    if rng.random() < 0.4:
        arr   = np.asarray(img, dtype=np.float32)
        noise = rng.gauss(0, 3)   # sigma=3，非常輕微
        arr   = np.clip(arr + np.random.normal(0, abs(noise), arr.shape), 0, 255)
        img   = Image.fromarray(arr.astype(np.uint8))

    return img


def main():
    args = parse_args()
    rng  = random.Random(args.seed)
    np.random.seed(args.seed)

    src_files = sorted(
        p for p in args.src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not src_files:
        raise FileNotFoundError(f"找不到影像：{args.src_dir}")

    n_original = len(src_files)
    n_needed   = args.target_count - n_original
    if n_needed <= 0:
        print(f"原始圖片數（{n_original}）已達目標（{args.target_count}），無需增強。")
        return

    args.dst_dir.mkdir(parents=True, exist_ok=True)

    # 先將原圖複製到 dst_dir
    for src in src_files:
        dst = args.dst_dir / src.name
        Image.open(src).save(dst)
    print(f"已複製 {n_original} 張原圖到 {args.dst_dir}")

    # 生成增強圖片，均勻輪流從原圖取樣
    generated = 0
    suffix    = f".{args.save_format}"
    while generated < n_needed:
        src = src_files[generated % n_original]
        with Image.open(src) as base:
            aug = augment_image(base, rng)
        out_name = f"{src.stem}_{generated:05d}{suffix}"
        aug.save(args.dst_dir / out_name)
        generated += 1
        if generated % 50 == 0:
            print(f"  已生成 {generated}/{n_needed} 張...")

    print(f"\n完成！{args.dst_dir} 共 {n_original + generated} 張")
    print(f"（原圖 {n_original} + 增強 {generated}）")


if __name__ == "__main__":
    main()
