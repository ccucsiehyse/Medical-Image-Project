"""
augment_train_horizontal_flip.py

把訓練集（train）做「水平翻轉」資料增強，並將資料量加倍（double）。

預期的資料夾結構：
    train_dir/
        ClassName1/
            xxx.jpg / xxx.png / ...
        ClassName2/
            ...

輸出結構（會保持相同類別子資料夾）：
    output_train_dir/
        ClassName1/
            xxx.jpg               (原圖，預設會複製)
            xxx_flip.jpg         (翻轉後的圖)
        ClassName2/
            ...

此腳本不使用 argparse；所有可調參數都集中在 __main__ 區塊中設定呼叫 main()。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _ensure_format_for_suffix(suffix: str) -> str | None:
    """根據副檔名給 PIL 顯式格式名稱（某些情境下更穩）。"""
    s = suffix.lower()
    if s in {".tif", ".tiff"}:
        return "TIFF"
    if s == ".webp":
        return "WEBP"
    # 多數副檔名 PIL 可自動推斷，這裡不強制
    return None


def augment_train_horizontal_flip(
    *,
    train_dir: Path,
    output_train_dir: Path,
    copy_original: bool = True,
    create_flip: bool = True,
    overwrite: bool = False,
) -> None:
    """
    對 train_dir 中每一個類別子資料夾：
    - 選擇性複製原圖到 output_train_dir
    - 產生水平翻轉版本，並存成 *_flip<suffix>
    """
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir 不存在：{train_dir}")

    class_dirs = [p for p in train_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"train_dir 下沒有任何類別資料夾：{train_dir}")

    output_train_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    flipped_files = 0
    copied_files = 0

    for class_dir in sorted(class_dirs, key=lambda p: p.name):
        out_class_dir = output_train_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        # 允許資料夾底下有巢狀結構，使用 rglob 掃描
        image_paths = [
            p
            for p in sorted(class_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image_path in image_paths:
            total_files += 1

            suffix = image_path.suffix
            stem = image_path.stem

            # 原圖輸出檔名：維持同樣檔名（不改副檔名）
            out_original_path = out_class_dir / f"{stem}{suffix}"
            # 翻轉圖輸出檔名：在 stem 後加 _flip
            out_flip_path = out_class_dir / f"{stem}_flip{suffix}"

            # 1) 複製原圖（讓資料量 double 成兩份：原圖 + 翻轉圖）
            if copy_original:
                if overwrite or not out_original_path.exists():
                    # copy2 保留修改時間與檔案屬性（通常對報告/追溯更好）
                    shutil.copy2(image_path, out_original_path)
                    copied_files += 1

            # 2) 產生翻轉圖
            if create_flip:
                if overwrite or not out_flip_path.exists():
                    with Image.open(image_path) as img:
                        # 翻轉不需要做太多前處理；只確保有像素資料即可
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)

                        save_format = _ensure_format_for_suffix(suffix)
                        if save_format is None:
                            img.save(out_flip_path)
                        else:
                            img.save(out_flip_path, format=save_format)
                        flipped_files += 1

        print(
            f"[{class_dir.name}] 原圖處理完成：total={len(image_paths)} "
            f"copied={copied_files} flipped_so_far={flipped_files}"
        )

    print(
        "Done.\n"
        f"  train_dir: {train_dir}\n"
        f"  output_train_dir: {output_train_dir}\n"
        f"  total_images_scanned: {total_files}\n"
        f"  copied_original_images: {copied_files}\n"
        f"  created_flipped_images: {flipped_files}\n"
    )


def main(
    *,
    train_dir: Path,
    output_train_dir: Path,
) -> None:
    augment_train_horizontal_flip(
        train_dir=train_dir,
        output_train_dir=output_train_dir,
        copy_original=True,  # double：原圖 + 翻轉圖
        create_flip=True,
        overwrite=False,  # 已存在就跳過（避免覆蓋）
    )


if __name__ == "__main__":
    # 請改成你的實際 train 資料夾路徑
    _train_dir = Path(r"../split/train")

    # 請改成你希望輸出的資料夾路徑
    _output_train_dir = Path(r"../split/train_double")

    main(
        train_dir=_train_dir,
        output_train_dir=_output_train_dir,
    )

