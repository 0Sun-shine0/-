#!/usr/bin/env python3
"""
数据集截图修复脚本
==================

``ai_operator/data_generation/dataset_creator.py`` 的 ``create_app_interface()`` 以前
漏掉了绘制调用，导致 ``web_browser`` / ``text_editor`` / ``file_explorer`` 三类截图
全部是纯背景色的空白图（像素标准差 = 0，约占数据集的 40%）。该 bug 已修复，
但已经生成好的 ``ai_operator_dataset`` 里仍然是空白图。

本脚本**只重画图片，不改动任何标注**：对每张空白截图，读取它自己的标注
``elements``（每个元素的类型/边框/文字），按原主题底色重新渲染成有内容的界面。
原始空白图会被备份到 ``<data-root>/_blank_screenshots_backup/``，可随时还原。

用法::

    # 先看会改哪些文件
    .venv\\Scripts\\python.exe scripts/repair_screenshots.py --dry-run

    # 实际修复
    .venv\\Scripts\\python.exe scripts/repair_screenshots.py

    # 还原
    .venv\\Scripts\\python.exe scripts/repair_screenshots.py --restore
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

BACKUP_DIRNAME = "_blank_screenshots_backup"

#: 每种 UI 元素在浅色/深色主题下的填充色
ELEMENT_FILL_LIGHT: Dict[str, Tuple[int, int, int]] = {
    "input": (255, 255, 255),
    "menu": (225, 225, 225),
    "toolbar": (232, 232, 232),
    "statusbar": (225, 225, 225),
    "sidebar": (238, 238, 238),
    "editor": (255, 255, 255),
    "list": (252, 252, 252),
    "content": (255, 255, 255),
    "file": (248, 248, 248),
    "folder": (250, 240, 200),
    "tab": (0, 120, 215),
    "button": (0, 120, 215),
}
ELEMENT_FILL_DARK: Dict[str, Tuple[int, int, int]] = {
    "input": (70, 70, 70),
    "menu": (58, 58, 58),
    "toolbar": (52, 52, 52),
    "statusbar": (58, 58, 58),
    "sidebar": (45, 45, 45),
    "editor": (30, 30, 30),
    "list": (38, 38, 38),
    "content": (40, 40, 40),
    "file": (48, 48, 48),
    "folder": (72, 62, 30),
    "tab": (0, 100, 170),
    "button": (0, 100, 170),
}


def _load_font(size: int):
    for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc", "arial.ttf"):
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def fix_mojibake(text: str) -> str:
    """修复生成器写库时产生的 UTF-8/GBK 混淆文本（例如 '鍥炬爣3' -> '图标3'）。"""
    if not text or not any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return text
    try:
        repaired = text.encode("gbk").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            repaired = text.encode("latin-1").decode("gbk")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    # 修复结果必须确实是更"正常"的中文，否则保留原文
    return repaired if repaired and len(repaired) >= len(text) // 2 else text


def _deterministic_color(seed: str) -> Tuple[int, int, int]:
    digest = abs(hash(seed)) % (256 ** 3)
    r = 60 + (digest & 0xFF) % 160
    g = 60 + ((digest >> 8) & 0xFF) % 160
    b = 60 + ((digest >> 16) & 0xFF) % 160
    return (r, g, b)


def is_blank(path: Path) -> bool:
    """判断截图是否为纯色空白图。"""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    return bool(arr.std() == 0)


def render_from_annotation(annotation: dict, size: Tuple[int, int],
                           base_color: Tuple[int, int, int], sample_id: str,
                           fix_text: bool = True) -> Image.Image:
    """按标注里的 elements 重新渲染一张界面截图。"""
    width, height = size
    image = Image.new("RGB", (width, height), base_color)
    draw = ImageDraw.Draw(image, "RGBA")

    dark = sum(base_color) < 320
    palette = ELEMENT_FILL_DARK if dark else ELEMENT_FILL_LIGHT
    border = (95, 95, 95) if dark else (198, 198, 198)
    text_color = (238, 238, 238) if dark else (28, 28, 28)

    for index, element in enumerate(annotation.get("elements") or []):
        bounds = element.get("bounds") or []
        if len(bounds) != 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in bounds)
        x1, x2 = sorted((max(x1, 0), min(x2, width)))
        y1, y2 = sorted((max(y1, 0), min(y2, height)))
        if x2 <= x1 or y2 <= y1:
            continue

        kind = str(element.get("type", "")).lower()
        text = str(element.get("text", ""))
        if fix_text:
            text = fix_mojibake(text)

        if kind == "icon":
            fill = _deterministic_color(f"{sample_id}-{index}")
            draw.rectangle([x1, y1, x2, y2], fill=fill, outline=border, width=2)
            if text:
                font = _load_font(max(min((y2 - y1) // 2, 22), 10))
                draw.text((x1, y2 + 2), text, fill=text_color, font=font)
            continue

        if kind == "link":
            font = _load_font(16)
            draw.text((x1, y1), text, fill=(60, 120, 220) if not dark else (120, 180, 255), font=font)
            try:
                tw = int(draw.textlength(text, font=font))
            except Exception:
                tw = len(text) * 10
            draw.line([x1, y2 - 2, x1 + max(tw, 20), y2 - 2],
                      fill=(60, 120, 220) if not dark else (120, 180, 255), width=1)
            continue

        fill = palette.get(kind, border)
        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=border, width=1)
        if text:
            font = _load_font(15 if (y2 - y1) > 34 else 12)
            ty = y1 + max((y2 - y1) // 2 - 9, 3) if (y2 - y1) > 34 else y1 + 3
            draw.text((x1 + 6, ty), text, fill=text_color, font=font)

    return image


def repair(data_root: Path, dry_run: bool, fix_text: bool, force: bool) -> int:
    processed = skipped = 0
    backup_root = data_root / BACKUP_DIRNAME

    for split in ("train", "val", "test"):
        ann_dir = data_root / split / "annotations"
        img_dir = data_root / split / "images"
        if not ann_dir.is_dir():
            continue
        for ann_file in sorted(ann_dir.glob("*.json")):
            with open(ann_file, "r", encoding="utf-8") as f:
                annotation = json.load(f)
            sample_id = annotation.get("sample_id") or ann_file.stem
            image_path = img_dir / sample_id / "screenshot.png"
            if not image_path.exists():
                continue

            if not force and not is_blank(image_path):
                skipped += 1
                continue

            with Image.open(image_path) as im:
                im = im.convert("RGB")
                size = im.size
                base_color = im.getpixel((0, 0))

            if dry_run:
                elements = len(annotation.get("elements") or [])
                print(f"  [dry-run] {split}/{sample_id}: {size[0]}x{size[1]} 底色{base_color} -> 重画 {elements} 个元素")
                processed += 1
                continue

            backup_path = backup_root / split / sample_id / "screenshot.png"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                shutil.copy2(image_path, backup_path)

            rendered = render_from_annotation(annotation, size, base_color, sample_id, fix_text)
            rendered.save(image_path, "PNG")
            processed += 1

    verb = "将修复" if dry_run else "已修复"
    print(f"\n{verb} {processed} 张空白截图，跳过 {skipped} 张非空白截图")
    if not dry_run and processed:
        print(f"原图备份: {backup_root}")
    return 0


def restore(data_root: Path) -> int:
    backup_root = data_root / BACKUP_DIRNAME
    if not backup_root.is_dir():
        print(f"没有找到备份目录: {backup_root}")
        return 1
    restored = 0
    for split_dir in backup_root.iterdir():
        if not split_dir.is_dir():
            continue
        for sample_dir in split_dir.iterdir():
            src = sample_dir / "screenshot.png"
            dst = data_root / split_dir.name / "images" / sample_dir.name / "screenshot.png"
            if src.exists() and dst.parent.exists():
                shutil.copy2(src, dst)
                restored += 1
    print(f"已还原 {restored} 张截图")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="修复数据集中的空白截图")
    parser.add_argument("--data-root", default=str(ROOT / "ai_operator_dataset"))
    parser.add_argument("--dry-run", action="store_true", help="只列出会被修复的文件")
    parser.add_argument("--restore", action="store_true", help="从备份还原原始空白图")
    parser.add_argument("--force", action="store_true", help="对非空白截图也重新渲染")
    parser.add_argument("--no-fix-text", action="store_true", help="不修复标注文本的乱码")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.restore:
        return restore(data_root)
    return repair(data_root, args.dry_run, not args.no_fix_text, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
