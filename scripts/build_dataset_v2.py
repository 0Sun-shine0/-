#!/usr/bin/env python3
"""
数据集 v2 生成器 —— 修正因果链 + 烘烤光标
=========================================

为什么需要 v2
-------------
v1 数据集有两个使任务**在信息论上无解**的缺陷：

1. ``dataset_creator.generate_app_specific_operation()`` 先 ``random.choice`` 选动作，
   再独立 ``random.choice`` 选目标元素，两者之间没有因果关系。
2. 截图里没有光标 / 高亮框，"用户点了哪里"在像素上不可见。

结果：模型既不知道点了哪个元素，也无法从画面推断动作，任何模型的上限都只有"猜最常见的类"。

v2 的两处修正
-------------
1. **因果动作**：先选目标元素，再由元素类型**唯一决定**动作
   （``ELEMENT_ACTION``）。同一个元素类型永远对应同一个动作。
2. **烘烤光标**：在目标元素中心画上箭头光标。这既让坐标回归可解，
   也让模型能通过"光标压在哪个元素上"来推断动作——这正是真实屏幕智能体的工作原理
   （真实桌面上光标本来就停在点击位置）。

输出目录与 v1 **完全同构**，因此 ``train_ai_operator.py`` / ``demo_desktop_agent.py``
无需任何改动即可直接使用::

    <out>/<split>/images/<sample_id>/screenshot.png
    <out>/<split>/annotations/<sample_id>.json

用法::

    python scripts/build_dataset_v2.py --out ai_operator_dataset_v2 --train 560 --val 120 --test 120
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

from ai_operator.data.desktop_dataset import cursor_points  # noqa: E402

# --------------------------------------------------------------------------- #
# 元素类型 -> 动作（v2 的核心：唯一确定的因果映射）
# --------------------------------------------------------------------------- #
ELEMENT_ACTION: Dict[str, str] = {
    "address_bar": "click_address_bar",
    "button": "left_click",
    "editor": "click_editor",
    "file": "click_file",
    "folder": "click_folder",
    "icon": "double_click",
    "link": "click_link",
    "list": "scroll_list",
    "menu": "click_menu",
    "sidebar": "click_sidebar",
    "tab": "click_tab",
    "toolbar": "click_toolbar",
    "content": "scroll_page",
    "start_button": "left_click",
}

#: 每个元素类型的外观：填充色（深色主题）, 填充色（浅色主题）, 是否画文字
ELEMENT_STYLE: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    #                        (dark fill)          (light fill)
    "address_bar": ((66, 66, 66), (255, 255, 255)),
    "button": ((0, 110, 190), (0, 120, 215)),
    "editor": ((28, 28, 28), (255, 255, 255)),
    "file": ((52, 52, 52), (246, 246, 246)),
    "folder": ((94, 78, 30), (250, 214, 120)),
    "icon": ((0, 110, 190), (0, 120, 215)),
    "link": ((28, 28, 28), (255, 255, 255)),
    "list": ((40, 40, 40), (250, 250, 250)),
    "menu": ((56, 56, 56), (228, 228, 228)),
    "sidebar": ((44, 44, 44), (236, 236, 236)),
    "tab": ((0, 96, 160), (0, 120, 215)),
    "toolbar": ((50, 50, 50), (232, 232, 232)),
    "content": ((35, 35, 35), (252, 252, 252)),
    "start_button": ((0, 110, 190), (0, 120, 215)),
}

THEMES: Dict[str, Dict[str, object]] = {
    "windows_light": {"bg": (240, 240, 240), "text": (28, 28, 28), "line": (200, 200, 200), "dark": False},
    "windows_dark": {"bg": (32, 32, 32), "text": (235, 235, 235), "line": (100, 100, 100), "dark": True},
    "macos_light": {"bg": (251, 251, 251), "text": (20, 20, 20), "line": (209, 209, 214), "dark": False},
    "macos_dark": {"bg": (28, 28, 30), "text": (240, 240, 240), "line": (72, 72, 74), "dark": True},
}

RESOLUTIONS: Sequence[Tuple[int, int]] = ((1920, 1080), (1600, 900), (1366, 768), (2560, 1440))

#: sample_id 前缀 -> 应用；必须与 demo 脚本的 APP_LABELS 对齐
APP_PREFIX: Dict[str, str] = {
    "desktop": "basic",
    "file_explorer": "file_explorer",
    "text_editor": "text_editor",
    "web_browser": "web_browser",
}


def _font(size: int):
    for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc", "arial.ttf"):
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            try:
                return ImageFont.truetype(str(path), max(size, 10))
            except OSError:
                continue
    return ImageFont.load_default()


class Element:
    """一个待绘制的 UI 元素。"""

    __slots__ = ("kind", "bounds", "text")

    def __init__(self, kind: str, bounds: Tuple[int, int, int, int], text: str = "") -> None:
        self.kind = kind
        self.bounds = bounds
        self.text = text

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bounds
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_json(self) -> Dict[str, object]:
        return {
            "type": self.kind,
            "bounds": list(self.bounds),
            "text": self.text,
            "action": ELEMENT_ACTION[self.kind],
        }


# --------------------------------------------------------------------------- #
# 各应用的界面布局
# --------------------------------------------------------------------------- #
def layout_web_browser(w: int, h: int, rng: random.Random) -> List[Element]:
    elements = [
        Element("address_bar", (60, 12, w - 260, 52), "https://www.example.com"),
        Element("toolbar", (60, 62, w - 260, 96), ""),
    ]
    tab_w = 170
    for i in range(3):
        elements.append(Element("tab", (60 + i * tab_w, 100, 60 + (i + 1) * tab_w - 6, 134), f"标签页 {i + 1}"))
    elements.append(Element("content", (60, 146, w - 60, h - 60), ""))
    elements.append(Element("button", (140, 230, 340, 290), "点击这里"))
    elements.append(Element("link", (140, 340, 460, 372), "了解更多"))
    return elements


def layout_text_editor(w: int, h: int, rng: random.Random) -> List[Element]:
    elements = [
        Element("menu", (0, 0, w, 34), "文件  编辑  视图  帮助"),
        Element("toolbar", (0, 34, w, 74), ""),
        Element("sidebar", (0, 74, 230, h - 34), "资源管理器"),
        Element("tab", (230, 74, 560, 108), "main.py"),
        Element("editor", (230, 108, w, h - 34), "# Welcome to Code Editor"),
    ]
    return elements


def layout_file_explorer(w: int, h: int, rng: random.Random) -> List[Element]:
    elements = [
        Element("address_bar", (60, 12, w - 80, 52), "C:\\Users\\Documents"),
        Element("sidebar", (0, 66, 240, h - 40), "快速访问"),
        Element("list", (240, 66, w, h - 40), ""),
    ]
    for i in range(4):
        elements.append(Element("folder", (270, 90 + i * 46, 560, 90 + i * 46 + 38), f"文件夹{i + 1}"))
    for i in range(4):
        elements.append(Element("file", (600, 90 + i * 46, 900, 90 + i * 46 + 38), f"文件{i + 1}.txt"))
    return elements


def layout_desktop(w: int, h: int, rng: random.Random) -> List[Element]:
    elements: List[Element] = []
    taskbar = 46
    elements.append(Element("start_button", (16, h - taskbar + 8, 96, h - 8), "开始"))
    # 图标放在固定的网格上，避免相互重叠导致光标指代不明
    cols, rows = 6, 3
    step_x = (w - 240) // cols
    step_y = (h - taskbar - 200) // rows
    slots = [(160 + c * step_x, 90 + r * step_y) for r in range(rows) for c in range(cols)]
    rng.shuffle(slots)
    for i, (x, y) in enumerate(slots[:9]):
        kind = "icon" if i % 2 == 0 else "folder"
        elements.append(Element(kind, (x, y, x + 74, y + 74), f"{'应用' if kind == 'icon' else '目录'}{i + 1}"))
    return elements


LAYOUTS = {
    "desktop": layout_desktop,
    "file_explorer": layout_file_explorer,
    "text_editor": layout_text_editor,
    "web_browser": layout_web_browser,
}


# --------------------------------------------------------------------------- #
# 绘制
# --------------------------------------------------------------------------- #
def draw_element(painter: ImageDraw.ImageDraw, element: Element, w: int, h: int,
                 theme: Dict[str, object], rng: random.Random) -> None:
    x1, y1, x2, y2 = element.bounds
    dark = bool(theme["dark"])
    text_color = theme["text"]  # type: ignore[assignment]
    line_color = theme["line"]  # type: ignore[assignment]
    fill_dark, fill_light = ELEMENT_STYLE[element.kind]
    fill = fill_dark if dark else fill_light
    radius = 6

    if element.kind == "link":
        # 链接不填充背景，只画蓝色带下划线的文字（靠光标位置区分）
        font = _font(20)
        painter.text((x1, y1), element.text, fill=(110, 170, 255) if dark else (40, 100, 220), font=font)
        try:
            tw = int(painter.textlength(element.text, font=font))
        except Exception:
            tw = len(element.text) * 12
        painter.line([x1, y2 - 3, x1 + max(tw, 24), y2 - 3],
                     fill=(110, 170, 255) if dark else (40, 100, 220), width=2)
        return

    if element.kind == "tab":
        painter.rounded_rectangle([x1, y1, x2, y2 + 10], radius=radius, fill=fill, outline=line_color, width=1)
    elif element.kind in ("menu", "toolbar"):
        painter.rectangle([x1, y1, x2, y2], fill=fill)
        painter.line([x1, y2, x2, y2], fill=line_color, width=1)
    elif element.kind in ("editor", "list", "content", "sidebar"):
        painter.rectangle([x1, y1, x2, y2], fill=fill, outline=line_color, width=1)
    elif element.kind in ("icon", "folder", "file", "button", "start_button"):
        painter.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=line_color, width=2)
    else:  # address_bar
        painter.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=line_color, width=2)

    if element.text:
        font = _font(min(int((y2 - y1) * 0.5), 22))
        if element.kind in ("icon", "folder"):
            # 图标类文字写在方块下方，与实际桌面一致，也让方块区域本身保持纯净
            painter.text((x1, y2 + 3), element.text, fill=text_color, font=font)
        else:
            offset = max((y2 - y1) // 2 - int((y2 - y1) * 0.25), 3)
            painter.text((x1 + 10, y1 + offset), element.text, fill=text_color, font=font)


def render_sample(app: str, w: int, h: int, theme_name: str, rng: random.Random
                  ) -> Tuple[Image.Image, Element, List[Element]]:
    """渲染一张截图，返回 (图像, 被点击的元素, 全部元素)。"""
    theme = THEMES[theme_name]
    image = Image.new("RGB", (w, h), color=theme["bg"])  # type: ignore[arg-type]
    painter = ImageDraw.Draw(image)

    elements = LAYOUTS[app](w, h, rng)
    for element in elements:
        draw_element(painter, element, w, h, theme, rng)

    # ---- v2 的核心修正 1：先选元素，动作由元素类型决定（不再独立随机） ----
    target = rng.choice(elements)

    # ---- v2 的核心修正 2：把光标画在目标元素中心 ----
    # 尺寸按宽度缩放，保证缩放到 224 输入后仍然是约 15px 的清晰光标
    cursor_size = max(16, int(round(15 * w / 224)))
    points = cursor_points(target.center[0], target.center[1], cursor_size)
    painter.polygon(points, fill=(255, 255, 255), outline=(0, 0, 0))
    painter.line(points + [points[0]], fill=(0, 0, 0), width=max(1, int(round(cursor_size / 7))))

    return image, target, elements


# --------------------------------------------------------------------------- #
# 生成 / 划分
# --------------------------------------------------------------------------- #
def build_split(out_root: Path, split: str, count: int, per_app: int, rng: random.Random,
                with_cursor: bool) -> Dict[str, object]:
    images_dir = out_root / split / "images"
    ann_dir = out_root / split / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    apps = list(LAYOUTS.keys())
    action_counts: Counter = Counter()
    app_counts: Counter = Counter()
    element_counts: Counter = Counter()
    written = 0

    for i in range(count):
        app = apps[i % len(apps)]
        w, h = rng.choice(RESOLUTIONS)
        theme_name = rng.choice(list(THEMES.keys()))
        image, target, elements = render_sample(app, w, h, theme_name, rng)

        prefix = APP_PREFIX[app]
        sample_id = f"{prefix}_{i:06d}"
        sample_dir = images_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        image.save(sample_dir / "screenshot.png", "PNG", optimize=True)

        x1, y1, x2, y2 = target.bounds
        annotation = {
            "sample_id": sample_id,
            "image_path": f"images/{sample_id}/screenshot.png",
            "app": app,
            "theme": theme_name,
            "resolution": f"{w}x{h}",
            "operation": {
                "action": ELEMENT_ACTION[target.kind],
                "target_element": target.kind,
                "target_bounds": [x1, y1, x2, y2],
                "target_text": target.text,
                "params": {},
                "timestamp": datetime.now().timestamp(),
                "cursor_drawn": bool(with_cursor),
            },
            "elements": [e.to_json() for e in elements],
            "created_at": datetime.now().isoformat(),
        }
        with open(ann_dir / f"{sample_id}.json", "w", encoding="utf-8") as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)

        written += 1
        action_counts[ELEMENT_ACTION[target.kind]] += 1
        app_counts[app] += 1
        element_counts[target.kind] += 1

    return {
        "split": split,
        "samples": written,
        "actions": dict(sorted(action_counts.items(), key=lambda kv: -kv[1])),
        "apps": dict(app_counts),
        "targeted_elements": dict(sorted(element_counts.items(), key=lambda kv: -kv[1])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成因果一致、带光标的桌面数据集 v2")
    parser.add_argument("--out", default=str(ROOT / "ai_operator_dataset_v2"))
    parser.add_argument("--train", type=int, default=560)
    parser.add_argument("--val", type=int, default=120)
    parser.add_argument("--test", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--no-cursor", action="store_true",
                        help="不烘烤光标，用于做 A/B 对照实验")
    args = parser.parse_args()

    out_root = Path(args.out)
    rng = random.Random(args.seed)
    with_cursor = not args.no_cursor

    print("=" * 72)
    print("生成数据集 v2（因果动作 + 烘烤光标）")
    print("=" * 72)
    print(f"输出目录: {out_root}")
    print(f"光标: {'开启' if with_cursor else '关闭（对照实验）'}")

    summaries = []
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        print(f"\n生成 {split}: {count} 样本 ...", flush=True)
        summary = build_split(out_root, split, count, len(LAYOUTS), rng, with_cursor)
        summaries.append(summary)
        print(f"  完成，动作分布: {summary['actions']}")

    manifest = {
        "name": "ai_operator_dataset_v2",
        "created_at": datetime.now().isoformat(),
        "seed": args.seed,
        "cursor_drawn": with_cursor,
        "element_action_map": ELEMENT_ACTION,
        "resolutions": [f"{w}x{h}" for w, h in RESOLUTIONS],
        "themes": list(THEMES.keys()),
        "splits": summaries,
        "notes": [
            "动作由目标元素类型唯一决定（因果一致），不再与元素独立随机。",
            "光标画在目标元素中心，使坐标回归与动作推断都成为良定义任务。",
        ],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "dataset_v2_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n清单: {out_root / 'dataset_v2_manifest.json'}")
    print("下一步: python scripts/train_ai_operator.py "
          f"--data-root {out_root} --out-dir models/desktop_v2 --epochs 20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
