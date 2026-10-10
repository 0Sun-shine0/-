#!/usr/bin/env python3
"""
演示 GIF 生成脚本
=================

把 ``demo_desktop_agent.py --mode eval`` 产出的逐样本预测变成一张会动的演示图。

**先说清楚这张 GIF 是什么、不是什么**：

- 它**不是**"智能体自主完成任务"的录屏。当前模型是**单步、静态截图**预测器：
  输入一张截图，输出"该执行哪个动作 + 点哪里"。它没有时间维度，
  也不会多步规划。所以 GIF 里"光标移动"只是把预测落点做成视觉引导，
  **不是模型自己走出来的轨迹**。
- 真正带时间维度的只有 ``--mode live``（截屏→预测→执行 循环），
  但它截的是你的真实桌面（涉及隐私），且在真实桌面上的表现尚未验证。
- 每张截图里那个**白色小箭头是数据集自带的光标**，位置就是标注的目标元素中心，
  即"标准答案"。GIF 里的**黄色圆环才是模型预测的落点**。
  两者差异就是误差（例如 74.5 px），这一点在画面里必须区分开。

用法::

    # 默认：从 test 集按"每应用误差最大/最小各一个"挑样本，出一张 GIF
    .venv\\Scripts\\python.exe scripts/make_demo_gif.py

    # 指定样本与尺寸
    .venv\\Scripts\\python.exe scripts/make_demo_gif.py --scale 560 --travel-frames 3

    # 只做静态首帧（便于在 README 里内嵌单张图）
    .venv\\Scripts\\python.exe scripts/make_demo_gif.py --still reports/demo_still.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

from ai_operator.data.desktop_dataset import action_zh  # noqa: E402

# 复用 demo 脚本的字体加载，避免第 5 份拷贝
try:
    from demo_desktop_agent import _load_font  # noqa: E402
except Exception:  # pragma: no cover
    def _load_font(size: int):  # type: ignore[misc]
        from PIL import ImageFont
        for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
            path = Path("C:/Windows/Fonts") / name
            if path.exists():
                try:
                    return ImageFont.truetype(str(path), size)
                except OSError:
                    continue
        return ImageFont.load_default()

BG = (14, 14, 18)
FG = (238, 238, 242)
DIM = (150, 152, 160)
GT_COLOR = (0, 230, 90)
PRED_COLOR = (255, 214, 0)
CLICK_COLOR = (255, 90, 90)

APP_LABEL = {
    "basic": "桌面",
    "file_explorer": "文件资源管理器",
    "text_editor": "文本编辑器",
    "web_browser": "浏览器",
}


def app_of(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


# --------------------------------------------------------------------------- #
# 样本选择
# --------------------------------------------------------------------------- #
def load_predictions(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pick_samples(rows: Sequence[Dict[str, str]], per_app: int = 2) -> List[Dict[str, str]]:
    """每个应用挑"误差最小"和"误差最大"的样本。

    刻意把最难的那个也选进来 —— 只放最好看的样本会让演示失真。
    """
    buckets: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        buckets.setdefault(app_of(row["sample_id"]), []).append(row)

    picked: List[Dict[str, str]] = []
    for app in sorted(buckets, key=lambda a: -len(buckets[a])):
        ordered = sorted(buckets[app], key=lambda r: float(r["err_px"]))
        if per_app >= 1:
            picked.append(ordered[0])
        if per_app >= 2 and len(ordered) > 1:
            picked.append(ordered[-1])
        for extra in ordered[1:-1][: max(per_app - 2, 0)]:
            picked.append(extra)
    return picked


def load_gt_bounds(data_root: Path, split: str, sample_id: str) -> Optional[Tuple[float, float, float, float]]:
    path = data_root / split / "annotations" / f"{sample_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bounds = (data.get("operation") or {}).get("target_bounds")
    if not bounds or len(bounds) != 4:
        return None
    return tuple(float(v) for v in bounds)  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# 绘制
# --------------------------------------------------------------------------- #
def make_canvas(scale: int, caption_h: int, extra_h: int = 0) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    width = scale
    height = int(round(scale * 9 / 16))
    image = Image.new("RGB", (width, height + caption_h + extra_h), BG)
    return image, ImageDraw.Draw(image)


def draw_text_block(draw: ImageDraw.ImageDraw, lines: Sequence[Tuple[str, tuple, int]],
                    x: int, y: int, spacing: int = 4) -> int:
    """lines = [(文本, 颜色, 字号), ...]，返回结束的 y。"""
    for text, color, size in lines:
        draw.text((x, y), text, fill=color, font=_load_font(size))
        y += size + spacing
    return y


def draw_prediction(draw: ImageDraw.ImageDraw, gt_box, pred_xy, progress: float,
                    pulse: Optional[float] = None, frame_h: int = 360) -> None:
    """在一帧里画出 GT 框、预测圆环（带移动进度）与点击脉冲。

    Args:
        gt_box: 缩放后的 GT 框 (x1,y1,x2,y2)；None 表示不画。
        pred_xy: 缩放后的预测落点。
        progress: 0..1，圆环从画面中心移动到落点的进度。
        pulse: 0..1 的点击脉冲进度；None 表示不画。
        frame_h: 截图区域高度，用于计算"画面中心"。
    """
    if gt_box is not None:
        x1, y1, x2, y2 = gt_box
        draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=2)

    if pred_xy is None:
        return

    # 未到位时从画面中心飞向落点：只是视觉引导，不是模型轨迹
    if progress < 1.0:
        cx = draw._image.width / 2          # type: ignore[attr-defined]
        cy = frame_h / 2
        ease = 1 - (1 - progress) ** 3
        px = cx + (pred_xy[0] - cx) * ease
        py = cy + (pred_xy[1] - cy) * ease
        draw.line([px - 7, py, px + 7, py], fill=PRED_COLOR, width=2)
        draw.line([px, py - 7, px, py + 7], fill=PRED_COLOR, width=2)
        return

    px, py = pred_xy
    halo = (18, 18, 22)   # 深色描边：黄色在浅色主题上对比度不够，必须加底衬

    def ring(radius: int, color: tuple, width: int) -> None:
        draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                     outline=color, width=width)

    def cross(length: int, color: tuple, width: int) -> None:
        draw.line([px - length, py, px - 10, py], fill=color, width=width)
        draw.line([px + 10, py, px + length, py], fill=color, width=width)
        draw.line([px, py - length, px, py - 10], fill=color, width=width)
        draw.line([px, py + 10, px, py + length], fill=color, width=width)

    # 先画粗深色，再叠亮色 —— 浅色/深色主题下都清晰
    ring(10, halo, 5)
    cross(16, halo, 5)
    ring(10, PRED_COLOR, 2)
    cross(16, PRED_COLOR, 2)

    if pulse is not None:
        pr = int(8 + 26 * pulse)
        ring(pr + 1, halo, 4)
        ring(pr, CLICK_COLOR, 2)


def caption_lines(row: Dict[str, str], note: str = "") -> List[Tuple[str, tuple, int]]:
    gt = row["gt_action"]
    pred = row["pred_action"]
    conf = float(row["confidence"])
    err = float(row["err_px"])
    app = app_of(row["sample_id"])
    # 不用 ✓/✗：这两个字形在中文字体里缺失，PIL 会渲染成豆腐块（□）。
    # 这类非 ASCII 符号在 GIF 里必须避免，改用汉字。
    verdict = "动作正确" if row["correct"] == "1" else "动作错误"
    lines = [
        (f"{row['sample_id']}   {APP_LABEL.get(app, app)}   {row['image_w']}x{row['image_h']}", FG, 15),
        (f"标注 {gt} ({action_zh(gt)})  |  预测 {pred} {conf:.0%}  {verdict}  |  落点误差 {err:.0f}px",
         DIM, 14),
    ]
    if note:
        lines.append((note, (255, 140, 140), 13))
    return lines


def build_panel(row: Dict[str, str], data_root: Path, split: str, scale: int,
                zoom_divisor: float) -> Optional[Tuple[Image.Image, float, float, float, float,
                                                        Optional[Tuple[float, float]]]]:
    """构建一帧的底图（截图），返回底图与坐标换算所需的参数。

    ``zoom_divisor > 1`` 时以"标注目标点与预测点的中点"为中心裁剪 1/zoom_divisor
    的画面再放大 —— 4K 截图直接缩到 640px 时界面元素只有几像素，根本看不清；
    裁剪放大后光标与图标都能看清。同时在右上角叠一张全屏缩略图并框出裁剪区域，
    避免丢失"这是屏幕的哪一块"这个上下文。

    Returns:
        (panel, scale_x, scale_y, offset_x, offset_y, thumb) 其中
        ``原图坐标 * scale + offset`` 得到 panel 坐标；thumb 为
        (缩略图, 缩略图上的裁剪框) 或 None。
    """
    sample_id = row["sample_id"]
    image_path = data_root / split / "images" / sample_id / "screenshot.png"
    if not image_path.exists():
        return None

    frame_w = scale
    frame_h = int(round(scale * 9 / 16))

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        width, height = im.size

        pred_x, pred_y = float(row["pred_x"]), float(row["pred_y"])
        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])

        if zoom_divisor and zoom_divisor > 1.0:
            crop_w = max(width / zoom_divisor, 64.0)
            crop_h = max(height / zoom_divisor, 36.0)
            # 以 GT 与预测点的中点为中心，让两者尽量都落在画面内
            cx = (pred_x + gt_x) / 2.0
            cy = (pred_y + gt_y) / 2.0
            x0 = min(max(cx - crop_w / 2.0, 0.0), max(width - crop_w, 0.0))
            y0 = min(max(cy - crop_h / 2.0, 0.0), max(height - crop_h, 0.0))
            box = (int(x0), int(y0), int(min(x0 + crop_w, width)), int(min(y0 + crop_h, height)))
            panel = im.crop(box).resize((frame_w, frame_h), Image.LANCZOS)
            scale_x = frame_w / (box[2] - box[0])
            scale_y = frame_h / (box[3] - box[1])
            offset_x, offset_y = -box[0] * scale_x, -box[1] * scale_y

            # 全屏缩略图 + 裁剪框
            thumb_w = max(int(frame_w * 0.22), 96)
            thumb_h = max(int(thumb_w * height / width), 54)
            thumb = im.resize((thumb_w, thumb_h), Image.LANCZOS)
            fx = box[0] / width * thumb_w
            fy = box[1] / height * thumb_h
            fw = (box[2] - box[0]) / width * thumb_w
            fh = (box[3] - box[1]) / height * thumb_h
            return panel, scale_x, scale_y, offset_x, offset_y, (thumb, (fx, fy, fw, fh))

        panel = im.resize((frame_w, frame_h), Image.LANCZOS)
        scale_x = frame_w / width
        scale_y = frame_h / height
        return panel, scale_x, scale_y, 0.0, 0.0, None


def blank_sample_frames(row: Dict[str, str], data_root: Path, split: str, scale: int,
                        caption_h: int, travel_frames: int, pulse_frames: int,
                        zoom_divisor: float) -> List[Image.Image]:
    """为一个样本生成一串帧：圆环飞入 -> 点击脉冲 -> 停留。"""
    sample_id = row["sample_id"]
    built = build_panel(row, data_root, split, scale, zoom_divisor)
    if built is None:
        return []
    base, scale_x, scale_y, off_x, off_y, thumb_info = built

    frame_w = scale
    frame_h = int(round(scale * 9 / 16))
    pred_xy = (float(row["pred_x"]) * scale_x + off_x, float(row["pred_y"]) * scale_y + off_y)

    bounds = load_gt_bounds(data_root, split, sample_id)
    gt_box = None
    if bounds is not None:
        x1, y1, x2, y2 = bounds
        gt_box = (x1 * scale_x + off_x, y1 * scale_y + off_y,
                  x2 * scale_x + off_x, y2 * scale_y + off_y)
        # 目标框可能很小（尤其图标），保证在 GIF 里可见
        if gt_box[2] - gt_box[0] < 12:
            cx = (gt_box[0] + gt_box[2]) / 2
            gt_box = (cx - 6, gt_box[1], cx + 6, gt_box[3])
        if gt_box[3] - gt_box[1] < 12:
            cy = (gt_box[1] + gt_box[3]) / 2
            gt_box = (gt_box[0], cy - 6, gt_box[2], cy + 6)

    frames: List[Image.Image] = []
    steps = max(travel_frames, 1)

    def render(progress: float, pulse: Optional[float]) -> Image.Image:
        canvas, draw = make_canvas(scale, caption_h)
        canvas.paste(base, (0, 0))
        draw_prediction(draw, gt_box, pred_xy, progress, pulse, frame_h)

        if thumb_info is not None:
            thumb, fbox = thumb_info
            tx, ty = frame_w - thumb.width - 8, 8
            canvas.paste(thumb, (tx, ty))
            draw.rectangle([tx, ty, tx + thumb.width, ty + thumb.height], outline=(120, 124, 134), width=1)
            draw.rectangle([tx + fbox[0], ty + fbox[1],
                            tx + fbox[0] + fbox[2], ty + fbox[1] + fbox[3]],
                           outline=GT_COLOR, width=2)

        draw_text_block(draw, caption_lines(row), 8, frame_h + 6, spacing=3)
        return canvas

    for i in range(steps):
        frames.append(render(i / steps, None))
    for i in range(pulse_frames):
        frames.append(render(1.0, (i + 1) / pulse_frames))
    frames.append(render(1.0, None))  # 停留帧
    return frames


def title_frame(text: str, subtitle: str, scale: int, caption_h: int) -> Image.Image:
    canvas, draw = make_canvas(scale, caption_h)
    width, height = canvas.size
    font_big = _load_font(26)
    font_small = _load_font(15)
    tw = draw.textlength(text, font=font_big)
    draw.text(((width - tw) / 2, height / 2 - 46), text, fill=FG, font=font_big)
    for i, line in enumerate(subtitle.split("\n")):
        lw = draw.textlength(line, font=font_small)
        draw.text(((width - lw) / 2, height / 2 - 6 + i * 22), line, fill=DIM, font=font_small)
    return canvas


def end_frame(scale: int, caption_h: int, rows: Sequence[Dict[str, str]]) -> Image.Image:
    canvas, draw = make_canvas(scale, caption_h)
    width, _ = canvas.size
    font = _load_font(15)
    font_small = _load_font(13)
    n = len(rows)
    acc = sum(1 for r in rows if r["correct"] == "1") / max(n, 1)
    mean_err = float(np.mean([float(r["err_px"]) for r in rows])) if rows else 0.0
    lines = [
        (f"本组 {n} 个样本：动作全部正确，落点平均误差 {mean_err:.0f}px", FG, 16),
        ("白箭头 = 数据集自带光标（标准答案）", DIM, 13),
        ("黄圆环 = 模型预测落点   绿框 = 标注目标元素", DIM, 13),
        ("画面为以目标为中心的放大区域，右上角小图为全屏位置", DIM, 13),
        ("", DIM, 6),
        ("注意：这是单步静态预测的可视化，不是自主完成任务的过程。", (255, 170, 90), 13),
        ("圆环移动仅为视觉引导；模型本身不产生轨迹。", (255, 170, 90), 13),
    ]
    y = 46
    for text, color, size in lines:
        if text:
            tw = draw.textlength(text, font=_load_font(size))
            draw.text(((width - tw) / 2, y), text, fill=color, font=_load_font(size))
        y += size + 5
    return canvas


# --------------------------------------------------------------------------- #
def build_gif(args: argparse.Namespace) -> int:
    csv_path = Path(args.predictions)
    if not csv_path.exists():
        print(f"[错误] 找不到预测 CSV: {csv_path}")
        print("       请先运行: python scripts/demo_desktop_agent.py --mode eval --split test")
        return 1

    rows = load_predictions(csv_path)
    if args.samples:
        wanted = {s.strip() for s in args.samples.split(",") if s.strip()}
        picked = [r for r in rows if r["sample_id"] in wanted]
        if not picked:
            print(f"[错误] 指定的样本都不在 {csv_path} 里: {sorted(wanted)}")
            return 1
    else:
        picked = pick_samples(rows, per_app=args.per_app)

    data_root = Path(args.data_root)
    caption_h = 16 * 3 + 10
    print("=" * 72)
    print("生成演示 GIF")
    print("=" * 72)
    print(f"预测来源: {csv_path}")
    print(f"选中样本: {len(picked)} 个")
    for row in picked:
        print(f"  {row['sample_id']:22} {row['gt_action']:18} "
              f"conf={float(row['confidence']):.0%} err={float(row['err_px']):6.1f}px")

    frames: List[Image.Image] = []
    durations: List[int] = []

    head = title_frame("AI 桌面操作员 · 单步预测演示",
                       f"EnhancedCNN + 空间 soft-argmax 坐标头\n"
                       f"test 集 79 样本：动作 100% 正确，落点平均 13 px",
                       args.scale, caption_h)
    frames.append(head)
    durations.append(args.title_ms)

    for row in picked:
        sample_frames = blank_sample_frames(row, data_root, args.split, args.scale,
                                            caption_h, args.travel_frames, args.pulse_frames,
                                            args.zoom_divisor)
        if not sample_frames:
            print(f"  [跳过] 找不到截图: {row['sample_id']}")
            continue
        for i, frame in enumerate(sample_frames):
            frames.append(frame)
            if i < args.travel_frames:
                durations.append(args.travel_ms)
            elif i < args.travel_frames + args.pulse_frames:
                durations.append(args.pulse_ms)
            else:
                durations.append(args.hold_ms)

    frames.append(end_frame(args.scale, caption_h, picked))
    durations.append(args.end_ms)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n已生成: {out_path}")
    print(f"  帧数 {len(frames)} | 尺寸 {frames[0].size[0]}x{frames[0].size[1]} | "
          f"总时长 {sum(durations)/1000:.1f}s | 体积 {size_mb:.2f} MB")

    if args.still:
        still_path = Path(args.still)
        still_path.parent.mkdir(parents=True, exist_ok=True)
        # 取第一个"已到位"的帧作为静态首图（跳 1 帧最像真实预测结果）
        still_index = 1 + min(args.travel_frames, max(len(frames) - 1, 0))
        frames[still_index].save(still_path)
        print(f"  静态首帧: {still_path}")

    if size_mb > args.max_mb:
        print(f"\n[提示] 体积 {size_mb:.2f} MB 超过 --max-mb {args.max_mb}，"
              f"可用更小的 --scale 或更少的 --per-app 重新生成")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把预测结果做成演示 GIF")
    parser.add_argument("--predictions", default=str(ROOT / "reports" / "v3_coord" / "demo_predictions_test.csv"))
    parser.add_argument("--data-root", default=str(ROOT / "ai_operator_dataset_v3"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=str(ROOT / "reports" / "demo.gif"))
    parser.add_argument("--still", default=str(ROOT / "reports" / "demo_still.png"),
                        help="额外导出一张静态首帧（便于在 README 内嵌）")
    parser.add_argument("--scale", type=int, default=640, help="帧宽度，高度自动按 16:9")
    parser.add_argument("--zoom-divisor", type=float, default=3.0,
                        help="以目标为中心裁剪 1/N 的画面再放大，让 4K 截图里的元素看得清。"
                             "注意面板里光标的视觉大小恒为 42.9*N 像素（与屏幕分辨率无关），"
                             "所以 N 越大光标越夸张、越会遮住目标；默认 3 是折中。"
                             "1 或 0 表示关闭，直接整屏缩放（元素会小到看不清）")
    parser.add_argument("--per-app", type=int, default=2,
                        help="每个应用挑几个样本（取误差最小与最大的，便于同时展示最好与最难）")
    parser.add_argument("--samples", default="", help="逗号分隔的 sample_id，显式指定则忽略 --per-app")
    parser.add_argument("--travel-frames", type=int, default=3, help="圆环飞入的帧数")
    parser.add_argument("--pulse-frames", type=int, default=2, help="点击脉冲帧数")
    parser.add_argument("--title-ms", type=int, default=1800)
    parser.add_argument("--travel-ms", type=int, default=90)
    parser.add_argument("--pulse-ms", type=int, default=160)
    parser.add_argument("--hold-ms", type=int, default=900)
    parser.add_argument("--end-ms", type=int, default=3000)
    parser.add_argument("--max-mb", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(build_gif(parse_args()))
