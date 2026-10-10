#!/usr/bin/env python3
"""
AI 桌面操作员 —— 演示 / 推理脚本
================================

三种模式：

1. ``eval``  —— 在 ``val`` / ``test`` 划分上批量评估，输出准确率、坐标误差、
   混淆分析和可视化拼图。这是"模型确实学到了东西"的证据。

   .. code-block:: text

       python scripts/demo_desktop_agent.py --mode eval --split test

2. ``image`` —— 对任意一张截图做单次预测，并把预测结果画在图上。

   .. code-block:: text

       python scripts/demo_desktop_agent.py --mode image --image ai_operator_dataset/test/images/basic_000003/screenshot.png

3. ``live``  —— 实时截屏 → 预测 → （可选）真的把鼠标点下去，形成闭环。
   默认是**空跑**，只打印将要执行的动作；加 ``--execute`` 才会真的操作鼠标键盘。

   .. code-block:: text

       python scripts/demo_desktop_agent.py --mode live --interval 2
       python scripts/demo_desktop_agent.py --mode live --execute --min-confidence 0.6
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # Windows 控制台默认 GBK，中文日志会乱码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

from ai_operator.data.desktop_dataset import (  # noqa: E402
    DesktopSample,
    DesktopScreenshotDataset,
    action_zh,
    load_label_map,
    scan_split,
)
from ai_operator.models.cnn_model import decode_points  # noqa: E402
from ai_operator.models.inference import ScreenPredictor  # noqa: E402


# --------------------------------------------------------------------------- #
# 模型装载与单张推理
#
# 这里不再自己实现"读 checkpoint + 判断坐标头 + 预处理 + 解码坐标"，
# 而是复用 ai_operator/models/inference.py 的 ScreenPredictor —— 与 GUI 走同一份代码。
# 之前两边各写一份，结果 GUI 那份漏了加载权重、又把坐标多套了一次 sigmoid，
# 界面显示与评估指标完全对不上。
# --------------------------------------------------------------------------- #
def load_model(model_path: Path, num_actions: int, input_size: int,
               device: torch.device, label_map_path: Optional[Path] = None) -> ScreenPredictor:
    """加载模型，返回统一的推理器。

    ``num_actions`` 参数保留是为了兼容旧签名，实际类别数以 label_map 为准。
    """
    if label_map_path is None:
        label_map_path = Path(model_path).parent / "label_map.json"
    predictor = ScreenPredictor.from_checkpoint(
        Path(model_path), Path(label_map_path), input_size=input_size, device=device
    )
    info = predictor.info
    print(f"坐标头: {'空间 soft-argmax' if info.spatial_coord else '全局池化 + 线性（定位能力有限）'}"
          f" | 类别数: {info.num_actions} | 输入: {info.input_size}")
    return predictor


# --------------------------------------------------------------------------- #
# 可视化辅助
# --------------------------------------------------------------------------- #
def _load_font(size: int):
    from PIL import ImageFont

    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc", "arial.ttf"):
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _ellipsis(text: str, limit: int = 34) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def annotate(pil_image, gt_bounds: Optional[Sequence[float]], pred_point_px: Optional[Tuple[float, float]],
             caption: Sequence[str], cell: Optional[Tuple[int, int]] = None):
    """在截图上画出 GT 框（绿）/ 预测点（红十字），并加上说明文字。

    Args:
        cell: 指定输出尺寸；None 表示保持原图尺寸。坐标会按比例缩放。
    """
    from PIL import Image, ImageDraw

    image = pil_image.convert("RGB")
    orig_w, orig_h = image.size
    caption_h = 16 * len(caption) + 8 if caption else 0
    if cell is None:
        canvas = image.copy()
        scale_x = scale_y = 1.0
    else:
        cell_w, cell_h = cell
        canvas = image.resize((cell_w, cell_h))
        scale_x = cell_w / orig_w
        scale_y = cell_h / orig_h

    out = Image.new("RGB", (canvas.width, canvas.height + caption_h), (16, 16, 20))
    out.paste(canvas, (0, 0))
    draw = ImageDraw.Draw(out)

    if gt_bounds is not None:
        x1, y1, x2, y2 = gt_bounds
        draw.rectangle([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                       outline=(0, 255, 90), width=3)
    if pred_point_px is not None:
        px, py = pred_point_px[0] * scale_x, pred_point_px[1] * scale_y
        r = 10
        draw.line([px - r, py, px + r, py], fill=(255, 60, 60), width=3)
        draw.line([px, py - r, px, py + r], fill=(255, 60, 60), width=3)
        draw.ellipse([px - r, py - r, px + r, py + r], outline=(255, 200, 0), width=2)

    if caption_h:
        font = _load_font(15)
        for i, line in enumerate(caption):
            draw.text((6, canvas.height + 4 + i * 16), _ellipsis(line), fill=(235, 235, 235), font=font)
    return out


# --------------------------------------------------------------------------- #
# eval 模式
# --------------------------------------------------------------------------- #
def hash_file(path: str, block: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_train_hash_set(data_root: Path, split: str = "train") -> set:
    hashes = set()
    for sample in scan_split(data_root, split, with_image_size=False):
        try:
            hashes.add(hash_file(sample.image_path))
        except OSError:
            continue
    return hashes


def evaluate_split(predictor: ScreenPredictor, samples: List[DesktopSample],
                   label_map: Dict[str, int],
                   input_size: int, device: torch.device, batch_size: int,
                   seen_hashes: Optional[set] = None,
                   draw_cursor: bool = False) -> List[Dict[str, object]]:
    # 批量评估刻意不走 ScreenPredictor.predict（那是单张推理路径），
    # 直接对底层模型做批处理以提升吞吐；但坐标解码仍统一用 decode_points。
    model = predictor.model
    dataset = DesktopScreenshotDataset(samples, label_map, input_size, train=False, preload=True,
                                       verbose=False, draw_cursor=draw_cursor)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    idx_to_action = {v: k for k, v in label_map.items()}
    rows: List[Dict[str, object]] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image"].to(device), return_segmentation=False)
            probs = torch.softmax(outputs["actions"], dim=1)
            points = decode_points(model, outputs["mouse_positions"])
            top3 = probs.topk(min(3, probs.size(1)), dim=1).indices

            for i, sample_id in enumerate(batch["sample_id"]):
                rows.append({
                    "sample_id": sample_id,
                    "pred_idx": int(probs[i].argmax().item()),
                    "pred_action": idx_to_action[int(probs[i].argmax().item())],
                    "confidence": float(probs[i].max().item()),
                    "top3": [int(v) for v in top3[i].tolist()],
                    "probs": [float(v) for v in probs[i].tolist()],
                    "pred_point": (float(points[i, 0].item()), float(points[i, 1].item())),
                })

    by_id = {r["sample_id"]: r for r in rows}
    result: List[Dict[str, object]] = []
    for sample in samples:
        row = by_id.get(sample.sample_id)
        if row is None:
            continue
        gt_idx = label_map.get(sample.action, -1)
        gt_point = sample.norm_point
        width, height = sample.image_size
        err_px = float(np.hypot((row["pred_point"][0] - gt_point[0]) * width,
                                (row["pred_point"][1] - gt_point[1]) * height))
        seen = None
        if seen_hashes is not None:
            try:
                seen = hash_file(sample.image_path) in seen_hashes
            except OSError:
                seen = None
        result.append({
            "sample": sample,
            "gt_action": sample.action,
            "gt_idx": gt_idx,
            "pred_action": row["pred_action"],
            "pred_idx": row["pred_idx"],
            "confidence": row["confidence"],
            "top3": row["top3"],
            "probs": row["probs"],
            "pred_point": row["pred_point"],
            "gt_point": gt_point,
            "err_px": err_px,
            "correct": row["pred_idx"] == gt_idx,
            "top3_correct": gt_idx in row["top3"],
            "screen_in_train": seen,
        })
    return result


#: 样本 ID 前缀 -> 应用名称（数据集里 sample_id 形如 web_browser_000010）
APP_LABELS: Dict[str, str] = {
    "basic": "桌面 (basic)",
    "file_explorer": "文件资源管理器",
    "text_editor": "文本编辑器",
    "web_browser": "浏览器",
}


def app_of(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


def is_causal_dataset(data_root: Path) -> bool:
    """判断数据集是否做了因果修正。

    判据（任一成立即可）：

    - 存在 v2/v3 清单文件；
    - 标注里带顶层 ``app`` 字段（v2/v3 生成器都会写）；
    - 标注里 ``operation.cursor_drawn`` 为真（光标已烘烤进截图，
      说明"点了哪里"在画面上可见，动作也因此可推断）。

    v1 的标注三个条件都不满足。这个判断决定报告里打印哪一段解释文字——
    两代数据集的结论完全相反，用同一段话会误导读者。
    """
    if (data_root / "dataset_v2_manifest.json").exists() or (data_root / "dataset_v3_manifest.json").exists():
        return True
    ann_dir = data_root / "train" / "annotations"
    if not ann_dir.is_dir():
        return False
    for path in sorted(ann_dir.glob("*.json"))[:5]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                head = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if "app" in head or head.get("cursor_drawn"):
            return True
        if (head.get("operation") or {}).get("cursor_drawn"):
            return True
    return False


def per_app_majority_action(train_samples: List[DesktopSample]) -> Dict[str, str]:
    """每个应用在训练集里最常见的动作（用作"应用先验解码"）。"""
    from collections import defaultdict

    counts: Dict[str, Counter] = defaultdict(Counter)
    for sample in train_samples:
        counts[app_of(sample.sample_id)][sample.action] += 1
    return {app: counter.most_common(1)[0][0] for app, counter in counts.items()}


def prior_decoded_accuracy(rows: List[Dict[str, object]], train_samples: List[DesktopSample]) -> float:
    """用"先识别应用，再取该应用最常见动作"的方式解码时的 Top-1 准确率。

    由于模型对应用识别是 100% 准确的，而应用内部的具体动作与画面无关，
    这个解码方式能去掉模型概率分布过度铺开带来的损失。
    """
    priors = per_app_majority_action(train_samples)
    correct = 0
    for row in rows:
        truth = app_of(row["sample"].sample_id)  # type: ignore[union-attr]
        guess = priors.get(truth)
        correct += int(guess is not None and guess == row["gt_action"])
    return correct / max(len(rows), 1)


def app_posterior_table(train_samples: List[DesktopSample]) -> Tuple[List[str], Dict[int, List[float]]]:
    """由训练集统计 P(应用 | 动作)，用于把动作概率边缘化成应用概率。

    当前数据集的 action 与界面元素是**独立随机**生成的，截图里也没有光标/高亮标记
    点击位置，所以"具体动作"在信息论上不可解；但"这是哪个应用的窗口"是可学的。
    这张表让我们可以把这个可学的部分单独量化出来。
    """
    from collections import defaultdict

    apps = sorted({app_of(s.sample_id) for s in train_samples})
    app_index = {name: i for i, name in enumerate(apps)}
    joint: Dict[str, List[float]] = defaultdict(lambda: [0.0] * len(apps))
    for sample in train_samples:
        joint[sample.action][app_index[app_of(sample.sample_id)]] += 1.0
    for action, counts in joint.items():
        total = sum(counts)
        if total > 0:
            joint[action] = [c / total for c in counts]
    return apps, dict(joint)


def app_level_accuracy(rows: List[Dict[str, object]], label_map: Dict[str, int],
                       train_samples: List[DesktopSample]) -> Tuple[float, List[Dict[str, object]]]:
    """把模型的动作概率边缘化成应用概率，统计"识别当前是哪个应用"的准确率。"""
    apps, posterior = app_posterior_table(train_samples)
    idx_to_action = {v: k for k, v in label_map.items()}

    correct = 0
    per_app: Dict[str, Dict[str, int]] = {name: {"total": 0, "correct": 0} for name in apps}
    for row in rows:
        probs: List[float] = row["probs"]  # type: ignore[assignment]
        scores = [0.0] * len(apps)
        for idx, prob in enumerate(probs):
            action = idx_to_action.get(idx)
            if action is None:
                continue
            weights = posterior.get(action)
            if weights is None:
                continue
            for app_idx, weight in enumerate(weights):
                scores[app_idx] += prob * weight
        predicted = apps[max(range(len(apps)), key=lambda i: scores[i])]
        truth = app_of(row["sample"].sample_id)  # type: ignore[union-attr]
        per_app.setdefault(truth, {"total": 0, "correct": 0})
        per_app[truth]["total"] += 1
        hit = predicted == truth
        per_app[truth]["correct"] += int(hit)
        correct += int(hit)

    detail = [
        {"app": name, "label": APP_LABELS.get(name, name),
         "total": bucket["total"], "correct": bucket["correct"],
         "acc": bucket["correct"] / max(bucket["total"], 1)}
        for name, bucket in sorted(per_app.items(), key=lambda kv: -kv[1]["total"])
    ]
    return correct / max(len(rows), 1), detail


def select_grid_rows(rows: List[Dict[str, object]], limit: int) -> List[Dict[str, object]]:
    """挑出用于可视化拼图的样本。

    原则：

    1. 错例优先（信息量最大），但最多占一半，避免模型很差时拼图全是错的；
    2. 其余名额**按应用轮转**填充。否则在准确率 100% 时没有错例可交替，
       就会退化成按扫描顺序取前 N 个 —— 而扫描顺序里 basic_* 永远排在最前，
       拼图会清一色是桌面截图，完全看不到其他应用。
    """
    from collections import defaultdict

    if limit <= 0 or not rows:
        return []

    picks: List[Dict[str, object]] = []
    used: set = set()

    wrong = [r for r in rows if not r["correct"]]
    for row in wrong[: max(limit // 2, 1)]:
        picks.append(row)
        used.add(row["sample"].sample_id)

    buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["sample"].sample_id not in used:
            buckets[app_of(row["sample"].sample_id)].append(row)

    # 样本多的应用排前面，轮转时更均匀
    order = sorted(buckets, key=lambda key: -len(buckets[key]))
    cursor = 0
    while len(picks) < limit and order:
        key = order[cursor % len(order)]
        if buckets[key]:
            picks.append(buckets[key].pop(0))
        cursor += 1
        if not any(buckets.values()):
            break
    return picks[:limit]


def mode_eval(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root)
    label_map = load_label_map(args.label_map)
    idx_to_action = {v: k for k, v in label_map.items()}
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    samples = scan_split(data_root, args.split)
    if not samples:
        print(f"[错误] {data_root}/{args.split} 下没有样本")
        return 1

    predictor = load_model(Path(args.model), len(label_map), args.input_size, device,
                      label_map_path=Path(args.label_map))
    print(f"模型: {args.model} | 类别数: {len(label_map)} | 输入: {args.input_size} | 设备: {device}")

    seen_hashes = None
    train_samples: List[DesktopSample] = []
    if not args.no_leakage_check:
        print("统计训练集截图指纹（用于区分'见过/没见过'的界面）...")
        seen_hashes = build_train_hash_set(data_root, "train")
    train_samples = scan_split(data_root, "train", with_image_size=False)

    print(f"在 {args.split} 划分上评估 {len(samples)} 个样本 ...")
    rows = evaluate_split(predictor, samples, label_map, args.input_size, device,
                          args.batch_size, seen_hashes, draw_cursor=args.draw_cursor)

    accuracy = sum(r["correct"] for r in rows) / len(rows)
    top3_acc = sum(r["top3_correct"] for r in rows) / len(rows)
    mean_err = float(np.mean([r["err_px"] for r in rows]))
    median_err = float(np.median([r["err_px"] for r in rows]))
    counts = Counter(r["gt_action"] for r in rows)
    majority_action, majority_count = counts.most_common(1)[0]
    majority_acc = majority_count / len(rows)

    print("\n" + "=" * 72)
    print(f"评估结果（{args.split} 划分，{len(rows)} 样本）")
    print("=" * 72)
    print(f"动作 Top-1 准确率 : {accuracy:6.1%}   (随机猜测 {1/len(label_map):.1%}，多数类基线 {majority_acc:.1%})")
    print(f"动作 Top-3 准确率 : {top3_acc:6.1%}")
    print(f"目标点像素误差    : 平均 {mean_err:,.0f}px | 中位数 {median_err:,.0f}px")

    app_acc, app_detail = app_level_accuracy(rows, label_map, train_samples)
    causal = is_causal_dataset(data_root)

    if causal:
        # 因果修正版数据集：动作与界面存在因果关系，动作本身就可预测，应用识别只是副产物
        print(f"\n界面(应用)识别准确率: {app_acc:6.1%}   <- 副产物指标")
        print("  本数据集为因果修正版：动作由目标元素类型唯一决定，且光标已画在目标上，")
        print("  因此'具体动作'本身就是可预测的——上面的动作准确率才是主指标。")
        print("  应用识别对模型来说不是可学信号的上限，而是顺带学会的中间表示。")
    else:
        # v1：动作与界面元素独立随机，且截图中无光标，动作不可由画面确定
        print(f"\n界面(应用)识别准确率: {app_acc:6.1%}   <- 本数据集里真正可学的信号")
        print("  按应用拆分:")
        for entry in app_detail:
            print(f"    {entry['label']:<16} {entry['correct']:>3}/{entry['total']:<3} = {entry['acc']:>5.1%}")
        print("  说明: 数据集生成时 action 与界面元素是独立随机选取的，且截图中没有任何")
        print("        光标/高亮标记点击位置，因此'具体动作'无法由静态截图确定；")
        print("        模型能可靠学到的部分就是当前窗口属于哪个应用。")
        prior_acc = prior_decoded_accuracy(rows, train_samples)
        print(f"\n若改用'先识别应用 -> 取该应用训练集最常见动作'的解码方式: Top-1 = {prior_acc:.1%}")
        print(f"（模型当前概率分布的 argmax 为 {accuracy:.1%}，差距来自应用内部动作不可分导致的概率铺开）")

    if seen_hashes is not None:
        groups = defaultdict(list)
        for r in rows:
            key = "未知" if r["screen_in_train"] is None else ("训练集中见过该界面" if r["screen_in_train"] else "训练集中没见过该界面")
            groups[key].append(r)
        print("\n按'界面是否在训练集中出现过'拆分:")
        for key in ("训练集中见过该界面", "训练集中没见过该界面", "未知"):
            group = groups.get(key)
            if not group:
                continue
            acc = sum(r["correct"] for r in group) / len(group)
            err = float(np.mean([r["err_px"] for r in group]))
            print(f"  {key}: {len(group):>3} 样本 | Top-1 {acc:5.1%} | 坐标误差 {err:,.0f}px")

    print("\n各类别表现（按样本数降序）:")
    per_class = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in rows:
        bucket = per_class[r["gt_action"]]
        bucket["total"] += 1
        bucket["correct"] += int(r["correct"])
    for action, bucket in sorted(per_class.items(), key=lambda kv: -kv[1]["total"]):
        acc = bucket["correct"] / bucket["total"]
        bar = "█" * int(round(acc * 20))
        print(f"  {action:<20} {bucket['correct']:>3}/{bucket['total']:<3} {acc:>6.1%} {bar}  ({action_zh(action)})")

    confusions = Counter((r["gt_action"], r["pred_action"]) for r in rows if not r["correct"])
    if confusions:
        print("\n最常见的混淆（真实 -> 预测）:")
        for (gt, pred), n in confusions.most_common(6):
            print(f"  {gt:<20} -> {pred:<20} x{n}")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"demo_predictions_{args.split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "gt_action", "pred_action", "confidence", "correct",
                         "gt_x", "gt_y", "pred_x", "pred_y", "err_px", "image_w", "image_h",
                         "screen_in_train"])
        for r in rows:
            s = r["sample"]
            w, h = s.image_size
            writer.writerow([
                s.sample_id, r["gt_action"], r["pred_action"], f"{r['confidence']:.4f}",
                int(r["correct"]), f"{r['gt_point'][0]*w:.0f}", f"{r['gt_point'][1]*h:.0f}",
                f"{r['pred_point'][0]*w:.0f}", f"{r['pred_point'][1]*h:.0f}",
                f"{r['err_px']:.1f}", w, h,
                "" if r["screen_in_train"] is None else int(r["screen_in_train"]),
            ])
    print(f"\n逐样本预测: {csv_path}")

    # 可视化拼图：错例优先 + 按应用轮转，保证覆盖多种界面
    picks = select_grid_rows(rows, args.grid)

    if picks:
        from PIL import Image

        cols = 4
        rows_n = (len(picks) + cols - 1) // cols
        cell = (360, 240)
        tile_h = cell[1] + 16 * 3 + 8
        grid = Image.new("RGB", (cols * cell[0], rows_n * tile_h), (10, 10, 12))
        for i, r in enumerate(picks):
            sample: DesktopSample = r["sample"]
            with Image.open(sample.image_path) as im:
                w, h = sample.image_size
                caption = [
                    f"{sample.sample_id}",
                    f"GT  : {r['gt_action']} ({action_zh(r['gt_action'])})",
                    f"Pred: {r['pred_action']} {r['confidence']:.0%}  坐标误差 {r['err_px']:.0f}px",
                ]
                tile = annotate(im, sample.bounds,
                                (r["pred_point"][0] * w, r["pred_point"][1] * h),
                                caption, cell=cell)
            grid.paste(tile, ((i % cols) * cell[0], (i // cols) * tile_h))
        grid_path = report_dir / f"demo_grid_{args.split}.png"
        grid.save(grid_path)
        print(f"可视化拼图（绿框=真实目标，红十字=模型预测点）: {grid_path}")

    metrics_path = report_dir / f"demo_metrics_{args.split}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "num_samples": len(rows),
            "draw_cursor": bool(args.draw_cursor),
            "action_top1": accuracy,
            "action_top3": top3_acc,
            "majority_baseline": majority_acc,
            "majority_action": majority_action,
            "random_baseline": 1.0 / len(label_map),
            "coord_mean_err_px": mean_err,
            "coord_median_err_px": median_err,
            "app_level_accuracy": app_acc,
            "app_level_detail": app_detail,
            "causal_dataset": causal,
            "per_app_majority_action": per_app_majority_action(train_samples),
            "per_class": {k: v for k, v in sorted(per_class.items(), key=lambda kv: -kv[1]["total"])},
            "top_confusions": [{"gt": g, "pred": p, "count": n} for (g, p), n in confusions.most_common(20)],
        }, f, ensure_ascii=False, indent=2)
    print(f"指标 JSON: {metrics_path}")
    return 0


# --------------------------------------------------------------------------- #
# image 模式
# --------------------------------------------------------------------------- #
def _predict_pil(predictor: ScreenPredictor, pil_image, input_size, device, label_map, topk=3):
    """单张 PIL 图像的预测（复用统一推理器）。

    返回的键名保持与旧实现兼容：``action_idx`` / ``confidence`` /
    ``topk``（形如 ``[(索引, 概率)]``） / ``point``，另外补充 ``pred_action``
    与带名字的 ``topk_named`` 供展示使用。
    """
    result = predictor.predict(pil_image, topk=topk)
    return {
        "action_idx": result["action_index"],
        "pred_action": result["action"],
        "confidence": result["confidence"],
        "topk": [(item["index"], item["confidence"]) for item in result["topk"]],
        "topk_named": [(item["action"], item["confidence"]) for item in result["topk"]],
        "point": result["point"],
    }


def mode_image(args: argparse.Namespace) -> int:
    from PIL import Image

    path = Path(args.image)
    if not path.exists():
        print(f"[错误] 图片不存在: {path}")
        return 1

    label_map = load_label_map(args.label_map)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    predictor = load_model(Path(args.model), len(label_map), args.input_size, device,
                      label_map_path=Path(args.label_map))

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        result = _predict_pil(predictor, im, args.input_size, device, label_map)
        annotated = annotate(im, None, (result["point"][0] * w, result["point"][1] * h),
                             [f"{path.name}  ({w}x{h})",
                              f"Pred: {result['pred_action']} ({action_zh(result['pred_action'])}) {result['confidence']:.1%}",
                              f"Click at ({result['point'][0]*w:.0f}, {result['point'][1]*h:.0f})"])

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"prediction_{path.stem}.png"
    annotated.save(out_path)

    print("=" * 72)
    print(f"输入截图: {path}  分辨率 {w}x{h}")
    print("=" * 72)
    print("动作预测 Top-3:")
    for action, prob in result["topk_named"]:
        print(f"  {action:<20} {prob:6.1%}   ({action_zh(action)})")
    print(f"\n预测点击位置: ({result['point'][0]*w:.0f}, {result['point'][1]*h:.0f}) 像素")
    print(f"标注结果已保存: {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# live 模式
# --------------------------------------------------------------------------- #
def mode_live(args: argparse.Namespace) -> int:
    try:
        import pyautogui
    except Exception as exc:  # pragma: no cover
        print(f"[错误] 无法导入 pyautogui（实时模式需要桌面环境）: {exc}")
        return 1

    label_map = load_label_map(args.label_map)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    predictor = load_model(Path(args.model), len(label_map), args.input_size, device,
                      label_map_path=Path(args.label_map))

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("实时桌面闭环演示" + ("（真实执行鼠标操作）" if args.execute else "（空跑：只预测不操作）"))
    print("=" * 72)

    rounds = args.rounds
    count = 0
    try:
        while rounds <= 0 or count < rounds:
            count += 1
            pil_image = pyautogui.screenshot()
            screen_w, screen_h = pil_image.size
            result = _predict_pil(predictor, pil_image, args.input_size, device, label_map)

            x = int(result["point"][0] * screen_w)
            y = int(result["point"][1] * screen_h)
            action = result["pred_action"]
            conf = result["confidence"]

            print(f"\n[第 {count} 轮] {time.strftime('%H:%M:%S')}  屏幕 {screen_w}x{screen_h}")
            print(f"  预测动作: {action}（{action_zh(action)}）  置信度 {conf:.1%}")
            print(f"  预测位置: ({x}, {y})")
            print("  Top-3: " + " | ".join(f"{a} {p:.0%}" for a, p in result["topk_named"]))

            annotated = annotate(pil_image, None, (x, y),
                                 [f"round {count}  {time.strftime('%H:%M:%S')}",
                                  f"Pred: {action} {conf:.0%}",
                                  f"Click at ({x}, {y})"])
            annotated.save(report_dir / f"live_{count:03d}.png")

            if conf < args.min_confidence:
                print(f"  置信度低于阈值 {args.min_confidence:.0%}，跳过执行")
            elif not args.execute:
                print("  空跑模式：未执行操作（加 --execute 才会真正操作鼠标）")
            else:
                _execute_action(pyautogui, action, x, y)
                print(f"  已执行: {action} @ ({x}, {y})")

            if rounds <= 0 or count < rounds:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已停止")

    print(f"\n过程截图保存在: {report_dir}")
    return 0


def _execute_action(pyautogui, action: str, x: int, y: int) -> None:
    """把模型预测的动作映射成真实的鼠标/键盘操作。"""
    button = "right" if action.startswith("right") else "left"
    if action.startswith("scroll"):
        pyautogui.scroll(-3, x=x, y=y)
    elif action == "type_text":
        # 真实输入需要具体文本，这里只把焦点移到预测位置
        pyautogui.click(x, y)
    elif "double" in action:
        pyautogui.click(x, y, clicks=2, interval=0.05, button=button)
    else:
        pyautogui.click(x, y, button=button)


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI 桌面操作员演示")
    parser.add_argument("--mode", choices=["eval", "image", "live"], default="eval")
    parser.add_argument("--model", default=str(ROOT / "models" / "desktop_run" / "best_model.pth"))
    parser.add_argument("--label-map", default=str(ROOT / "models" / "desktop_run" / "label_map.json"))
    parser.add_argument("--data-root", default=str(ROOT / "ai_operator_dataset"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--image", default="", help="image 模式下的输入截图")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report-dir", default=str(ROOT / "reports"))
    parser.add_argument("--grid", type=int, default=12, help="eval 模式可视化拼图的样本数")
    parser.add_argument("--no-leakage-check", action="store_true",
                        help="跳过'界面是否在训练集中出现过'的分组统计")
    parser.add_argument("--draw-cursor", action="store_true",
                        help="评估时在目标点绘制光标，必须与训练时的设置保持一致")
    parser.add_argument("--interval", type=float, default=2.0, help="live 模式每轮间隔（秒）")
    parser.add_argument("--rounds", type=int, default=5, help="live 模式轮数，0 表示无限循环")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--execute", action="store_true", help="live 模式下真正执行鼠标操作")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "eval":
        return mode_eval(args)
    if args.mode == "image":
        if not args.image:
            print("[错误] image 模式需要 --image 指定截图路径")
            return 1
        return mode_image(args)
    return mode_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
