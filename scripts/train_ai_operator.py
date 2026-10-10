#!/usr/bin/env python3
"""
AI 桌面操作员 —— 真实数据集训练脚本
====================================

这是本项目**唯一**能够真正利用 ``ai_operator_dataset`` 的训练入口：它读取截图 +
JSON 标注，把 ``operation.action`` 当作动作分类标签、把 ``operation.target_bounds``
的中心点当作"点击位置"坐标回归标签，对 :class:`~ai_operator.models.cnn_model.EnhancedCNN`
做端到端微调。

用法::

    # 完整训练（CPU 上 i7-11800H 约 1 分钟/轮）
    .venv\\Scripts\\python.exe scripts/train_ai_operator.py --epochs 20

    # 快速冒烟测试（每类只取少量样本，2 轮）
    .venv\\Scripts\\python.exe scripts/train_ai_operator.py --epochs 2 --max-samples 60 --out-dir models/smoke

    # 小图加速
    .venv\\Scripts\\python.exe scripts/train_ai_operator.py --input-size 128 --epochs 25

产出（默认 ``models/desktop_run/``）::

    best_model.pth        验证集准确率最高的权重（state_dict）
    last_model.pth        最后一轮的权重
    label_map.json        动作名称 -> 类别索引
    training_history.json 每轮 loss / 准确率 / 坐标误差
    metrics.json          最优轮的完整指标与基线对照
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows 控制台默认使用 GBK，会把中文日志写成乱码；统一改成 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - 老版本解释器没有 reconfigure
    pass

from ai_operator.data.desktop_dataset import (  # noqa: E402
    DesktopSample,
    DesktopScreenshotDataset,
    action_zh,
    build_label_map,
    dataset_summary,
    load_label_map,
    save_label_map,
    scan_split,
)
from ai_operator.models.cnn_model import EnhancedCNN, decode_points  # noqa: E402


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(dataset: DesktopScreenshotDataset, batch_size: int, shuffle: bool,
                num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def majority_baseline(samples: List[DesktopSample]) -> Tuple[str, float]:
    """多数类基线，用于判断模型是否真的学到了东西。"""
    if not samples:
        return "-", 0.0
    counts = Counter(s.action for s in samples)
    action, count = counts.most_common(1)[0]
    return action, count / len(samples)


# --------------------------------------------------------------------------- #
# 训练 / 评估
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion_ce, criterion_coord, optimizer, device,
              coord_weight: float, train: bool) -> Dict[str, float]:
    model.train(train)

    total_loss = total_ce = total_coord = 0.0
    correct = top3_correct = 0
    coord_err_sum = 0.0
    coord_err_px_sum = 0.0
    seen = 0

    for batch in loader:
        images = batch["image"].to(device)
        actions = batch["action"].to(device)
        points = batch["point"].to(device)
        # 原图 (W, H)，用于把归一化坐标误差换算回像素
        sizes = batch["image_size"].to(device).float()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(images, return_segmentation=False)
            logits = outputs["actions"]
            pred_points = decode_points(model, outputs["mouse_positions"])

            loss_ce = criterion_ce(logits, actions)
            loss_coord = criterion_coord(pred_points, points)
            loss = loss_ce + coord_weight * loss_coord

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = actions.size(0)
        seen += batch_size
        total_loss += float(loss.item()) * batch_size
        total_ce += float(loss_ce.item()) * batch_size
        total_coord += float(loss_coord.item()) * batch_size

        pred_cls = logits.argmax(dim=1)
        correct += int((pred_cls == actions).sum().item())
        top3 = logits.topk(min(3, logits.size(1)), dim=1).indices
        top3_correct += int((top3 == actions.unsqueeze(1)).any(dim=1).sum().item())

        # 坐标误差：归一化空间 与 原图像素空间
        err = (pred_points - points).abs()
        coord_err_sum += float(err.sum().item())
        width = sizes[:, 0].clamp(min=1.0)
        height = sizes[:, 1].clamp(min=1.0)
        err_px = torch.stack([err[:, 0] * width, err[:, 1] * height], dim=1)
        coord_err_px_sum += float(err_px.sum().item())

    n_points = max(seen * 2, 1)
    return {
        "loss": total_loss / max(seen, 1),
        "loss_action": total_ce / max(seen, 1),
        "loss_coord": total_coord / max(seen, 1),
        "acc": correct / max(seen, 1),
        "top3_acc": top3_correct / max(seen, 1),
        "coord_mae": coord_err_sum / n_points,
        "coord_mae_px": coord_err_px_sum / n_points,
    }


def evaluate_per_class(model, dataset: DesktopScreenshotDataset, device,
                       batch_size: int) -> Dict[str, Dict[str, float]]:
    """按动作类别统计验证准确率，用于定位模型究竟学到了哪些操作。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    predictions: Dict[str, int] = {}
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device), return_segmentation=False)["actions"]
            preds = logits.argmax(dim=1).cpu().tolist()
            for sample_id, pred in zip(batch["sample_id"], preds):
                predictions[sample_id] = int(pred)

    result: Dict[str, Dict[str, float]] = {}
    for sample in dataset.samples:
        pred = predictions.get(sample.sample_id)
        if pred is None:
            continue
        idx = dataset.label_map.get(sample.action, -1)
        bucket = result.setdefault(sample.action, {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(pred == idx)
    for bucket in result.values():
        bucket["acc"] = bucket["correct"] / max(bucket["total"], 1)
    return result


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在真实桌面数据集上训练 EnhancedCNN")
    parser.add_argument("--data-root", default=str(ROOT / "ai_operator_dataset"),
                        help="数据集根目录")
    parser.add_argument("--out-dir", default=str(ROOT / "models" / "desktop_run"),
                        help="模型与指标输出目录")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=224, help="输入分辨率（正方形）")
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="ResNet 主干学习率")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="预测头学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--coord-weight", type=float, default=1.0, help="坐标回归损失权重")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 进程数（图像已预载，0 即可）")
    parser.add_argument("--max-samples", type=int, default=0, help=">0 时每个 split 只取这么多样本（冒烟测试用）")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="不加载 ImageNet 预训练权重")
    parser.add_argument("--draw-cursor", action="store_true",
                        help="在目标点绘制光标指针。数据集本身没有光标像素，"
                             "不开这个开关时坐标回归任务是无解的（点击位置在画面上不可见）")
    parser.add_argument("--spatial-coord", action="store_true",
                        help="使用带空间信息的 soft-argmax 坐标头（强烈建议开启）。"
                             "不开启时坐标头只能看到全局平均池化后的特征，无法定位，"
                             "坐标误差会卡在几百像素")
    parser.add_argument("--patience", type=int, default=0, help=">0 时监控指标连续多少轮不提升就早停")
    parser.add_argument("--monitor", choices=["acc", "coord"], default="acc",
                        help="模型选优与早停依据：acc=验证准确率（默认，向后兼容）；"
                             "coord=验证坐标MAE。准确率会饱和在 100% 而坐标往往还在下降，"
                             "关心定位精度时用 coord")
    parser.add_argument("--augment-shift", type=int, default=0,
                        help="平移增强的最大像素数（输入空间）。图像与目标点标签同步平移，"
                             "属保标签增强，用于提升对布局偏移的鲁棒性。0=关闭")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="监控指标至少改善这么多才算提升，用于滤掉噪声触发的'提升'")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("AI 桌面操作员 - 真实数据集训练")
    print("=" * 72)
    print(f"设备: {device} | torch {torch.__version__} | 线程 {torch.get_num_threads()}")
    print(f"数据根目录: {args.data_root}")

    train_samples = scan_split(args.data_root, "train")
    val_samples = scan_split(args.data_root, "val")
    if not train_samples:
        print(f"[错误] 在 {args.data_root}/train 下没有找到有效样本")
        return 1
    if not val_samples:
        print("[警告] 没有验证集，将从训练集切分 15%")
        cut = int(len(train_samples) * 0.85)
        train_samples, val_samples = train_samples[:cut], train_samples[cut:]

    if args.max_samples > 0:
        # 等间隔抽样，保证冒烟测试也能覆盖多类界面（而不是只取最前面几个 basic_*）
        def subsample(items: List[DesktopSample], k: int) -> List[DesktopSample]:
            k = min(k, len(items))
            if k <= 0:
                return []
            stride = max(len(items) // k, 1)
            return items[::stride][:k]

        train_samples = subsample(train_samples, args.max_samples)
        val_samples = subsample(val_samples, max(args.max_samples // 4, 1))
        print(f"[冒烟模式] 训练 {len(train_samples)} / 验证 {len(val_samples)} 样本")

    label_map = build_label_map(train_samples)
    # 保证验证集里出现的类别也在映射中（否则记为 -1，不参与统计）
    for action in sorted({s.action for s in val_samples}):
        label_map.setdefault(action, len(label_map))
    num_actions = len(label_map)

    summary = dataset_summary(train_samples)
    print(f"训练集: {summary['num_samples']} 样本 / {num_actions} 个动作类别")
    print("类别分布:")
    for action, count in summary["action_counts"].items():
        print(f"  {action:<20} {count:>4}  ({action_zh(action)})")
    baseline_action, baseline_acc = majority_baseline(val_samples)
    print(f"验证集: {len(val_samples)} 样本 | 多数类基线: {baseline_action} -> {baseline_acc:.1%}")

    print("\n构建数据集（首次会预加载并缩放截图）...")
    if args.draw_cursor:
        print("已启用光标注入：目标点会画上箭头光标，坐标回归因此可解")
    t0 = time.time()
    train_ds = DesktopScreenshotDataset(train_samples, label_map, args.input_size, train=True,
                                        preload=True, draw_cursor=args.draw_cursor,
                                        augment_shift=args.augment_shift)
    val_ds = DesktopScreenshotDataset(val_samples, label_map, args.input_size, train=False,
                                      preload=True, draw_cursor=args.draw_cursor)
    print(f"数据集就绪，用时 {time.time() - t0:.1f}s")

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)

    print(f"\n初始化 EnhancedCNN(num_actions={num_actions}, pretrained={not args.no_pretrained}, "
          f"spatial_coord={args.spatial_coord}) ...")
    model = EnhancedCNN(num_actions, screen_size=(args.input_size, args.input_size),
                        pretrained=not args.no_pretrained,
                        spatial_coord=args.spatial_coord).to(device)

    # 主干与预测头使用不同学习率：预训练特征只需微调，新初始化的头需要更快收敛
    backbone_params = list(model.resnet.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_coord = nn.SmoothL1Loss(beta=0.05)

    history: Dict[str, List[float]] = {
        "epoch": [], "train_loss": [], "train_acc": [], "train_top3": [],
        "val_loss": [], "val_acc": [], "val_top3": [],
        "val_coord_mae": [], "val_coord_mae_px": [], "lr": [], "seconds": [],
    }

    # 选优准则。默认沿用"验证准确率"以保持行为兼容，但准确率会在 100% 处饱和，
    # 而坐标误差往往还在继续下降（实测 v3：第 3 轮 acc 已达 100%、坐标 26px，
    # 第 7 轮坐标降到 11px 却被早停丢弃，白丢 15px 定位精度）。
    # 关心定位时用 --monitor coord 让选优与早停都跟随坐标误差。
    monitor = args.monitor
    higher_is_better = monitor == "acc"

    def metric_of(metrics: Dict[str, float]) -> float:
        return metrics["acc"] if monitor == "acc" else metrics["coord_mae_px"]

    def is_better(value: float, reference: float) -> bool:
        if higher_is_better:
            return value > reference + args.min_delta
        return value < reference - args.min_delta

    best_value = float("-inf") if higher_is_better else float("inf")
    best = {"epoch": -1, "value": None, "metrics": {}}
    best_by = {"acc": {"epoch": -1, "value": float("-inf")},
               "coord": {"epoch": -1, "value": float("inf")}}
    epochs_without_improvement = 0
    monitor_label = "验证准确率" if monitor == "acc" else "验证坐标MAE(px)"

    print(f"\n模型选优与早停均依据: {monitor_label}"
          f"（min-delta={args.min_delta}）")
    print("-" * 72)
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = run_epoch(model, train_loader, criterion_ce, criterion_coord,
                                  optimizer, device, args.coord_weight, train=True)
        val_metrics = run_epoch(model, val_loader, criterion_ce, criterion_coord,
                                optimizer, device, args.coord_weight, train=False)
        elapsed = time.time() - started
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["train_top3"].append(train_metrics["top3_acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_top3"].append(val_metrics["top3_acc"])
        history["val_coord_mae"].append(val_metrics["coord_mae"])
        history["val_coord_mae_px"].append(val_metrics["coord_mae_px"])
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["seconds"].append(elapsed)

        # 两个准则各自的最优轮次都记录下来，便于事后对比
        if val_metrics["acc"] > best_by["acc"]["value"]:
            best_by["acc"] = {"epoch": epoch, "value": val_metrics["acc"]}
        if val_metrics["coord_mae_px"] < best_by["coord"]["value"]:
            best_by["coord"] = {"epoch": epoch, "value": val_metrics["coord_mae_px"]}

        marker = ""
        if is_better(metric_of(val_metrics), best_value):
            best_value = metric_of(val_metrics)
            best = {"epoch": epoch, "value": best_value, "metrics": val_metrics}
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            epochs_without_improvement = 0
            marker = "  <- 保存 best_model.pth"
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch:>3}/{args.epochs} | "
              f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.1%} | "
              f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.1%} "
              f"top3 {val_metrics['top3_acc']:.1%} | "
              f"坐标MAE {val_metrics['coord_mae_px']:.0f}px | {elapsed:.1f}s{marker}", flush=True)

        torch.save(model.state_dict(), out_dir / "last_model.pth")

        if args.patience and epochs_without_improvement >= args.patience:
            print(f"{monitor_label}连续 {args.patience} 轮未提升，提前结束训练")
            break

    total_time = sum(history["seconds"])
    print("-" * 72)
    if monitor == "acc":
        print(f"训练结束：按准确率选优 -> 第 {best['epoch']} 轮，"
              f"验证准确率 {best['value']:.1%}（多数类基线 {baseline_acc:.1%}），累计 {total_time:.1f}s")
    else:
        print(f"训练结束：按坐标误差选优 -> 第 {best['epoch']} 轮，"
              f"验证坐标MAE {best['value']:.0f}px（多数类基线 {baseline_acc:.1%}），累计 {total_time:.1f}s")
    print(f"两组准则各自的最优: 准确率 {best_by['acc']['value']:.1%} @ 第 {best_by['acc']['epoch']} 轮 | "
          f"坐标 {best_by['coord']['value']:.0f}px @ 第 {best_by['coord']['epoch']} 轮")

    # 载回被保存的最优权重再统计逐类表现。
    # 早停会让"最后一轮"与"最优轮"不是同一组权重，若直接用内存里的 model
    # 就会得到一份与 best_model.pth 不符的逐类报告（metrics.json 里会前后矛盾）。
    best_path = out_dir / "best_model.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"\n已载回最优权重用于逐类统计: 第 {best['epoch']} 轮")

    per_class = evaluate_per_class(model, val_ds, device, args.batch_size)
    sorted_classes = sorted(per_class.items(), key=lambda kv: -kv[1]["total"])
    print("\n按类别验证准确率（样本数降序）:")
    for action, bucket in sorted_classes:
        print(f"  {action:<20} {bucket['correct']:>3}/{bucket['total']:<3} = {bucket['acc']:>5.1%}  ({action_zh(action)})")

    save_label_map(label_map, out_dir / "label_map.json")
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    metrics = {
        "best_epoch": best["epoch"],
        "monitor": monitor,
        "min_delta": args.min_delta,
        # selected_* 是被保存到 best_model.pth 的那一轮；best_acc_*/best_coord_*
        # 是两个准则各自的最优轮次，便于判断"是否因为选优准则而丢掉了精度"
        "selected_val_acc": best["metrics"].get("acc"),
        "selected_val_top3_acc": best["metrics"].get("top3_acc"),
        "selected_val_coord_mae": best["metrics"].get("coord_mae"),
        "selected_val_coord_mae_px": best["metrics"].get("coord_mae_px"),
        "best_acc_epoch": best_by["acc"]["epoch"],
        "best_acc": best_by["acc"]["value"],
        "best_coord_epoch": best_by["coord"]["epoch"],
        "best_coord_mae_px": best_by["coord"]["value"],
        "majority_baseline_action": baseline_action,
        "majority_baseline_acc": baseline_acc,
        "random_baseline_acc": 1.0 / num_actions,
        "num_actions": num_actions,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "input_size": args.input_size,
        "batch_size": args.batch_size,
        "draw_cursor": bool(args.draw_cursor),
        "augment_shift": int(args.augment_shift),
        "spatial_coord": bool(args.spatial_coord),
        "epochs_run": len(history["epoch"]),
        "total_seconds": total_time,
        "device": str(device),
        "per_class_val": {k: v for k, v in sorted_classes},
        "label_map": label_map,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n最佳模型: {out_dir / 'best_model.pth'}")
    print(f"指标汇总: {out_dir / 'metrics.json'}")
    print(f"下一步演示: python scripts/demo_desktop_agent.py --model {out_dir / 'best_model.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
