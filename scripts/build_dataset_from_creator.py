#!/usr/bin/env python3
"""
用（已修复因果性的）ProfessionalDatasetCreator 生成可训练数据集
================================================================

这是**旧生成器路径**的可复现入口。它存在的意义是验证
``scripts/../ai_operator/data_generation/dataset_creator.py`` 里那两处修复确实端到端有效：

1. ``generate_app_specific_operation()`` / ``generate_random_operation()``
   的动作改为由目标元素类型唯一决定（原来动作与元素各自独立随机）；
2. ``save_sample()`` 把光标画到目标元素中心，使坐标与动作在画面上可见。

如果你只是想要一份好数据，**优先用 ``scripts/build_dataset_v2.py``**（结果见 README）。
本脚本用于验证旧路径，并附带一份因果性审计报告。

用法::

    python scripts/build_dataset_from_creator.py --out ai_operator_dataset_v3 \\
        --train 300 --val 80 --test 80 --seed 20240101
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

from ai_operator.data_generation.dataset_creator import ProfessionalDatasetCreator  # noqa: E402

APP_TYPES = ("web_browser", "text_editor", "file_explorer")


def _app_of_sample(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


def audit_causality(annotations_dir: Path) -> dict:
    """审计动作与目标元素之间的因果关系。

    同时报告两个粒度：

    - ``global_map``：元素类型 -> 动作（跨应用）。这必须唯一，否则同一个视觉元素
      在不同场景下被贴上不同标签，模型学不到稳定规律。
    - ``app_map``： (应用, 元素类型) -> 动作。这是精确的条件粒度。
    """
    global_map: dict = defaultdict(set)
    app_map: dict = defaultdict(set)
    actions: Counter = Counter()
    elements: Counter = Counter()
    total = 0

    for path in sorted(annotations_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        operation = data.get("operation") or {}
        element_type = operation.get("target_element")
        action = operation.get("action")
        if not element_type or not action:
            continue
        sample_id = data.get("sample_id") or path.stem
        app = data.get("app") or _app_of_sample(sample_id)

        global_map[element_type].add(action)
        app_map[(app, element_type)].add(action)
        actions[action] += 1
        elements[element_type] += 1
        total += 1

    global_violations = {k: sorted(v) for k, v in global_map.items() if len(v) > 1}
    app_violations = {f"{a}/{e}": sorted(v) for (a, e), v in app_map.items() if len(v) > 1}

    return {
        "samples": total,
        "element_types": len(global_map),
        "global_violations": global_violations,
        "app_violations": app_violations,
        "action_counts": dict(sorted(actions.items(), key=lambda kv: -kv[1])),
        "element_counts": dict(sorted(elements.items(), key=lambda kv: -kv[1])),
    }


def build_split(root: Path, split: str, total: int, seed: int) -> dict:
    random.seed(seed)
    creator = ProfessionalDatasetCreator(str(root), draw_cursor=True, layout=split)

    basic_count = total // 2
    per_app = (total - basic_count) // len(APP_TYPES)

    creator.create_basic_dataset(basic_count)
    for app in APP_TYPES:
        creator.create_app_specific_dataset(app, per_app)

    audit = audit_causality(root / split / "annotations")
    return {"split": split, "requested": total, **audit}


def main() -> int:
    parser = argparse.ArgumentParser(description="用修复后的生成器产出可训练数据集")
    parser.add_argument("--out", default=str(ROOT / "ai_operator_dataset_v3"))
    parser.add_argument("--train", type=int, default=300)
    parser.add_argument("--val", type=int, default=80)
    parser.add_argument("--test", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--no-cursor", action="store_true", help="不画光标（对照实验用）")
    args = parser.parse_args()

    out = Path(args.out)
    print("=" * 72)
    print("用 ProfessionalDatasetCreator 生成因果一致 + 带光标的数据集")
    print("=" * 72)
    print(f"输出: {out} | 光标: {'关闭' if args.no_cursor else '开启'}")

    summaries = []
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        print(f"\n生成 {split} ({count} 样本) ...", flush=True)
        summary = build_split(out, split, count, args.seed)
        summaries.append(summary)
        print(f"  实际样本 {summary['samples']} | 元素类型 {summary['element_types']}")
        print(f"  跨应用因果性违反: {len(summary['global_violations'])}"
              + (f" -> {summary['global_violations']}" if summary["global_violations"] else " (OK)"))
        print(f"  应用内因果性违反: {len(summary['app_violations'])}"
              + (f" -> {summary['app_violations']}" if summary["app_violations"] else " (OK)"))
        print(f"  动作分布: {summary['action_counts']}")

    manifest = {
        "name": "ai_operator_dataset_v3",
        "generator": "ProfessionalDatasetCreator (因果性已修复)",
        "created_at": datetime.now().isoformat(),
        "seed": args.seed,
        "cursor_drawn": not args.no_cursor,
        "splits": summaries,
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "dataset_v3_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n清单: {out / 'dataset_v3_manifest.json'}")
    print("下一步: python scripts/train_ai_operator.py "
          f"--data-root {out} --out-dir models/desktop_v3 --spatial-coord --epochs 10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
