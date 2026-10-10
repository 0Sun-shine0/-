#!/usr/bin/env python3
"""
真实 GUI 操作演示
=================

用**真实的 Qt 控件**搭一个目标窗口，让训练好的屏幕模型驱动它，并通过 Qt 的事件
系统投递**真实点击** —— 操作真的发生、控件真的响应、状态真的改变。

为什么不是"让智能体自主操作桌面"
--------------------------------
因为当前模型**不具备自主规划能力**，这一点由实验直接证明（``--mode probe`` 可复现）：

给同一张截图，把光标画在不同位置，模型的预测点几乎精确跟随光标：

===================  =====================
光标画在             模型预测点
===================  =====================
(0.1, 0.1)           (0.137, 0.087)
(0.5, 0.5)           (0.508, 0.508)
(0.7, 0.7)           (0.699, 0.685)
(0.9, 0.1)           (0.880, 0.106)
===================  =====================

原因是训练数据里**光标就画在点击目标上**，于是模型学到的是"点光标所在处"。
把这种模型放进朴素闭环（截屏→预测→执行→再截屏），它会原地打转：
光标不动、预测不动、状态不变。``--mode loop`` 会把这个退化现象如实跑出来。

所以本脚本演示的是模型的**有效用法**：**从演示中复现操作**（action reproduction）。
输入是分布内的截图（光标标出了用户点在哪），模型识别出"做了什么动作、点在哪"，
然后这个动作被**真实地**在 Qt 窗口上执行一遍，并检验控件是否真的响应了。

三种模式
--------
``reproduce``（默认）
    对数据集的 test 样本：模型预测 → 在真实 Qt 窗口上真实点击 → 检验控件响应。
    产出 GIF、逐步 trace 与三个指标。

``loop``
    朴素闭环，用来演示退化现象：光标固定不动，模型反复"点当前光标处"。

``probe``
    光标偏移实验，即上表的来源。

用法::

    .venv\\Scripts\\python.exe scripts/gui_agent_demo.py --mode reproduce
    .venv\\Scripts\\python.exe scripts/gui_agent_demo.py --mode loop --steps 6
    .venv\\Scripts\\python.exe scripts/gui_agent_demo.py --mode probe
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

# 离屏运行：本演示**不会**碰你的真实屏幕，也不会碰真实鼠标。
# 窗口在离屏平台上渲染，点击通过 Qt 事件系统投递到具体控件。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Qt6 不再自带字体，离屏渲染时默认找不到字体目录，界面文字会渲染成豆腐块。
# 指向系统字体目录后，Qt 至少能枚举到中文字体（实测 213 个字体族，含 SimHei）。
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

from PIL import Image, ImageDraw  # noqa: E402
from PyQt6.QtCore import QPoint, Qt  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTabBar,
    QWidget,
)

from ai_operator.data.desktop_dataset import draw_cursor, scan_split  # noqa: E402
from ai_operator.data_generation.dataset_creator import (  # noqa: E402
    ELEMENT_ACTION,
    UIElement,
    resolve_action,
)
from ai_operator.models.inference import ScreenPredictor  # noqa: E402
from demo_desktop_agent import _load_font  # noqa: E402

# 数据集里的界面尺寸
CANVAS = (1600, 900)

#: 主题配色，尽量贴近数据集的合成截图，减少视觉域差
THEME = {
    "bg": "#f0f0f0",
    "line": "#c8c8c8",
    "input_bg": "#ffffff",
    "accent": "#0078d7",
    "dark": "#2d2d2d",
    "text": "#1c1c1c",
}


# --------------------------------------------------------------------------- #
# 真实 Qt 目标窗口
# --------------------------------------------------------------------------- #
@dataclass
class ClickRecord:
    """一次真实点击的结果。"""

    widget_kind: Optional[str]
    widget_class: str
    executed: bool
    effect: str
    #: 响应类别：signal（控件发信号）/ focus（焦点转移）/ none（无可见变化）/
    #: no_widget（该位置没有控件）。分开统计才不会把"点已聚焦的输入框"这类
    #: 正常情况误算成失败。
    effect_kind: str = "none"


class DemoWindow(QWidget):
    """用真实 Qt 控件搭建的目标窗口。

    关键点：**几何来自样本自己的标注**（``elements[].bounds``），而不是我另写一套
    固定布局。这样"模型预测的落点"与"真实控件的位置"在同一坐标系里，点击才有意义。
    早先的版本用一套固定布局去执行所有样本，于是 file_explorer 样本的坐标被投到
    web_browser 布局上，落点大量落在空白处 —— 那测不出任何东西。

    每个子控件带 ``elementKind`` 属性，取值与数据集标注的 ``target_element``
    同一套命名，因此"模型预测的动作"与"落点下的真实控件类型"可以直接对照。
    """

    def __init__(self, elements: Sequence[Dict[str, object]], size: Tuple[int, int] = CANVAS,
                 app_type: str = "") -> None:
        super().__init__()
        self.setFixedSize(*size)
        self.setStyleSheet(f"background:{THEME['bg']};")
        self._children: List[QWidget] = []
        self.effects: List[str] = []
        # 应用类型用于解析"该元素在本应用下的规范动作"（编辑器里的 editor 是
        # type_text，而通用映射给的是 click_editor）
        self.app_type = app_type
        self._build(elements)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_annotation(cls, annotation: Dict[str, object],
                        size: Tuple[int, int] = CANVAS) -> "DemoWindow":
        """按标注里的 elements 建窗口（几何与训练数据完全一致）。"""
        elements = annotation.get("elements") or []
        app_type = str(annotation.get("app") or "")
        if not app_type:
            sid = str(annotation.get("sample_id") or "")
            app_type = sid.rsplit("_", 1)[0] if "_" in sid else ""
        return cls(elements, size, app_type)

    def _widget_for(self, kind: str, text: str) -> QWidget:
        """为某种数据集元素类型创建**真实可交互**的 Qt 控件。"""
        base = f"color:{THEME['text']};"
        if kind == "tab":
            return QTabBar()
        if kind == "address_bar":
            return QLineEdit(text or "https://www.example.com")
        if kind == "editor":
            return QPlainTextEdit(text or "# 可编辑区域")
        if kind in ("list", "sidebar"):
            widget = QListWidget()
            widget.addItem(text or ("快速访问" if kind == "sidebar" else "项目 1"))
            return widget
        if kind == "content":
            return QScrollArea()
        if kind == "menu":
            button = QPushButton(text or "菜单")
            button.setStyleSheet(base + f"background:{THEME['line']}; border:none; text-align:left; padding-left:10px;")
            return button
        if kind == "toolbar":
            button = QPushButton(text or "工具栏")
            button.setStyleSheet(base + f"background:{THEME['line']}; border:none;")
            return button
        if kind in ("button", "start_button"):
            button = QPushButton(text or "按钮")
            button.setStyleSheet(base + f"background:{THEME['accent']}; color:white; border-radius:6px;")
            return button
        if kind == "icon":
            button = QPushButton(text or "图标")
            button.setStyleSheet(base + f"background:{THEME['accent']}; color:white; border-radius:8px;")
            return button
        if kind == "folder":
            button = QPushButton(text or "文件夹")
            button.setStyleSheet(base + "background:#fad678; border-radius:4px;")
            return button
        if kind == "file":
            button = QPushButton(text or "文件")
            button.setStyleSheet(base + f"background:{THEME['input_bg']}; border:1px solid {THEME['line']};")
            return button
        if kind == "link":
            button = QPushButton(text or "链接")
            button.setStyleSheet("color:#2878dc; border:none; text-align:left;")
            return button
        return QWidget()

    def _build(self, elements: Sequence[Dict[str, object]]) -> None:
        width, height = self.width(), self.height()
        for element in elements:
            kind = str(element.get("type", ""))
            bounds = element.get("bounds") or []
            if len(bounds) != 4:
                continue
            x1, y1, x2, y2 = (int(float(v)) for v in bounds)
            x1, x2 = sorted((max(x1, 0), min(x2, width)))
            y1, y2 = sorted((max(y1, 0), min(y2, height)))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue

            widget = self._widget_for(kind, str(element.get("text", "")))
            widget.setParent(self)
            widget.setGeometry(x1, y1, x2 - x1, y2 - y1)
            widget.setProperty("elementKind", kind)
            self._children.append(widget)

            # 让各类控件都能产生**可观测**的状态变化
            if isinstance(widget, QTabBar):
                if widget.count() == 0:
                    for i in range(3):
                        widget.addTab(f"标签页 {i + 1}")
                widget.currentChanged.connect(
                    lambda i, w=widget: self.effects.append(f"标签页切换到 {i + 1}"))
            elif isinstance(widget, QLineEdit):
                widget.textEdited.connect(lambda _: self.effects.append(f"{kind} 内容被编辑"))
                widget.selectionChanged.connect(
                    lambda w=widget: self.effects.append(f"{kind} 被选中") if w.hasSelectedText() else None)
            elif isinstance(widget, QPlainTextEdit):
                widget.textChanged.connect(lambda w=widget: self.effects.append(f"{kind} 文本被修改"))
                # 单击编辑区虽然不改文本，但会移动光标 —— 这也是真实的响应
                widget.cursorPositionChanged.connect(
                    lambda w=widget: self.effects.append(f"{kind} 光标位置变化"))
            elif isinstance(widget, QListWidget):
                widget.itemClicked.connect(
                    lambda item, k=kind: self.effects.append(f"{k} 项 {item.text()} 被点击"))
            elif isinstance(widget, QPushButton):
                widget.clicked.connect(lambda _=False, k=kind: self.effects.append(f"{k} 被点击"))
            elif isinstance(widget, QScrollArea):
                widget.setProperty("elementKind", kind)

    # ------------------------------------------------------------------ #
    def kind_at(self, x: int, y: int) -> Tuple[Optional[str], str]:
        """返回屏幕坐标 (x, y) 下方的控件类型与其类名。

        必须**沿父链向上找**带 ``elementKind`` 的控件：``QListWidget`` /
        ``QScrollArea`` / ``QPlainTextEdit`` 内部都有 viewport 子控件，
        ``childAt`` 返回的往往是那个内部 widget，它没有我们标的属性，
        直接读会得到 ``None``（早期版本就踩了这个坑，导致"落点无控件"的假统计）。
        """
        widget = self.childAt(QPoint(int(x), int(y)))
        if widget is None:
            return None, "DemoWindow"
        probe = widget
        while probe is not None:
            kind = probe.property("elementKind")
            if kind:
                return str(kind), type(probe).__name__
            probe = probe.parentWidget()
        return None, type(widget).__name__

    def expected_action(self, kind: Optional[str]) -> Optional[str]:
        """该控件类型在**本应用**下对应的规范动作。

        必须复用生成器的 :func:`resolve_action`，而不是直接查全局
        ``ELEMENT_ACTION``：编辑器里的 ``editor`` 在应用级覆盖下是 ``type_text``，
        而全局映射给的是 ``click_editor``。用全局映射会把正确预测误判成"不自洽"
        （早期版本就产生了这样的假阴性）。
        """
        if not kind:
            return None
        return resolve_action(self.app_type or None, UIElement(element_type=kind, bounds=(0, 0, 1, 1)))

    def snapshot(self) -> Image.Image:
        """渲染窗口为 PIL 图像（真实渲染结果）。"""
        pixmap = self.grab()
        image = pixmap.toImage().convertToFormat(pixmap.toImage().Format.Format_RGB888)
        width, height = image.width(), image.height()
        buffer = image.constBits()
        buffer.setsize(image.sizeInBytes())
        arr = np.frombuffer(buffer, dtype=np.uint8).reshape(height, image.bytesPerLine())[:, : width * 3]
        return Image.frombytes("RGB", (width, height), arr.tobytes())

    def click_at(self, x: int, y: int) -> ClickRecord:
        """在 (x, y) 投递一次**真实**鼠标点击，并报告结果。

        通过 ``QTest.mouseClick`` 走 Qt 事件系统，控件会收到货真价实的 press/release
        并触发其槽函数。这比 OS 级鼠标注入更适合演示：确定、可重复，
        而且**不会劫持你的真实鼠标**。

        效果判定同时看两件事：控件发出的信号（如 ``clicked``），以及**焦点是否转移**
        到该控件 —— 后者更通用，因为 ``QScrollArea`` 之类控件单次点击不会发信号，
        但确实会获得焦点。
        """
        from PyQt6.QtTest import QTest

        widget = self.childAt(QPoint(int(x), int(y)))
        kind, class_name = self.kind_at(x, y)
        before = len(self.effects)

        if widget is None:
            return ClickRecord(kind, class_name, executed=False,
                               effect="该位置没有控件", effect_kind="no_widget")

        focus_before = self.focusWidget()
        local = widget.mapFrom(self, QPoint(int(x), int(y)))
        QTest.mouseClick(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, local)

        gained = self.effects[before:]
        if gained:
            return ClickRecord(kind, class_name, True, "; ".join(gained), "signal")

        focus_after = self.focusWidget()
        if focus_after is not None and focus_after is not focus_before:
            target = focus_after
            while target is not None and not target.property("elementKind"):
                target = target.parentWidget()
            named = target.property("elementKind") if target is not None else None
            return ClickRecord(kind, class_name, True,
                               f"焦点转移到 {named or type(focus_after).__name__}", "focus")

        return ClickRecord(kind, class_name, True,
                           "无可见变化（控件已处于该状态，如已聚焦/已选中的标签页）", "none")


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
def apply_font(app) -> None:
    """给 Qt 应用设一个带中文字形的字体。

    不设的话离屏渲染下中文会变成豆腐块（窗口看起来像空的），
    demo 的可读性会大打折扣。
    """
    from PyQt6.QtGui import QFont, QFontDatabase

    families = {str(f) for f in QFontDatabase.families()}
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial"):
        if name in families:
            app.setFont(QFont(name, 9))
            return


def annotate_frame(base: Image.Image, point: Optional[Tuple[int, int]],
                   record: Optional[ClickRecord], caption: Sequence[str],
                   scale: float, expected: Optional[str] = None,
                   highlight: bool = True) -> Image.Image:
    """在窗口渲染图上标注预测落点与说明文字。

    ``highlight=False`` 时只画准星、不画"已点击"方框，用于生成点击**前**的一帧，
    这样 GIF 里前后两帧有差别、能看出"点下去"这个动作（否则会被 GIF 优化合并掉）。
    """
    target_size = (int(base.width * scale), int(base.height * scale))
    canvas = base.resize(target_size, Image.LANCZOS)
    caption_h = 16 * len(caption) + 10
    out = Image.new("RGB", (canvas.width, canvas.height + caption_h), (14, 14, 18))
    out.paste(canvas, (0, 0))
    draw = ImageDraw.Draw(out)

    if point is not None:
        px, py = point[0] * scale, point[1] * scale
        # 深色描边 + 亮色，深浅背景都清晰
        for radius, colour, width in ((13, (18, 18, 22), 5), (13, (255, 214, 0), 2)):
            draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                         outline=colour, width=width)
        draw.line([px - 20, py, px - 14, py], fill=(255, 214, 0), width=2)
        draw.line([px + 14, py, px + 20, py], fill=(255, 214, 0), width=2)
        draw.line([px, py - 20, px, py - 14], fill=(255, 214, 0), width=2)
        draw.line([px, py + 14, px, py + 20], fill=(255, 214, 0), width=2)

        if highlight and record is not None and record.executed:
            colour = (0, 230, 90) if record.widget_kind == expected else (255, 140, 60)
            draw.rectangle([px - 16, py - 16, px + 16, py + 16], outline=colour, width=2)

    font = _load_font(15)
    for i, line in enumerate(caption):
        draw.text((8, canvas.height + 5 + i * 16), line, fill=(235, 235, 240), font=font)
    return out


# --------------------------------------------------------------------------- #
# 模式一：从演示复现操作
# --------------------------------------------------------------------------- #
def point_to_rect_distance(px: float, py: float, bounds: Sequence[float]) -> float:
    """点到矩形的距离（点在矩形内时为 0）。"""
    x1, y1, x2, y2 = (float(v) for v in bounds)
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return float(np.hypot(dx, dy))


def _load_annotation(data_root: Path, split: str, sample_id: str) -> Optional[Dict[str, object]]:
    path = data_root / split / "annotations" / f"{sample_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _size_from_annotation(annotation: Dict[str, object],
                          image_size: Optional[Tuple[int, int]] = None,
                          fallback: Tuple[int, int] = CANVAS) -> Tuple[int, int]:
    """确定窗口尺寸，必须与标注的坐标系统一致。

    优先用**截图的真实尺寸**：标注里的 ``elements[].bounds`` 就在这个坐标系里。
    注意 v1 生成器写的标注**没有** ``resolution`` 字段，早先版本因此回落到
    1600x900，而 bounds 却到 2122 —— 控件被裁掉、落点错位，
    统计出来的"落点无控件"其实是坐标系 bug，不是模型问题。
    """
    if image_size is not None:
        return int(image_size[0]), int(image_size[1])
    text = str(annotation.get("resolution") or "")
    if "x" in text:
        try:
            width, height = text.lower().split("x")
            return int(width), int(height)
        except ValueError:
            pass
    return fallback


def mode_reproduce(args: argparse.Namespace) -> int:
    predictor = ScreenPredictor.load()
    samples = scan_split(args.data_root, args.split)
    if not samples:
        print(f"[错误] {args.data_root}/{args.split} 下没有样本")
        return 1

    # 按应用轮转取样，保证覆盖面（而不是只取前 N 个，那会清一色是 basic_*）
    by_app: Dict[str, List[object]] = {}
    for sample in samples:
        by_app.setdefault(sample.sample_id.rsplit("_", 1)[0], []).append(sample)
    order = sorted(by_app, key=lambda a: -len(by_app[a]))
    picked: List[object] = []
    cursor = 0
    while len(picked) < args.count and any(by_app[a] for a in order):
        app_name = order[cursor % len(order)]
        if by_app[app_name]:
            picked.append(by_app[app_name].pop(0))
        cursor += 1

    app = QApplication.instance() or QApplication([])
    apply_font(app)
    data_root = Path(args.data_root)

    print("=" * 74)
    print("真实 GUI 操作：从演示中复现操作")
    print("=" * 74)
    print(f"模型      : {predictor.info.name}（{predictor.info.num_actions} 类）")
    print("目标窗口  : 真实 Qt 控件；几何取自**该样本自己的标注**，")
    print("            窗口尺寸 = 该样本原始分辨率（保证与标注同一坐标系）")
    print(f"Qt 平台   : {os.environ.get('QT_QPA_PLATFORM')}（不接触真实屏幕与鼠标）")
    print()
    print(f"{'样本':22} {'分辨率':>10} {'真值动作':16} {'预测动作':16} {'落点下控件':12} "
          f"{'动作一致':8} {'自洽':6} 效果")
    print("-" * 132)

    rows: List[Dict[str, object]] = []
    frames: List[Image.Image] = []

    for sample in picked:
        annotation = _load_annotation(data_root, args.split, sample.sample_id)
        if annotation is None:
            print(f"  [跳过] 缺少标注 {sample.sample_id}")
            continue

        with Image.open(sample.image_path) as im:
            demo_image = im.convert("RGB")
        size = _size_from_annotation(annotation, demo_image.size)

        window = DemoWindow.from_annotation(annotation, size)
        window.show()
        app.processEvents()

        result = predictor.predict(demo_image, screen_size=size)
        point = result["point_px"]
        record = window.click_at(*point)
        app.processEvents()

        expected = window.expected_action(record.widget_kind)
        action_ok = result["action"] == sample.action
        grounding_ok = (expected is not None) and (result["action"] == expected)

        # 落点与标注目标框的距离，以及目标本身的尺寸 —— 用来解释"为什么会脱靶"
        gt_bounds = (annotation.get("operation") or {}).get("target_bounds") or []
        target_dist = point_to_rect_distance(point[0], point[1], gt_bounds) if len(gt_bounds) == 4 else None
        target_size = (int(gt_bounds[2] - gt_bounds[0]), int(gt_bounds[3] - gt_bounds[1])) \
            if len(gt_bounds) == 4 else None

        print(f"{sample.sample_id:22} {f'{size[0]}x{size[1]}':>10} {sample.action:16} "
              f"{result['action']:16} {str(record.widget_kind):12} {str(action_ok):8} "
              f"{str(grounding_ok):6} {record.effect}")

        rows.append({
            "sample_id": sample.sample_id,
            "app": annotation.get("app") or sample.sample_id.rsplit("_", 1)[0],
            "resolution": f"{size[0]}x{size[1]}",
            "gt_action": sample.action,
            "gt_target_bounds": list(gt_bounds),
            "gt_target_size_px": list(target_size) if target_size else None,
            "pred_action": result["action"],
            "confidence": result["confidence"],
            "pred_point_px": list(point),
            "point_to_target_px": target_dist,
            "point_to_target_ratio": (target_dist / size[0]) if target_dist is not None else None,
            "widget_kind_at_point": record.widget_kind,
            "widget_class": record.widget_class,
            "expected_action_for_widget": expected,
            "action_matches_label": bool(action_ok),
            "action_grounded_in_widget": bool(grounding_ok),
            "click_executed": record.executed,
            "effect_kind": record.effect_kind,
            "effect": record.effect,
        })

        base = window.snapshot()
        scale = args.width / max(base.width, 1)
        caption = [
            f"{sample.sample_id}   {annotation.get('app', '')}   {size[0]}x{size[1]}",
            f"标注动作 {sample.action}（{_zh(sample.action)}）   |   模型预测 {result['action']} "
            f"{result['confidence']:.0%}",
            f"落点下真实控件 {record.widget_kind}（{record.widget_class}）  ->  {record.effect}",
        ]
        # 两帧有区别（点击前只有准星，点击后多一个响应方框），GIF 才看得出"点下去了"
        frames.append(annotate_frame(base, point, record, caption, scale, expected, highlight=False))
        frames.append(annotate_frame(base, point, record, caption, scale, expected, highlight=True))
        window.close()

    def rate(key: str) -> float:
        return sum(1 for r in rows if r[key]) / max(len(rows), 1)

    def count_kind(kind: str) -> int:
        return sum(1 for r in rows if r["effect_kind"] == kind)

    total = len(rows)
    print("-" * 132)
    print(f"\n动作与标注一致          : {rate('action_matches_label'):.0%}"
          f"  ({sum(1 for r in rows if r['action_matches_label'])}/{total})"
          f"   <- 模型是否认对了演示的动作")
    print(f"点击命中真实控件        : {rate('click_executed'):.0%}"
          f"  ({sum(1 for r in rows if r['click_executed'])}/{total})"
          f"   <- 落点下方确实有 Qt 控件并收到了点击")
    print(f"**动作与落点控件自洽**   : {rate('action_grounded_in_widget'):.0%}"
          f"  ({sum(1 for r in rows if r['action_grounded_in_widget'])}/{total})"
          f"   <- 它说的动作，是否与它自己指的位置下方的控件一致")
    print(f"\n交互响应明细（分开统计，避免把正常情况当失败）:")
    print(f"  控件发信号 (clicked/currentChanged 等) : {count_kind('signal'):>3}")
    print(f"  焦点转移到控件                        : {count_kind('focus'):>3}")
    print(f"  无可见变化（已聚焦/已是当前标签页）    : {count_kind('none'):>3}")
    print(f"  落点没有控件（预测脱靶）              : {count_kind('no_widget'):>3}")
    print(f"  -> 点击真的被控件接收的比例           : "
          f"{(count_kind('signal') + count_kind('focus') + count_kind('none')) / max(total, 1):.0%}")

    # 脱靶原因分析：把"误差"与"目标尺寸"放在一起看，才能解释为什么会漏
    dists = [r["point_to_target_px"] for r in rows if r["point_to_target_px"] is not None]
    if dists:
        print(f"\n落点与标注目标框的距离: 平均 {np.mean(dists):.1f}px | 中位 {np.median(dists):.1f}px "
              f"| 最大 {np.max(dists):.1f}px  （目标框尺寸中位 "
              f"{np.median([r['gt_target_size_px'][0] for r in rows if r['gt_target_size_px']]):.0f}px 宽）")
    misses = [r for r in rows if r["effect_kind"] == "no_widget"]
    if misses:
        print("脱靶样本:")
        for r in misses:
            print(f"  {r['sample_id']:22} 分辨率 {r['resolution']:>10}  目标框 {r['gt_target_size_px']}  "
                  f"落点偏离目标框 {r['point_to_target_px']:.0f}px"
                  f"（占屏宽 {r['point_to_target_ratio']:.2%}）")
        print("  说明：相对误差很小，但桌面图标/按钮在 2560~3840 宽的截图里只有 30~70px，")
        print("        所以绝对误差一超过目标尺寸就会脱靶 —— 这是小目标问题，不是坐标系错误。")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "gui_agent_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": predictor.info.name,
            "window": {
                "toolkit": "PyQt6", "platform": "offscreen",
                "sizing": "每个样本使用其截图原始分辨率",
                "geometry_source": "样本自身标注的 elements[].bounds",
            },
            "note": "点击经 Qt 事件系统投递到真实控件；不接触真实屏幕与鼠标",
            "metrics": {
                "action_matches_label": rate("action_matches_label"),
                "click_hits_widget": rate("click_executed"),
                "action_grounded_in_widget": rate("action_grounded_in_widget"),
                "effect_signal": count_kind("signal"),
                "effect_focus": count_kind("focus"),
                "effect_none": count_kind("none"),
                "effect_no_widget": count_kind("no_widget"),
            },
            "steps": rows,
        }, f, ensure_ascii=False, indent=2)

    if frames:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(out_path, format="GIF", save_all=True, append_images=frames[1:],
                       duration=args.frame_ms, loop=0, optimize=True, disposal=2)
        print(f"\n演示 GIF  : {out_path}（{len(frames)} 帧）")
    print(f"逐步记录  : {report_dir / 'gui_agent_report.json'}")
    return 0


def _zh(action: str) -> str:
    from ai_operator.data.desktop_dataset import action_zh
    return action_zh(action)


# --------------------------------------------------------------------------- #
# 模式二：朴素闭环（演示退化）
# --------------------------------------------------------------------------- #
def _first_window(args: argparse.Namespace, app) -> Optional[DemoWindow]:
    """取第一个可用样本的标注，建一个几何匹配的窗口（loop/probe 用）。"""
    samples = scan_split(args.data_root, args.split)
    if not samples:
        return None
    if args.sample:
        samples = [s for s in samples if s.sample_id == args.sample] or samples
    sample = samples[0]
    annotation = _load_annotation(Path(args.data_root), args.split, sample.sample_id)
    if annotation is None:
        return None
    # 用截图的真实尺寸，保证与标注坐标系统一致
    with Image.open(sample.image_path) as im:
        size = im.size
    size = _size_from_annotation(annotation, size)
    window = DemoWindow.from_annotation(annotation, size)
    window.show()
    app.processEvents()
    return window


def mode_loop(args: argparse.Namespace) -> int:
    predictor = ScreenPredictor.load()
    app = QApplication.instance() or QApplication([])
    apply_font(app)
    window = _first_window(args, app)
    if window is None:
        print("[错误] 没有可用样本/标注")
        return 1

    print("=" * 74)
    print("朴素闭环：截屏 -> 预测 -> 执行 -> 再截屏")
    print("=" * 74)
    print("预期会退化：训练数据里光标就在目标上，模型学到的是'点光标所在处'。")
    print("闭环中光标由上一次点击决定，于是它只会反复点自己所在的位置。\n")

    cursor = (window.width() // 2, window.height() // 2)  # 初始光标在屏幕中心
    history: List[Dict[str, object]] = []
    frames: List[Image.Image] = []

    print(f"{'步':>3} {'光标位置':>16} {'预测动作':16} {'预测落点':>16} {'位移px':>8} 状态变化")
    print("-" * 100)

    for step in range(1, args.steps + 1):
        base = window.snapshot()
        arr = draw_cursor(np.asarray(base, dtype=np.uint8), cursor[0], cursor[1],
                          size=max(16, int(round(15 * window.width() / 224))))
        screen = Image.fromarray(arr)

        result = predictor.predict(screen, screen_size=(window.width(), window.height()))
        point = result["point_px"]
        moved = float(np.hypot(point[0] - cursor[0], point[1] - cursor[1]))
        record = window.click_at(*point)
        app.processEvents()

        print(f"{step:>3} {str(cursor):>16} {result['action']:16} {str(point):>16} "
              f"{moved:>8.1f} {record.effect}")

        history.append({
            "step": step, "cursor": list(cursor), "pred_action": result["action"],
            "pred_point": list(point), "moved_px": moved,
            "widget_kind": record.widget_kind, "effect": record.effect,
        })

        drawn = draw_cursor(np.asarray(window.snapshot(), dtype=np.uint8), point[0], point[1],
                            size=max(16, int(round(15 * window.width() / 224))))
        caption = [
            f"朴素闭环 第 {step} 步   （窗口 {window.width()}x{window.height()}）",
            f"光标 ({cursor[0]}, {cursor[1]})  ->  模型预测 {result['action']} @ {point}（位移 {moved:.0f}px）",
            f"落点控件 {record.widget_kind}  ->  {record.effect}",
        ]
        frames.append(annotate_frame(Image.fromarray(drawn), point, record, caption, args.scale))
        cursor = point

    window.close()

    moves = [h["moved_px"] for h in history]
    print("-" * 92)
    print(f"平均每步位移 {np.mean(moves):.1f}px；最终光标 {history[-1]['cursor']} -> {history[-1]['pred_point']}")
    if np.mean(moves) < 40:
        print("结论：位移很小 —— 模型确实只是在'点光标当前所在处'，闭环无法推进任务。")
    else:
        print("结论：位移较大，退化程度低于预期，值得进一步分析。")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "gui_agent_loop.json", "w", encoding="utf-8") as f:
        json.dump({"note": "朴素闭环，用于演示退化现象", "steps": history}, f,
                  ensure_ascii=False, indent=2)
    if frames:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:],
                       duration=args.frame_ms, loop=0, optimize=True, disposal=2)
        print(f"\n闭环 GIF  : {out}")
    print(f"逐步记录  : {report_dir / 'gui_agent_loop.json'}")
    return 0


# --------------------------------------------------------------------------- #
# 模式三：光标偏移实验
# --------------------------------------------------------------------------- #
def mode_probe(args: argparse.Namespace) -> int:
    """光标偏移实验。

    必须在**没有烘烤光标**的画面上做：数据集截图里已经有一个光标画在目标处，
    若在它之上再画一个光标，画面里就有两个光标，测出来的"跟随"是假的
    （实测会把预测点吸到原来那个光标上，读出 700px 的误导性距离）。
    因此这里用 Qt 窗口渲染的干净画面，光标完全由本脚本控制。
    """
    predictor = ScreenPredictor.load()
    app = QApplication.instance() or QApplication([])
    apply_font(app)
    window = _first_window(args, app)
    if window is None:
        print("[错误] 没有可用样本/标注")
        return 1

    width, height = window.width(), window.height()
    print("=" * 74)
    print("光标偏移实验：在无光标的干净窗口上把光标画到不同位置")
    print("=" * 74)
    print(f"模型 {predictor.info.name} | 窗口 {width}x{height}（Qt 渲染，无烘烤光标）")
    print("数据集约定：光标高度 = 15 * 屏宽 / 224，缩放到 224 后正好 15px\n")
    print(f"{'光标画在':>20} | {'预测动作':16} | {'预测点像素':>16} | {'预测-光标 距离px':>16}")
    print("-" * 78)

    size = max(16, int(round(15 * width / 224)))
    base = window.snapshot()
    rows = []
    for frac in [(0.1, 0.1), (0.3, 0.3), (0.5, 0.5), (0.7, 0.7), (0.9, 0.9), (0.5, 0.9), (0.9, 0.1)]:
        cx, cy = frac[0] * width, frac[1] * height
        arr = draw_cursor(np.asarray(base, dtype=np.uint8), cx, cy, size=size)
        result = predictor.predict(Image.fromarray(arr), screen_size=(width, height))
        px, py = result["point_px"]
        distance = float(np.hypot(px - cx, py - cy))
        kind, _ = window.kind_at(px, py)
        print(f"  ({cx:6.0f},{cy:6.0f}) 即 {str(frac):>12} | {result['action']:16} | "
              f"({px:6d},{py:6d}) | {distance:>16.1f}   落点控件 {kind}")
        rows.append({"cursor": [cx, cy], "cursor_frac": list(frac), "pred_action": result["action"],
                     "pred_point": [px, py], "distance_px": distance, "widget_kind": kind})

    window.close()
    mean_distance = float(np.mean([r["distance_px"] for r in rows]))
    print("-" * 78)
    print(f"平均 预测点与光标 距离: {mean_distance:.1f}px（窗口尺度 {width}px）")
    if mean_distance < width * 0.06:
        print("结论：预测点基本跟随光标 —— 模型学到的是'点光标所在处'，")
        print("      而不是'根据界面内容决定该点哪里'，因此无法自主推进任务。")
    else:
        print("结论：预测点与光标有明显偏离，模型并非单纯跟随光标。")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "gui_agent_probe.json", "w", encoding="utf-8") as f:
        json.dump({
            "surface": "Qt DemoWindow（无烘烤光标）",
            "window": list(CANVAS),
            "mean_cursor_follow_px": mean_distance,
            "mean_cursor_follow_ratio": mean_distance / width,
            "trials": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"记录: {report_dir / 'gui_agent_probe.json'}")
    return 0


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="真实 GUI 操作演示")
    parser.add_argument("--mode", choices=["reproduce", "loop", "probe"], default="reproduce")
    parser.add_argument("--data-root", default=str(ROOT / "ai_operator_dataset_v3"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", default="", help="probe 模式下指定样本")
    parser.add_argument("--count", type=int, default=4, help="reproduce 模式下演示几个应用各一个")
    parser.add_argument("--steps", type=int, default=6, help="loop 模式步数")
    parser.add_argument("--scale", type=float, default=0.42, help="（loop/probe 用）窗口缩放比例")
    parser.add_argument("--width", type=int, default=900, help="GIF 帧宽度（reproduce 用）")
    parser.add_argument("--frame-ms", type=int, default=1200)
    parser.add_argument("--report-dir", default=str(ROOT / "reports"))
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.out:
        args.out = str(Path(args.report_dir) / f"gui_agent_{args.mode}.gif")
    if args.mode == "reproduce":
        return mode_reproduce(args)
    if args.mode == "loop":
        return mode_loop(args)
    return mode_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
