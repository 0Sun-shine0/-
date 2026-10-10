"""
真实 GUI 操作演示的回归测试。

守的是搭这个 demo 过程中实际踩到的四个坑，它们都会让统计结果**看起来是模型问题、
实际是 demo 自己的 bug**：

1. ``childAt`` 返回的是 ``QListWidget``/``QScrollArea`` 的内部 viewport，
   直接读属性得到 ``None`` —— 会把"点中了列表"误报成"落点没有控件"。
2. 标注解析用了 ``resolution`` 字段，但 v1 生成器**不写**这个字段，
   于是窗口尺寸回落到 1600x900，而 bounds 到 2122 —— 控件被裁、落点错位。
3. "该元素在本应用下的规范动作"直接查全局 ``ELEMENT_ACTION``，
   而编辑器里的 ``editor`` 在应用级覆盖下是 ``type_text`` —— 正确预测被误判为不自洽。
4. ``loop``/``probe`` 用了固定画布尺寸而不是窗口实际尺寸，
   坐标系不一致导致坐标全错。

运行::

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


class PointToRectTest(unittest.TestCase):
    """落点到目标框的距离计算。"""

    def test_inside_is_zero(self):
        from gui_agent_demo import point_to_rect_distance

        self.assertEqual(point_to_rect_distance(50, 50, [0, 0, 100, 100]), 0.0)

    def test_outside_uses_euclidean_distance(self):
        from gui_agent_demo import point_to_rect_distance

        # 正右方 10px
        self.assertAlmostEqual(point_to_rect_distance(110, 50, [0, 0, 100, 100]), 10.0)
        # 右下角 3-4-5
        self.assertAlmostEqual(point_to_rect_distance(103, 104, [0, 0, 100, 100]), 5.0)
        # 正上方
        self.assertAlmostEqual(point_to_rect_distance(50, -7, [0, 0, 100, 100]), 7.0)


class SizeResolutionTest(unittest.TestCase):
    """窗口尺寸必须与标注坐标系统一致。"""

    def test_prefers_image_size_over_missing_resolution(self):
        """v1 标注没有 resolution 字段 —— 必须回落到截图真实尺寸。

        这正是"basic_* 样本控件被裁掉"的根因：早先版本回落到 1600x900，
        而 bounds 到 2122。
        """
        from gui_agent_demo import _size_from_annotation

        annotation = {"sample_id": "basic_000001"}  # 无 resolution
        self.assertEqual(_size_from_annotation(annotation, (2560, 1440)), (2560, 1440))

    def test_uses_resolution_when_no_image_size(self):
        from gui_agent_demo import _size_from_annotation

        self.assertEqual(_size_from_annotation({"resolution": "3840x2160"}), (3840, 2160))

    def test_falls_back_to_default(self):
        from gui_agent_demo import CANVAS, _size_from_annotation

        self.assertEqual(_size_from_annotation({}), CANVAS)


class DemoWindowTest(unittest.TestCase):
    """真实 Qt 窗口的构建、命中测试与点击投递。"""

    _qapp = None

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        if os.name == "nt":
            os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
        try:
            from PyQt6.QtWidgets import QApplication  # noqa: F401
        except Exception:
            raise unittest.SkipTest("PyQt6 不可用")

    def _app(self):
        from PyQt6.QtWidgets import QApplication

        if DemoWindowTest._qapp is None:
            DemoWindowTest._qapp = QApplication.instance() or QApplication([])
        return DemoWindowTest._qapp

    def _annotation(self):
        return {
            "sample_id": "text_editor_000001",
            "app": "text_editor",
            "resolution": "1600x900",
            "operation": {"action": "type_text", "target_bounds": [300, 200, 900, 600]},
            "elements": [
                {"type": "menu", "bounds": [0, 0, 1600, 30], "text": "文件 编辑", "action": "click_menu"},
                {"type": "toolbar", "bounds": [0, 30, 1600, 70], "text": "", "action": "click_toolbar"},
                {"type": "sidebar", "bounds": [0, 70, 200, 900], "text": "资源", "action": "click_sidebar"},
                {"type": "editor", "bounds": [200, 70, 1600, 900], "text": "code", "action": "type_text"},
            ],
        }

    def test_builds_one_widget_per_element(self):
        from gui_agent_demo import DemoWindow

        self._app()
        annotation = self._annotation()
        window = DemoWindow.from_annotation(annotation, (1600, 900))
        self.assertEqual(len(window._children), len(annotation["elements"]))
        window.close()

    def test_kind_at_walks_up_from_inner_viewport(self):
        """命中测试必须沿父链找到带 elementKind 的控件。

        ``QListWidget`` / ``QScrollArea`` 内部都有 viewport 子控件，
        ``childAt`` 常返回那个内部 widget（没有我们的属性），
        直接读会得到 None —— 早先版本因此把"点中列表"误报成"落点没有控件"。
        """
        from gui_agent_demo import DemoWindow

        self._app()
        window = DemoWindow.from_annotation(self._annotation(), (1600, 900))
        window.show()
        self._app().processEvents()

        # sidebar 是 QListWidget，editor 是 QPlainTextEdit，两者都有内部 viewport
        for kind, point in (("sidebar", (100, 400)), ("editor", (800, 400)),
                            ("menu", (800, 15)), ("toolbar", (800, 50))):
            hit, _cls = window.kind_at(*point)
            self.assertEqual(hit, kind, f"{point} 处应命中 {kind}，实际 {hit}")
        window.close()

    def test_expected_action_uses_app_override(self):
        """editor 在编辑器里是 type_text，而不是全局映射的 click_editor。

        早先版本直接查全局 ELEMENT_ACTION，把正确预测误判成"不自洽"。
        """
        from gui_agent_demo import DemoWindow

        self._app()
        window = DemoWindow.from_annotation(self._annotation(), (1600, 900))
        self.assertEqual(window.expected_action("editor"), "type_text")
        self.assertEqual(window.expected_action("menu"), "click_menu")
        self.assertEqual(window.expected_action("sidebar"), "click_sidebar")
        self.assertIsNone(window.expected_action("statusbar"))
        window.close()

    def test_click_really_delivered(self):
        """点击必须真的投递到控件，并产生可观测响应（信号或焦点转移）。"""
        from gui_agent_demo import DemoWindow

        app = self._app()
        window = DemoWindow.from_annotation(self._annotation(), (1600, 900))
        window.show()
        app.processEvents()

        for kind, point, in (("menu", (800, 15)), ("editor", (800, 400))):
            record = window.click_at(*point)
            app.processEvents()
            self.assertTrue(record.executed, f"{kind} 点击未投递")
            self.assertEqual(record.widget_kind, kind)
            self.assertIn(record.effect_kind, ("signal", "focus"),
                          f"{kind} 点击没有任何可观测响应")
        window.close()

    def test_click_outside_any_widget_is_reported(self):
        from gui_agent_demo import DemoWindow

        self._app()
        # 一个只有一个小组件的窗口，点在空白处
        window = DemoWindow([{"type": "button", "bounds": [10, 10, 60, 50], "text": "b"}],
                            (400, 300), "web_browser")
        window.show()
        self._app().processEvents()
        record = window.click_at(300, 250)
        self.assertFalse(record.executed)
        self.assertEqual(record.effect_kind, "no_widget")
        window.close()

    def test_snapshot_renders_at_requested_size(self):
        from gui_agent_demo import DemoWindow

        self._app()
        window = DemoWindow.from_annotation(self._annotation(), (1600, 900))
        window.show()
        self._app().processEvents()
        image = window.snapshot()
        self.assertEqual(image.size, (1600, 900))
        self.assertEqual(image.mode, "RGB")
        window.close()

    def test_window_size_follows_annotation_resolution(self):
        from gui_agent_demo import DemoWindow

        self._app()
        window = DemoWindow([{"type": "icon", "bounds": [2000, 1000, 2100, 1100], "text": "i"}],
                            (2560, 1440), "basic")
        self.assertEqual((window.width(), window.height()), (2560, 1440))
        # bounds 在 2560x1440 坐标系内，应落在窗口范围内
        hit, _ = window.kind_at(2050, 1050)
        self.assertEqual(hit, "icon")
        window.close()


class DemoEndToEndTest(unittest.TestCase):
    """端到端冒烟：真的跑一遍复现模式并产出报告。缺素材时跳过。"""

    def test_reproduce_produces_report_and_metrics(self):
        import subprocess

        csv_dir = ROOT / "ai_operator_dataset_v3"
        if not (csv_dir / "test" / "annotations").is_dir():
            self.skipTest("缺少 v3 数据集")
        if not (ROOT / "models" / "desktop_v3_coord" / "best_model.pth").is_file():
            self.skipTest("缺少已训练模型")

        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "gui_agent_demo.py"),
                 "--mode", "reproduce", "--count", "4",
                 "--report-dir", tmp, "--out", str(Path(tmp) / "d.gif")],
                capture_output=True, text=True,
                # 脚本按 UTF-8 输出；Windows 上若让 subprocess 用默认 GBK 解码会抛
                # UnicodeDecodeError（这本身就是本仓库已知的编码坑之一）
                encoding="utf-8", errors="replace", cwd=str(ROOT),
            )
            report_path = Path(tmp) / "gui_agent_report.json"
            self.assertTrue(report_path.is_file(),
                            f"未生成报告。stdout/stderr:\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
            report = json.loads(report_path.read_text(encoding="utf-8"))

        metrics = report["metrics"]
        # 动作识别在这批样本上应当全对（这是模型的强项）
        self.assertEqual(metrics["action_matches_label"], 1.0)
        # 每个 step 都要有完整字段，便于事后分析
        for step in report["steps"]:
            for key in ("sample_id", "gt_action", "pred_action", "pred_point_px",
                        "widget_kind_at_point", "click_executed", "effect_kind"):
                self.assertIn(key, step)
            # 落点必须落在窗口坐标系内
            width, height = (int(v) for v in step["resolution"].split("x"))
            px, py = step["pred_point_px"]
            self.assertTrue(0 <= px <= width, f"{step['sample_id']} 落点 x 越界")
            self.assertTrue(0 <= py <= height, f"{step['sample_id']} 落点 y 越界")


if __name__ == "__main__":
    unittest.main()
