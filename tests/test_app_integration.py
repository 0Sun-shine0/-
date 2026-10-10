"""
应用集成层测试：配置加载、统一推理入口、以及"产品入口能否真正跑起来"。

这一组测试守的是本次修掉的一批**集成类**缺陷 —— 它们不会让单元测试失败，
但会让程序在用户手里直接崩掉或悄悄输出垃圾：

- 项目自带 ``.env`` 却没有任何代码读取它
- 缺少 API key 时 ``DeepSeekAgent`` 抛异常，导致 GUI 连窗口都打不开
- GUI 构造随机初始化的模型、从不加载权重（界面"在工作"，输出全是噪声）
- GUI 把 OpenCV 的 numpy 数组直接喂给 ``nn.Module``
- 坐标被多套一次 ``sigmoid``，或乘上模型输入尺寸而非真实屏幕尺寸

运行::

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


class DotenvLoadingTest(unittest.TestCase):
    """`.env` 解析：语法、引号、以及与已有环境变量的优先级。"""

    def setUp(self):
        self._saved = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)

    def _write(self, tmp: str, text: str) -> Path:
        path = Path(tmp) / ".env"
        path.write_text(text, encoding="utf-8")
        return path

    def test_parses_basic_syntax(self):
        import tempfile

        from ai_operator.core.config import load_dotenv_file

        with tempfile.TemporaryDirectory() as tmp:
            for key in ("A_PLAIN", "B_QUOTED", "C_SPACED", "D_COMMENTED", "E_EXPORT"):
                os.environ.pop(key, None)
            path = self._write(tmp, "\n".join([
                "# 这是注释",
                "A_PLAIN=value1",
                'B_QUOTED="value 2"',
                "C_SPACED   =   value3   ",
                "# D_COMMENTED=should_not_load",
                "export E_EXPORT=value5",
                "",
                "MALFORMED_LINE_NO_EQUALS",
            ]))
            loaded = load_dotenv_file(path)

        self.assertEqual(loaded.get("A_PLAIN"), "value1")
        self.assertEqual(loaded.get("B_QUOTED"), "value 2")
        self.assertEqual(loaded.get("C_SPACED"), "value3")
        self.assertNotIn("D_COMMENTED", loaded)
        self.assertEqual(loaded.get("E_EXPORT"), "value5")

    def test_nonempty_env_wins_but_empty_falls_through(self):
        """显式设的变量优先；"存在但为空"要能被 .env 补上。

        Windows 上 ``$env:KEY=""`` 会留下一个已定义的空变量。若把空值当成
        "已设置"，``.env`` 就会被静默忽略 —— 与用户直觉完全相反。
        """
        import tempfile

        from ai_operator.core.config import load_dotenv_file

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "PRIORITY_KEY=from_dotenv\nEMPTY_KEY=from_dotenv\n")

            os.environ["PRIORITY_KEY"] = "from_env"
            loaded = load_dotenv_file(path)
            self.assertEqual(os.environ["PRIORITY_KEY"], "from_env")
            self.assertNotIn("PRIORITY_KEY", loaded, "非空环境变量不应被 .env 覆盖")

            os.environ["EMPTY_KEY"] = ""
            loaded = load_dotenv_file(path)
            self.assertEqual(os.environ["EMPTY_KEY"], "from_dotenv", "空环境变量应被 .env 补上")
            self.assertIn("EMPTY_KEY", loaded)

    def test_missing_file_returns_empty(self):
        from ai_operator.core.config import load_dotenv_file

        self.assertEqual(load_dotenv_file(Path("__definitely_not_here__.env")), {})

    def test_project_env_is_loaded_at_import_time(self):
        """项目根的 .env 必须真的被加载（否则它只是个装饰文件）。"""
        env_path = ROOT / ".env"
        if not env_path.is_file():
            self.skipTest("项目根目录没有 .env")

        keys = [
            line.split("=", 1)[0].strip()
            for line in env_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.strip().startswith("#") and "=" in line
        ]
        if not keys:
            self.skipTest(".env 里没有键值对")

        from ai_operator.core.config import Config

        # Config 的字段在类定义时就从 os.getenv 取过值，所以这里能读到就说明
        # 加载发生在类定义之前（顺序写错的话这里会是 None）
        for key in keys:
            if key == "DEEPSEEK_API_KEY":
                self.assertIsNotNone(Config().DEEPSEEK_API_KEY,
                                     "DEEPSEEK_API_KEY 未从 .env 载入：加载时机可能晚于类定义")


class DeepSeekAgentLazyClientTest(unittest.TestCase):
    """缺少密钥不应阻止对象创建，只应阻止实际调用。"""

    def _agent_without_key(self):
        from ai_operator.agents.deepseek_agent import DeepSeekAgent
        from ai_operator.core.config import Config

        cfg = Config()
        cfg.DEEPSEEK_API_KEY = None
        return DeepSeekAgent(cfg)

    def test_constructs_without_key(self):
        """修复前这里会抛 OpenAIError，导致 GUI 完全无法启动。"""
        try:
            agent = self._agent_without_key()
        except Exception as e:  # pragma: no cover - 修复失效时会走到这里
            self.fail(f"缺少密钥时构造 agent 抛出了异常: {type(e).__name__}: {e}")
        self.assertFalse(agent.is_available())

    def test_client_access_raises_actionable_error(self):
        agent = self._agent_without_key()
        with self.assertRaises(RuntimeError) as ctx:
            _ = agent.client
        message = str(ctx.exception)
        self.assertIn("DEEPSEEK_API_KEY", message)
        self.assertIn(".env", message, "错误信息应告诉用户可以用 .env")

    def test_process_task_degrades_gracefully(self):
        """无密钥时 process_task 应走兜底分支而不是把异常抛给界面。"""
        agent = self._agent_without_key()
        result = agent.process_task("打开记事本")
        self.assertIsInstance(result, dict)
        self.assertTrue(result, "兜底响应不应是空字典")


class CnnPredictContractTest(unittest.TestCase):
    """`EnhancedCNN.predict` 的输入输出契约。"""

    def _model(self):
        from ai_operator.models.cnn_model import EnhancedCNN

        torch.manual_seed(0)
        return EnhancedCNN(4, pretrained=False, spatial_coord=True,
                           screen_size=(64, 64)).eval()

    def test_accepts_numpy_array(self):
        """GUI 传的就是 numpy 数组；修复前会抛
        `TypeError: conv2d() received numpy.ndarray`。"""
        model = self._model()
        frame = np.random.randint(0, 255, (90, 160, 3), dtype=np.uint8)
        result = model.predict(frame)
        self.assertIn("top_actions", result)
        self.assertEqual(len(result["mouse_position"]), 2)

    def test_accepts_tensor_and_resizes(self):
        model = self._model()
        result = model.predict(torch.rand(3, 300, 500))
        self.assertEqual(len(result["mouse_position"]), 2)

    def test_rejects_unsupported_type(self):
        model = self._model()
        with self.assertRaises(TypeError):
            model.predict("not an image")

    def test_coordinates_are_normalised_not_scaled(self):
        """坐标必须是 [0,1] 归一化值。

        旧实现对 spatial_coord 模型会再套一次 sigmoid（把坐标压向 0.5），
        然后乘上 ``screen_size``（模型输入 224，不是真实屏幕），结果毫无意义。
        """
        model = self._model()
        result = model.predict(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        x, y = result["mouse_position"]
        self.assertGreaterEqual(x, 0.0)
        self.assertLessEqual(x, 1.0)
        self.assertGreaterEqual(y, 0.0)
        self.assertLessEqual(y, 1.0)

    def test_action_names_are_resolved_when_given(self):
        model = self._model()
        names = ["left_click", "right_click", "double_click", "scroll_page"]
        result = model.predict(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                               action_names=names)
        self.assertIn(result["top_actions"][0]["action_name"], names)

    def test_segmentation_omitted_by_default(self):
        """默认不算分割：该分支代价约等于再跑一遍主干，而训练数据没有分割标签。"""
        model = self._model()
        result = model.predict(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        self.assertNotIn("segmentation_map", result)
        result_with = model.predict(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                                    with_segmentation=True)
        self.assertIn("segmentation_map", result_with)


class ScreenPredictorTest(unittest.TestCase):
    """统一推理入口：发现模型、加载、预测。"""

    def _models_dir(self) -> Path:
        return ROOT / "models"

    def test_discover_returns_only_complete_models(self):
        from ai_operator.models.inference import discover_models

        models = discover_models(self._models_dir())
        if not models:
            self.skipTest("本地没有已训练模型")
        for info in models:
            self.assertTrue(info.checkpoint.is_file())
            self.assertTrue(info.label_map.is_file())
            self.assertGreater(info.num_actions, 0)

    def test_preference_order_puts_best_first(self):
        from ai_operator.models.inference import discover_models

        models = discover_models(self._models_dir())
        if not models:
            self.skipTest("本地没有已训练模型")
        names = [m.name for m in models]
        # desktop_v3_coord 是 README 里记录的最佳模型，应排在最前
        if "desktop_v3_coord" in names:
            self.assertEqual(names[0], "desktop_v3_coord")

    def test_missing_dir_returns_empty_not_crash(self):
        from ai_operator.models.inference import discover_models

        self.assertEqual(discover_models(Path("__no_such_dir__")), [])

    def test_unknown_name_lists_candidates(self):
        from ai_operator.models.inference import ScreenPredictor

        if not ScreenPredictor and False:  # pragma: no cover
            return
        try:
            ScreenPredictor.load(name="__no_such_model__")
        except FileNotFoundError as e:
            self.assertIn("__no_such_model__", str(e))
        else:
            models_dir = self._models_dir()
            from ai_operator.models.inference import discover_models
            if discover_models(models_dir):
                self.fail("指定不存在的模型名时应当报错")

    def test_predict_output_contract(self):
        from ai_operator.models.inference import ScreenPredictor

        try:
            predictor = ScreenPredictor.load()
        except FileNotFoundError:
            self.skipTest("本地没有已训练模型")

        from PIL import Image

        size = (640, 360)
        image = Image.new("RGB", size, (200, 200, 200))
        result = predictor.predict(image, screen_size=size)

        self.assertIn("action", result)
        self.assertIn("action_zh", result)
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        nx, ny = result["point"]
        self.assertTrue(0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0)
        # point_px 必须是归一化坐标乘真实屏幕尺寸
        px, py = result["point_px"]
        self.assertAlmostEqual(px, round(nx * size[0]), delta=1)
        self.assertAlmostEqual(py, round(ny * size[1]), delta=1)
        self.assertLessEqual(len(result["topk"]), 3)

    def test_accepts_numpy_and_matches_pil(self):
        """同内容下 numpy 与 PIL 输入应给出相同动作（回归：GUI 走 numpy 路径）。"""
        from ai_operator.models.inference import ScreenPredictor

        try:
            predictor = ScreenPredictor.load()
        except FileNotFoundError:
            self.skipTest("本地没有已训练模型")

        from PIL import Image

        image = Image.new("RGB", (320, 180), (120, 140, 160))
        pil_result = predictor.predict(image)
        np_result = predictor.predict(np.asarray(image))
        self.assertEqual(pil_result["action"], np_result["action"])

    def test_from_checkpoint_reads_num_actions_from_label_map(self):
        """类别数必须来自 label_map，而不是让调用方手写 —— 手写正是 GUI 曾经
        用 34 类随机模型去跑 12 类训练结果的原因。"""
        import json

        from ai_operator.models.inference import ScreenPredictor, discover_models

        models = discover_models(self._models_dir())
        if not models:
            self.skipTest("本地没有已训练模型")
        info = models[0]
        predictor = ScreenPredictor.from_checkpoint(info.checkpoint, info.label_map)
        label_map = json.loads(info.label_map.read_text(encoding="utf-8"))
        self.assertEqual(predictor.info.num_actions, len(label_map))
        self.assertEqual(len(predictor.actions), len(label_map))
        self.assertTrue(all(predictor.actions), "每个类别索引都应映射到动作名")

    def test_from_checkpoint_missing_files(self):
        from ai_operator.models.inference import ScreenPredictor

        with self.assertRaises(FileNotFoundError):
            ScreenPredictor.from_checkpoint(Path("__no__.pth"), Path("__no__.json"))


class NoPhantomControllerMethodsTest(unittest.TestCase):
    """控制器上被调用但不存在的方法 —— 静态检查全部模块。

    这个检查抓出过 8 处真实调用：``WritingWorkflow`` 调用了
    ``keyboard_controller.press_shortcut()`` 与 ``press_key()``，
    而 ``KeyboardController`` 提供的其实是 ``hotkey(*keys)`` 与 ``press(key)``。
    结果 `--mode workflow` 一运行就崩在 AttributeError 上。

    这类错误单元测试通常覆盖不到（控制器需要真实键鼠环境才会被实例化调用），
    所以用 AST 做静态核对。
    """

    #: 属性名 -> (模块, 类名)
    CONTROLLERS = {
        "keyboard_controller": ("ai_operator.automation.keyboard_controller", "KeyboardController"),
        "mouse_controller": ("ai_operator.automation.mouse_controller", "AdvancedMouseController"),
        "web_controller": ("ai_operator.automation.web_controller", "WebController"),
        "multi_app_controller": ("ai_operator.automation.web_controller", "MultiAppController"),
        "workflow_manager": ("ai_operator.automation.workflow_manager", "WorkflowManager"),
        "smart_decision_engine": ("ai_operator.automation.workflow_manager", "SmartDecisionEngine"),
        "advanced_content_generator": (
            "ai_operator.automation.advanced_content_generator", "AdvancedContentGenerator"),
        "smart_error_handler": ("ai_operator.automation.advanced_content_generator", "SmartErrorHandler"),
        "performance_monitor": ("ai_operator.automation.advanced_content_generator", "PerformanceMonitor"),
    }

    def test_no_calls_to_missing_controller_methods(self):
        import ast
        import importlib

        api = {}
        for attr, (module_name, class_name) in self.CONTROLLERS.items():
            try:
                module = importlib.import_module(module_name)
                api[attr] = set(dir(getattr(module, class_name)))
            except Exception:
                continue  # 环境缺依赖时跳过该控制器

        self.assertTrue(api, "没有成功导入任何控制器，检查无意义")

        problems = []
        for path in sorted((ROOT / "ai_operator").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute):
                    continue
                base = func.value
                # 匹配 self.<controller_attr>.<method>(...)
                if not isinstance(base, ast.Attribute):
                    continue
                if not (isinstance(base.value, ast.Name) and base.value.id == "self"):
                    continue
                known = api.get(base.attr)
                if known is not None and func.attr not in known:
                    problems.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} "
                        f"{base.attr}.{func.attr}() 不存在"
                    )

        self.assertEqual(problems, [], "存在对不存在方法的调用：\n" + "\n".join(problems))


class GuiIntegrationTest(unittest.TestCase):
    """GUI 必须能在没有 API key、且有已训练模型的情况下真正构建起来。"""

    # 必须持有 QApplication 的强引用：若只在函数里创建，局部变量出作用域后
    # Qt 会把它回收，随后构造 QWidget 就会报
    # "QWidget: Must construct a QApplication before a QWidget"。
    _qapp = None

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt6.QtWidgets import QApplication
        except Exception:
            raise unittest.SkipTest("PyQt6 不可用")

    def _app(self):
        from PyQt6.QtWidgets import QApplication

        if GuiIntegrationTest._qapp is None:
            GuiIntegrationTest._qapp = QApplication.instance() or QApplication([])
        return GuiIntegrationTest._qapp

    def test_builds_without_api_key_and_loads_trained_model(self):
        """修复前：缺 key => OpenAIError；有 key => 仍旧用随机初始化模型。"""
        import ai_operator.ui.main_window as mw

        self._app()

        saved = mw.DeepSeekAgent

        class _UnavailableAgent:
            def __init__(self, *a, **k):
                pass

            def is_available(self):
                return False

            def process_task(self, *a, **k):
                return {"task_analysis": "stub"}

        mw.DeepSeekAgent = _UnavailableAgent
        try:
            window = mw.MainWindow()
        finally:
            mw.DeepSeekAgent = saved

        # 无密钥时必须给出明确提示，而不是静默
        self.assertIn("未配置密钥", window.agent_status_label.text())

        models = mw.discover_models(ROOT / "models")
        if models:
            self.assertIsNotNone(window.predictor, "有可用模型时应当自动加载")
            self.assertEqual(window.model_combo.count(), len(models))
            # 预测器类别数必须与该模型的 label_map 一致
            self.assertEqual(window.predictor.info.num_actions, models[0].num_actions)
        else:
            self.assertIsNone(window.predictor)
            self.assertFalse(window.model_combo.isEnabled())
        window.close()

    def test_prediction_display_is_human_readable(self):
        """界面上应显示动作名与像素坐标，而不是一串类别整数索引。"""
        import ai_operator.ui.main_window as mw

        self._app()
        window = mw.MainWindow()
        if window.predictor is None:
            window.close()
            self.skipTest("本地没有已训练模型")

        window.update_prediction_display({
            "model": "demo",
            "action": "click_tab",
            "action_zh": "点击标签页",
            "confidence": 0.98,
            "point": (0.25, 0.5),
            "point_px": (400, 300),
            "topk": [{"index": 0, "action": "click_tab", "action_zh": "点击标签页", "confidence": 0.98}],
        })
        text = window.prediction_text.toPlainText()
        self.assertIn("click_tab", text)
        self.assertIn("点击标签页", text)
        self.assertIn("(400, 300)", text)
        window.close()


if __name__ == "__main__":
    unittest.main()
