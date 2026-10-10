"""桌面数据集与坐标解码的单元测试。

覆盖三处容易悄悄坏掉的地方：
- 数据集目录 / 标注格式的解析（``ai_operator_dataset`` 的目录约定）
- 光标注入是否真的改变了像素、且不移动其他内容
- 空间坐标头与旧版全局池化头在 state_dict 上能否正确区分

运行::

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai_operator.data.desktop_dataset import (  # noqa: E402
    DesktopSample,
    DesktopScreenshotDataset,
    build_label_map,
    cursor_points,
    draw_cursor,
    scan_split,
)
from ai_operator.models.cnn_model import (  # noqa: E402
    EnhancedCNN,
    decode_points,
    spatial_coord_from_state,
)


def _make_creator(tmp: str):
    """构造一个输出到临时目录的生成器（必须走 __init__，否则 self.themes 不存在）。"""
    from ai_operator.data_generation.dataset_creator import ProfessionalDatasetCreator

    return ProfessionalDatasetCreator(tmp)


class GeneratorCausalityTest(unittest.TestCase):
    """数据集生成器的动作必须由元素类型唯一决定。

    这一组测试针对的是一个曾经存在的严重 bug：``generate_app_specific_operation()``
    先 ``random.choice`` 选动作、再独立 ``random.choice`` 选元素，导致动作标签在给定
    截图下完全不可预测（模型上限退化为"猜最常见的类"）。下面的测试会在该 bug 复现时失败。
    """

    def test_resolve_action_is_pure_function(self):
        from ai_operator.data_generation.dataset_creator import UIElement, resolve_action

        cases = [
            (None, "icon", "double_click"),
            (None, "button", "left_click"),
            # button 在所有应用里都必须是同一个动作，不得碎片化标签空间
            ("web_browser", "button", "left_click"),
            ("file_explorer", "button", "left_click"),
            ("text_editor", "editor", "type_text"),
            (None, "editor", "click_editor"),
            (None, "input", "click_address_bar"),
            (None, "file", "click_file"),
            (None, "folder", "click_folder"),
        ]
        for app_type, kind, expected in cases:
            element = UIElement(element_type=kind, bounds=(0, 0, 10, 10))
            self.assertEqual(resolve_action(app_type, element), expected,
                             f"{app_type}/{kind} 的动作不符合预期")

        # 状态栏没有操作语义，不得被选为目标
        statusbar = UIElement(element_type="statusbar", bounds=(0, 0, 10, 10))
        self.assertIsNone(resolve_action("text_editor", statusbar))

    def test_app_specific_operation_action_follows_element(self):
        """反复采样：同一元素类型必须始终得到同一动作。"""
        from collections import defaultdict

        with tempfile.TemporaryDirectory() as tmp:
            creator = _make_creator(tmp)
            for app_type in ("web_browser", "text_editor", "file_explorer"):
                _, elements = creator.create_app_interface(app_type)
                mapping = defaultdict(set)
                for _ in range(300):
                    operation = creator.generate_app_specific_operation(app_type, elements)
                    mapping[operation["target_element"]].add(operation["action"])
                for element_type, actions in mapping.items():
                    self.assertEqual(len(actions), 1,
                                     f"{app_type}/{element_type} 出现了多个动作 "
                                     f"{sorted(actions)} —— 动作又变成随机的了")

    def test_generated_basic_operation_action_follows_element(self):
        from collections import defaultdict

        with tempfile.TemporaryDirectory() as tmp:
            creator = _make_creator(tmp)
            _, elements = creator.create_desktop_screenshot("windows_light", (1280, 720))
            mapping = defaultdict(set)
            for _ in range(300):
                operation = creator.generate_random_operation(elements)
                mapping[operation["target_element"]].add(operation["action"])
            for element_type, actions in mapping.items():
                self.assertEqual(len(actions), 1,
                                 f"desktop/{element_type} 出现了多个动作 {sorted(actions)}")

    def test_every_target_element_has_action(self):
        """每个可能被选中的元素都必须能解析出动作，不能出现 None/idle。"""
        with tempfile.TemporaryDirectory() as tmp:
            creator = _make_creator(tmp)
            for app_type in ("web_browser", "text_editor", "file_explorer"):
                _, elements = creator.create_app_interface(app_type)
                for _ in range(100):
                    operation = creator.generate_app_specific_operation(app_type, elements)
                    self.assertNotIn(operation["action"], (None, "idle"),
                                     f"{app_type} 生成了无动作的样本: {operation}")

    def test_generated_app_screenshots_are_not_blank(self):
        """回归：create_app_interface 曾经漏掉绘制调用，产出纯色空白图。"""
        with tempfile.TemporaryDirectory() as tmp:
            creator = _make_creator(tmp)
            for app_type in ("web_browser", "text_editor", "file_explorer"):
                image, elements = creator.create_app_interface(app_type)
                arr = np.asarray(image.convert("L"))
                self.assertGreater(float(arr.std()), 1.0,
                                   f"{app_type} 的截图是纯色空白图（绘制调用可能又丢了）")
                self.assertGreater(len(elements), 0)

    def test_save_sample_draws_cursor_at_target(self):
        """保存样本时应把光标画到目标元素中心，使坐标在画面上可见。"""
        import json

        from PIL import Image as _Image

        with tempfile.TemporaryDirectory() as tmp:
            creator = _make_creator(tmp)
            image = _Image.new("RGB", (400, 300), (200, 200, 200))
            operation = {"action": "left_click", "target_element": "button",
                         "target_bounds": [100, 50, 200, 100], "target_text": ""}
            creator.save_sample("s1", image, operation, [])

            saved = _Image.open(f"{tmp}/samples/s1/screenshot.png").convert("L")
            arr = np.asarray(saved)
            self.assertGreater(float(arr.std()), 1.0, "光标没有被画上")

            # 变化像素（黑色描边）应集中在目标框中心 (150, 75) 附近
            changed = np.argwhere(arr < 150)
            self.assertGreater(len(changed), 0)
            self.assertLess(abs(int(changed[:, 1].mean()) - 150), 25, "光标没有落在目标 x")
            self.assertLess(abs(int(changed[:, 0].mean()) - 75), 25, "光标没有落在目标 y")

            with open(f"{tmp}/samples/s1/s1.json", encoding="utf-8") as f:
                annotation = json.load(f)
            self.assertTrue(annotation["operation"]["cursor_drawn"])

    def test_cursor_can_be_disabled_for_ablation(self):
        """draw_cursor=False 时必须保持纯背景，供对照实验使用。"""
        from PIL import Image as _Image

        with tempfile.TemporaryDirectory() as tmp:
            creator = _make_creator(tmp)
            creator.draw_cursor = False
            image = _Image.new("RGB", (400, 300), (200, 200, 200))
            operation = {"action": "left_click", "target_bounds": [100, 50, 200, 100]}
            creator.save_sample("s2", image, operation, [])
            arr = np.asarray(_Image.open(f"{tmp}/samples/s2/screenshot.png").convert("L"))
            self.assertLess(float(arr.std()), 1e-6, "关闭光标后仍出现了结构")


class CursorDrawingTest(unittest.TestCase):
    """:func:`draw_cursor` 的行为。"""

    def test_cursor_changes_pixels_only_near_tip(self):
        canvas = np.full((64, 64, 3), 200, dtype=np.uint8)
        drawn = draw_cursor(canvas, 32, 32, size=12)

        self.assertEqual(drawn.shape, canvas.shape)
        # 原数组不能被就地修改
        self.assertTrue((canvas == 200).all())
        # 尖端附近必须变化
        self.assertFalse((drawn[32:44, 32:44] == 200).all())
        # 远端角落必须保持原样（光标不能泄漏到别处，否则坐标标签就不可信了）
        self.assertTrue((drawn[0:8, 0:8] == 200).all())

    def test_cursor_tip_follows_target(self):
        """不同目标点画出的光标必须落在不同位置 —— 这是坐标回归可解的前提。"""
        canvas = np.full((64, 64, 3), 200, dtype=np.uint8)
        left = draw_cursor(canvas, 12, 12, size=12)
        right = draw_cursor(canvas, 48, 48, size=12)

        left_changed = np.argwhere((left != 200).any(axis=2))
        right_changed = np.argwhere((right != 200).any(axis=2))
        self.assertLess(left_changed[:, 0].min(), 16)
        self.assertGreater(right_changed[:, 0].min(), 40)

    def test_cursor_points_scale_with_size(self):
        small = cursor_points(10, 10, size=15)
        large = cursor_points(10, 10, size=30)
        self.assertEqual(len(small), len(large))
        self.assertAlmostEqual(large[1][1] - large[0][1], 2 * (small[1][1] - small[0][1]))


class DatasetParsingTest(unittest.TestCase):
    """:func:`scan_split` 的目录/标注解析。"""

    def _make_fake_split(self, root):
        from PIL import Image

        ann_dir = os.path.join(root, "train", "annotations")
        img_dir = os.path.join(root, "train", "images", "basic_000001")
        os.makedirs(ann_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        Image.new("RGB", (200, 100), (255, 255, 255)).save(os.path.join(img_dir, "screenshot.png"))
        import json

        with open(os.path.join(ann_dir, "basic_000001.json"), "w", encoding="utf-8") as f:
            json.dump({
                "sample_id": "basic_000001",
                "operation": {"action": "double_click", "target_bounds": [50, 20, 150, 80]},
            }, f)

        # 缺 target_bounds 的坏样本应被跳过，而不是让整次训练崩掉
        with open(os.path.join(ann_dir, "broken.json"), "w", encoding="utf-8") as f:
            json.dump({"sample_id": "broken", "operation": {"action": "left_click"}}, f)

    def test_scan_skips_invalid_and_centers_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_fake_split(tmp)
            samples = scan_split(tmp, "train")
            self.assertEqual(len(samples), 1)
            sample = samples[0]
            self.assertEqual(sample.action, "double_click")
            self.assertEqual(sample.image_size, (200, 100))
            # 目标点 = 边界框中心
            self.assertEqual(sample.point, (100.0, 50.0))
            # 归一化：u = x / W, v = y / H
            self.assertAlmostEqual(sample.norm_point[0], 0.5)
            self.assertAlmostEqual(sample.norm_point[1], 0.5)

    def test_missing_split_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(scan_split(tmp, "train"), [])


class DatasetCursorTest(unittest.TestCase):
    """数据集层面：标签必须与画出的光标位置一致。"""

    def test_label_point_matches_cursor_position(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "train", "annotations"))
            img_dir = os.path.join(tmp, "train", "images", "basic_000002")
            os.makedirs(img_dir)
            # 平坦底色，这样任何非底色像素都只可能来自光标
            Image.new("RGB", (200, 100), (128, 128, 128)).save(os.path.join(img_dir, "screenshot.png"))
            import json

            # 目标框中心 = (150, 25) -> 归一化 (0.75, 0.25)
            with open(os.path.join(tmp, "train", "annotations", "basic_000002.json"), "w", encoding="utf-8") as f:
                json.dump({"sample_id": "basic_000002",
                           "operation": {"action": "left_click",
                                         "target_bounds": [140, 20, 160, 30]}}, f)

            samples = scan_split(tmp, "train")
            label_map = build_label_map(samples)
            size = 128
            ds = DesktopScreenshotDataset(samples, label_map, size, preload=True,
                                          verbose=False, draw_cursor=True,
                                          mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
            item = ds[0]
            arr = item["image"].permute(1, 2, 0).numpy()

            # 缩放到 128 后，光标尖端应落在 (0.75, 0.25) * 128
            expected = (int(0.75 * size), int(0.25 * size))
            changed = np.argwhere(np.abs(arr - arr[0, 0]).sum(axis=2) > 0.05)
            self.assertGreater(len(changed), 0, "光标没有画上去")
            # 光标最上方的像素应贴近尖端所在行
            self.assertLessEqual(abs(int(changed[:, 0].min()) - expected[1]), 3)

            # 归一化坐标标签本身
            self.assertAlmostEqual(float(item["point"][0]), 0.75, places=5)
            self.assertAlmostEqual(float(item["point"][1]), 0.25, places=5)

    def test_cursor_disabled_leaves_flat_image(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "train", "annotations"))
            img_dir = os.path.join(tmp, "train", "images", "basic_000003")
            os.makedirs(img_dir)
            Image.new("RGB", (200, 100), (128, 128, 128)).save(os.path.join(img_dir, "screenshot.png"))
            import json

            with open(os.path.join(tmp, "train", "annotations", "basic_000003.json"), "w", encoding="utf-8") as f:
                json.dump({"sample_id": "basic_000003",
                           "operation": {"action": "left_click",
                                         "target_bounds": [10, 10, 30, 30]}}, f)

            samples = scan_split(tmp, "train")
            size = 64
            ds = DesktopScreenshotDataset(samples, build_label_map(samples), size, preload=True,
                                          verbose=False, draw_cursor=False,
                                          mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
            arr = ds[0]["image"].numpy()
            self.assertLess(float(arr.std()), 1e-6, "未开启光标却出现了结构")


class SpatialCoordHeadTest(unittest.TestCase):
    """空间坐标头 vs 旧版全局池化头的接口一致性。"""

    def test_old_and_new_heads_shapes(self):
        for spatial in (False, True):
            model = EnhancedCNN(4, pretrained=False, spatial_coord=spatial).eval()
            with torch.no_grad():
                out = model(torch.randn(2, 3, 64, 64), return_segmentation=False)
            self.assertEqual(tuple(out["actions"].shape), (2, 4))
            self.assertEqual(tuple(out["mouse_positions"].shape), (2, 2))
            self.assertIsNone(out["segmentation"])

    def test_decode_points_contract(self):
        """decode_points 必须总是给出 [0, 1] 内的坐标，且两条分支语义不同。"""
        plain = EnhancedCNN(3, pretrained=False, spatial_coord=False).eval()
        raw = torch.tensor([[2.0, -2.0]])
        decoded = decode_points(plain, raw)
        # 全局池化头走 sigmoid
        self.assertTrue(torch.allclose(decoded, torch.sigmoid(raw)))

        spatial = EnhancedCNN(3, pretrained=False, spatial_coord=True).eval()
        already_normalised = torch.tensor([[0.25, 0.75]])
        self.assertTrue(torch.allclose(decode_points(spatial, already_normalised), already_normalised))

    def test_state_dict_detection(self):
        plain = EnhancedCNN(3, pretrained=False, spatial_coord=False)
        spatial = EnhancedCNN(3, pretrained=False, spatial_coord=True)
        self.assertFalse(spatial_coord_from_state(plain.state_dict()))
        self.assertTrue(spatial_coord_from_state(spatial.state_dict()))

    def test_old_checkpoint_still_loads(self):
        """旧 checkpoint（无 coord_head）必须还能载入，否则历史模型全部作废。"""
        old = EnhancedCNN(5, pretrained=False, spatial_coord=False)
        fresh = EnhancedCNN(5, pretrained=False, spatial_coord=False)
        fresh.load_state_dict(old.state_dict())
        self.assertFalse(any(k.startswith("coord_head.") for k in old.state_dict()))

    def test_segmentation_branch_shape(self):
        # 分割分支插值到 screen_size，所以构造时两者要一致
        model = EnhancedCNN(3, pretrained=False, spatial_coord=True,
                            screen_size=(64, 64)).eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 64, 64), return_segmentation=True)
        self.assertEqual(tuple(out["segmentation"].shape), (1, 10, 64, 64))


class SpatialCoordGradientTest(unittest.TestCase):
    """soft-argmax 的梯度必须能流回坐标头 —— 否则坐标永远学不动。"""

    def test_gradient_reaches_coord_head(self):
        model = EnhancedCNN(3, pretrained=False, spatial_coord=True)
        model.train()
        out = model(torch.randn(2, 3, 64, 64), return_segmentation=False)
        target = torch.tensor([[0.2, 0.3], [0.8, 0.7]])
        loss = torch.nn.functional.smooth_l1_loss(decode_points(model, out["mouse_positions"]), target)
        loss.backward()

        grads = [p.grad for p in model.coord_head.parameters() if p.grad is not None]
        self.assertTrue(grads, "coord_head 没有收到任何梯度")
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in grads), "coord_head 梯度全为 0")

    def test_soft_argmax_can_overfit_one_point(self):
        """给足步数后应能把坐标推向目标 —— 证明这条路径确实可学。"""
        torch.manual_seed(0)
        model = EnhancedCNN(3, pretrained=False, spatial_coord=True)
        model.train()
        # batch 必须是 2 以上：action_head 里的 BatchNorm1d 在训练态拒绝单样本
        x = torch.randn(2, 3, 64, 64)
        target = torch.tensor([[0.8, 0.2], [0.8, 0.2]])
        optimizer = torch.optim.Adam(model.coord_head.parameters(), lr=1e-3)

        first = last = None
        for step in range(60):
            optimizer.zero_grad()
            out = model(x, return_segmentation=False)
            loss = torch.nn.functional.smooth_l1_loss(decode_points(model, out["mouse_positions"]), target)
            loss.backward()
            optimizer.step()
            if step == 0:
                first = float(loss.item())
            last = float(loss.item())

        self.assertLess(last, first * 0.5, f"坐标损失没有明显下降: {first:.5f} -> {last:.5f}")

    def test_global_pooled_head_cannot_localise(self):
        """反向证明：旧版全局池化头收到的特征图空间维度已为 1x1，无法定位。"""
        model = EnhancedCNN(3, pretrained=False, spatial_coord=False).eval()
        captured = {}

        def hook(_module, _inputs, output):
            captured["feature_map"] = tuple(output.shape)

        model.resnet.layer4.register_forward_hook(hook)
        with torch.no_grad():
            model(torch.randn(1, 3, 64, 64), return_segmentation=False)

        # layer4 本身是有空间的（2x2），说明信息在进入 mouse_head 之前就存在
        self.assertEqual(len(captured["feature_map"]), 4)
        self.assertGreater(captured["feature_map"][2], 1)

        # 而 mouse_head 的第 0 层收到的是被压平的全局向量
        received = {}

        def head_hook(_module, inputs, _output):
            received["input"] = tuple(inputs[0].shape)

        model.mouse_head[0].register_forward_hook(head_hook)
        with torch.no_grad():
            model(torch.randn(2, 3, 64, 64), return_segmentation=False)
        self.assertEqual(len(received["input"]), 2, "mouse_head 应该只收到 2D 的全局特征")


if __name__ == "__main__":
    unittest.main()
