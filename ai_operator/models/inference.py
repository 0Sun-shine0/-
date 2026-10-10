"""
训练模型的统一推理入口
======================

在此之前，"加载训练好的模型并预测"这件事在仓库里有两份各写各的实现
（``scripts/demo_desktop_agent.py`` 与 GUI 里的随机初始化模型），
而 GUI 那份**根本不加载任何权重**。本模块把它收敛成唯一实现，
供 GUI、演示脚本与新代码共用。

职责：

- 扫描 ``models/*/``，找出真正可用的"checkpoint + label_map"组合
- 从 ``metrics.json`` / 权重形状推断输入尺寸与坐标头类型，无需调用方手写配置
- 把预测结果翻译成人可读的形式：动作名、中文说明、归一化坐标与屏幕像素坐标

用法::

    from ai_operator.models.inference import ScreenPredictor

    predictor = ScreenPredictor.load()           # 自动挑最佳可用模型
    result = predictor.predict(pil_image, screen_size=(1920, 1080))
    print(result["action"], result["action_zh"], result["point_px"])
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data.desktop_dataset import IMAGENET_MEAN, IMAGENET_STD, action_zh
from .cnn_model import EnhancedCNN, decode_points, spatial_coord_from_state

#: 按此顺序优先选择默认模型（越靠前越优先）
DEFAULT_PREFERENCE: Sequence[str] = (
    "desktop_v3_coord",
    "desktop_v2",
    "desktop_v3",
    "desktop_v1_cursor",
    "desktop_run",
)


@dataclass
class ModelInfo:
    """一个可用模型的元信息。"""

    name: str
    checkpoint: Path
    label_map: Path
    num_actions: int
    input_size: int
    spatial_coord: bool
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def display(self) -> str:
        """给下拉框用的展示名，带关键指标。"""
        acc = self.metrics.get("best_acc")
        coord = self.metrics.get("best_coord_mae_px")
        bits = [f"{self.num_actions}类"]
        if isinstance(acc, (int, float)):
            bits.append(f"准确率{acc:.0%}")
        if isinstance(coord, (int, float)):
            bits.append(f"坐标{coord:.0f}px")
        return f"{self.name}（{', '.join(bits)}）"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def discover_models(models_dir: Path) -> List[ModelInfo]:
    """扫描目录，返回可用于推理的模型列表（按偏好排序）。

    只有同时具备 ``best_model.pth`` 与 ``label_map.json`` 的目录才算可用 ——
    缺了 label_map 就无法把类别索引翻译成动作名，预测结果没有意义。
    """
    if not models_dir.is_dir():
        return []

    found: List[ModelInfo] = []
    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        checkpoint = child / "best_model.pth"
        label_map_path = child / "label_map.json"
        if not (checkpoint.is_file() and label_map_path.is_file()):
            continue

        label_map = _read_json(label_map_path)
        if not label_map:
            continue
        metrics = _read_json(child / "metrics.json")

        # 坐标头类型从权重键名判断，避免用户手动对齐配置
        try:
            state = torch.load(checkpoint, map_location="cpu")
        except Exception:
            continue
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        spatial = spatial_coord_from_state(state)

        input_size = int(metrics.get("input_size") or 224)
        found.append(ModelInfo(
            name=child.name,
            checkpoint=checkpoint,
            label_map=label_map_path,
            num_actions=len(label_map),
            input_size=input_size,
            spatial_coord=spatial,
            metrics=metrics,
        ))

    order = {name: i for i, name in enumerate(DEFAULT_PREFERENCE)}
    found.sort(key=lambda info: (order.get(info.name, len(order)), info.name))
    return found


class ScreenPredictor:
    """加载一个训练好的模型，并对截图给出可读的预测结果。"""

    def __init__(self, info: ModelInfo, device: Optional[torch.device] = None) -> None:
        self.info = info
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        label_map = _read_json(info.label_map)
        self.label_map: Dict[str, int] = {str(k): int(v) for k, v in label_map.items()}
        # 索引 -> 动作名
        self.actions: List[str] = [""] * len(self.label_map)
        for name, idx in self.label_map.items():
            if 0 <= idx < len(self.actions):
                self.actions[idx] = name

        self.model = EnhancedCNN(
            info.num_actions,
            screen_size=(info.input_size, info.input_size),
            pretrained=False,
            spatial_coord=info.spatial_coord,
        )

        state = torch.load(info.checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, models_dir: Optional[Path] = None, name: Optional[str] = None,
             device: Optional[torch.device] = None) -> "ScreenPredictor":
        """加载模型。

        Args:
            models_dir: 模型根目录，默认项目根的 ``models/``。
            name: 指定模型目录名；None 表示按 :data:`DEFAULT_PREFERENCE` 自动选。

        Raises:
            FileNotFoundError: 没有任何可用模型，或指定的名字不存在。
                错误信息里会列出实际找到的候选，便于排查。
        """
        if models_dir is None:
            models_dir = Path(__file__).resolve().parents[2] / "models"
        models = discover_models(models_dir)
        if not models:
            raise FileNotFoundError(
                f"在 {models_dir} 下没有找到可用模型。\n"
                "可用模型需要同时包含 best_model.pth 与 label_map.json。\n"
                "请先训练，例如：\n"
                "  python scripts/build_dataset_v2.py --out ai_operator_dataset_v2\n"
                "  python scripts/train_ai_operator.py --data-root ai_operator_dataset_v2 "
                "--out-dir models/desktop_v2 --spatial-coord --monitor coord"
            )
        if name:
            for info in models:
                if info.name == name:
                    return cls(info, device)
            raise FileNotFoundError(
                f"未找到名为 {name!r} 的模型。实际可用：{[m.name for m in models]}"
            )
        return cls(models[0], device)

    @classmethod
    def from_checkpoint(cls, checkpoint, label_map, input_size: Optional[int] = None,
                        device: Optional[torch.device] = None) -> "ScreenPredictor":
        """从显式路径加载（供 `--model/--label-map` 这类命令行用法）。

        坐标头类型从权重键名自动判断，类别数从 label_map 推断，因此调用方
        不必也不应该手写这两个参数 —— 手写正是 GUI 与评估结果脱节的根源。
        """
        checkpoint = Path(checkpoint)
        label_map_path = Path(label_map)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"找不到 checkpoint: {checkpoint}")
        if not label_map_path.is_file():
            raise FileNotFoundError(f"找不到 label_map: {label_map_path}")

        mapping = _read_json(label_map_path)
        if not mapping:
            raise ValueError(f"label_map 为空或无法解析: {label_map_path}")

        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]

        metrics_path = checkpoint.parent / "metrics.json"
        metrics = _read_json(metrics_path)
        info = ModelInfo(
            name=checkpoint.parent.name,
            checkpoint=checkpoint,
            label_map=label_map_path,
            num_actions=len(mapping),
            input_size=int(input_size or metrics.get("input_size") or 224),
            spatial_coord=spatial_coord_from_state(state),
            metrics=metrics,
        )
        return cls(info, device)

    # ------------------------------------------------------------------ #
    def _to_tensor(self, image) -> torch.Tensor:
        """PIL / numpy(BGR 或 RGB) / 张量 -> (1,3,S,S) 的 ImageNet 归一化张量。

        缩放必须用 ``Image.BILINEAR``，与 ``DesktopScreenshotDataset`` 的预处理
        完全一致；若用 Pillow 的默认重采样（新版为 BICUBIC），像素会与训练/评估时
        略有差异，实测坐标会偏移 1~3 px，界面显示就与 README 里的指标对不上了。
        """
        from PIL import Image

        size = (self.info.input_size, self.info.input_size)
        if isinstance(image, Image.Image):
            arr = np.asarray(image.convert("RGB").resize(size, Image.BILINEAR), dtype=np.uint8)
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            # 传进来的若是 BGR（OpenCV 默认），按 RGB 处理会偏色；界面截图的通道
            # 顺序差异对动作/位置预测影响很小，需要严格一致时请调用方先转 RGB。
            arr = np.asarray(
                Image.fromarray(arr.astype(np.uint8)).convert("RGB").resize(size, Image.BILINEAR),
                dtype=np.uint8,
            )
        elif isinstance(image, torch.Tensor):
            tensor = image.detach().float()
            if tensor.max() > 1.5:
                tensor = tensor / 255.0
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[1] != 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(0, 3, 1, 2)
            return tensor.to(self.device)
        else:
            raise TypeError(f"不支持的图像类型: {type(image)!r}")

        arr = arr.astype(np.float32) / 255.0
        arr = (arr - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image, topk: int = 3,
                screen_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """预测单张截图。

        Args:
            image: PIL 图像 / numpy 数组 / 张量。
            topk: 返回前 k 个动作。
            screen_size: 真实屏幕 ``(宽, 高)``。给出后结果里会带上
                ``point_px``（真实像素坐标）；不给出则只有归一化坐标。

        Returns:
            含 ``action`` / ``action_zh`` / ``confidence`` / ``topk`` /
            ``point``（归一化） / ``point_px``（可选）的字典。
        """
        tensor = self._to_tensor(image)
        outputs = self.model(tensor, return_segmentation=False)
        probs = torch.softmax(outputs["actions"], dim=1)[0]
        point = decode_points(self.model, outputs["mouse_positions"])[0]

        k = max(1, min(int(topk), probs.numel()))
        values, indices = probs.topk(k)
        topk_list = [
            {
                "index": int(i.item()),
                "action": self.actions[i.item()] if i.item() < len(self.actions) else f"class_{i.item()}",
                "action_zh": action_zh(self.actions[i.item()]) if i.item() < len(self.actions) else "",
                "confidence": float(v.item()),
            }
            for i, v in zip(indices, values)
        ]

        nx, ny = float(point[0].item()), float(point[1].item())
        result: Dict[str, Any] = {
            "model": self.info.name,
            "action_index": topk_list[0]["index"],
            "action": topk_list[0]["action"],
            "action_zh": topk_list[0]["action_zh"],
            "confidence": topk_list[0]["confidence"],
            "topk": topk_list,
            "point": (nx, ny),
            "point_px": None,
        }
        if screen_size is not None:
            width, height = screen_size
            result["point_px"] = (int(round(nx * width)), int(round(ny * height)))
        return result


__all__ = ["DEFAULT_PREFERENCE", "ModelInfo", "ScreenPredictor", "discover_models"]
