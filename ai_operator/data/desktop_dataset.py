"""
真实桌面数据集加载器
====================

把 ``ai_operator_dataset/{split}/`` 下的"截图 + JSON 标注"转换成训练/推理需要的张量与标签。

数据集的目录约定::

    ai_operator_dataset/
    ├── train/
    │   ├── images/<sample_id>/screenshot.png
    │   └── annotations/<sample_id>.json
    ├── val/...
    └── test/...

单个标注文件的结构（节选）::

    {
      "sample_id": "basic_000000",
      "operation": {
        "action": "right_click",
        "target_bounds": [1234, 550, 1294, 610]   # [x1, y1, x2, y2] 原图像素坐标
      },
      "elements": [ ... ]
    }

本模块负责：

1. 读取标注，产出 **动作类别标签**（``operation.action``）与
   **目标点归一化坐标**（``target_bounds`` 的中心点 / 图像宽高）。
2. 按 ImageNet 统计量做归一化（骨干网络是 ImageNet 预训练的 ResNet50，
   归一化方式必须与之匹配，否则预训练权重的收益会大幅衰减）。
3. 训练时只做不改变目标点的增强（颜色抖动），避免破坏坐标标签。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# ImageNet 归一化参数（与 torchvision 预训练 ResNet 对齐）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: 动作名称 -> 中文说明，便于演示时阅读
ACTION_ZH: Dict[str, str] = {
    "click_address_bar": "点击地址栏",
    "click_button": "点击按钮",
    "click_editor": "点击编辑器",
    "click_file": "点击文件",
    "click_folder": "点击文件夹",
    "click_link": "点击链接",
    "click_menu": "点击菜单",
    "click_sidebar": "点击侧边栏",
    "click_tab": "点击标签页",
    "click_toolbar": "点击工具栏",
    "double_click": "双击",
    "left_click": "左键单击",
    "right_click": "右键单击",
    "scroll_list": "滚动列表",
    "scroll_page": "滚动页面",
    "type_text": "输入文本",
}


def action_zh(action: str) -> str:
    """返回动作的中文说明，未登记的动作原样返回。"""
    return ACTION_ZH.get(action, action)


@dataclass
class DesktopSample:
    """一条桌面操作样本（尚未加载像素）。"""

    sample_id: str
    image_path: str
    action: str
    #: 目标点像素坐标 (x, y)
    point: Tuple[float, float]
    #: 目标框 [x1, y1, x2, y2] 像素坐标
    bounds: Tuple[float, float, float, float]
    image_size: Tuple[int, int] = (0, 0)  # (width, height)

    @property
    def norm_point(self) -> Tuple[float, float]:
        """归一化到 [0, 1] 的目标点。"""
        w, h = self.image_size
        if w <= 0 or h <= 0:
            return (0.0, 0.0)
        return (
            min(max(self.point[0] / w, 0.0), 1.0),
            min(max(self.point[1] / h, 0.0), 1.0),
        )


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_image(split_dir: Path, sample_id: str, declared: Optional[str]) -> Optional[Path]:
    """定位样本截图：优先按 sample_id 目录查找，其次回退到标注里声明的相对路径。"""
    candidate = split_dir / "images" / sample_id / "screenshot.png"
    if candidate.exists():
        return candidate
    sample_dir = split_dir / "images" / sample_id
    if sample_dir.is_dir():
        for p in sorted(sample_dir.iterdir()):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                return p
    if declared:
        # 标注里通常是 "samples\\basic_000000\\screenshot.png" 这类相对路径
        tail = Path(declared.replace("\\", "/")).name
        guess = split_dir / "images" / sample_id / tail
        if guess.exists():
            return guess
    return None


def scan_split(root: str | os.PathLike, split: str, with_image_size: bool = True) -> List[DesktopSample]:
    """扫描一个 split，返回全部有效样本。

    Args:
        root: 数据集根目录（例如 ``ai_operator_dataset``）。
        split: ``train`` / ``val`` / ``test`` 之一。
        with_image_size: 是否读取图片尺寸（坐标归一化需要）。关闭可加快扫描。
    """
    split_dir = Path(root) / split
    ann_dir = split_dir / "annotations"
    if not ann_dir.is_dir():
        return []

    samples: List[DesktopSample] = []
    for ann_file in sorted(ann_dir.glob("*.json")):
        try:
            data = _read_json(ann_file)
        except (json.JSONDecodeError, OSError):
            continue

        sample_id = data.get("sample_id") or ann_file.stem
        operation = data.get("operation") or {}
        action = operation.get("action")
        bounds = operation.get("target_bounds")
        if not action or not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            continue

        image_path = _resolve_image(split_dir, sample_id, data.get("image_path"))
        if image_path is None:
            continue

        x1, y1, x2, y2 = (float(v) for v in bounds)
        point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        size = (0, 0)
        if with_image_size:
            try:
                with Image.open(image_path) as im:
                    size = im.size
            except OSError:
                continue

        samples.append(
            DesktopSample(
                sample_id=sample_id,
                image_path=str(image_path),
                action=str(action),
                point=point,
                bounds=(x1, y1, x2, y2),
                image_size=size,
            )
        )
    return samples


def build_label_map(samples: Sequence[DesktopSample]) -> Dict[str, int]:
    """按动作名称字典序构建稳定的 类别名 -> 索引 映射。"""
    return {name: idx for idx, name in enumerate(sorted({s.action for s in samples}))}


def save_label_map(label_map: Dict[str, int], path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)


def load_label_map(path: str | os.PathLike) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


#: 箭头光标在模型输入空间中的默认高度（像素）
CURSOR_SIZE = 15

#: 箭头光标的顶点，定义在 15x16 的归一化框内，尖端在 (0, 0)
_CURSOR_SHAPE: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.0), (0.0, 15.0), (3.5, 11.0), (6.0, 16.0),
    (9.0, 14.5), (6.5, 10.0), (11.0, 10.0),
)


def cursor_points(x: float, y: float, size: int = CURSOR_SIZE) -> List[Tuple[float, float]]:
    """返回箭头光标的顶点列表，尖端位于 ``(x, y)``。

    抽出来是为了让"实时注入"（:func:`draw_cursor`）和"离线烘烤进数据集"
    （``scripts/build_dataset_v2.py``）画出完全相同的光标形状。
    """
    scale = size / 15.0
    return [(x + px * scale, y + py * scale) for px, py in _CURSOR_SHAPE]


def draw_cursor(arr: np.ndarray, x: float, y: float, size: int = CURSOR_SIZE) -> np.ndarray:
    """在 uint8 RGB 数组上画一个箭头光标，箭头尖端位于 ``(x, y)``。

    数据集只保存了静态截图，既没有光标也没有高亮框，"用户点了哪里"在画面上完全不可见，
    坐标回归因此无法被训练或评估。真实桌面上光标恰好停在点击位置，所以这里把光标补回来，
    让任务变成"在截图里定位光标"——这与真实推理时的输入是一致的。

    Args:
        arr: ``(H, W, 3)`` 的 uint8 RGB 数组，会被复制而不是就地修改。
        x, y: 箭头尖端坐标（像素，输入空间中）。
        size: 光标高度（像素）。
    """
    from PIL import Image as _Image
    from PIL import ImageDraw

    points = cursor_points(x, y, size)

    canvas = _Image.fromarray(arr)
    painter = ImageDraw.Draw(canvas)
    painter.polygon(points, fill=(255, 255, 255), outline=(0, 0, 0))
    # PIL 的 polygon outline 在深色底上不够醒目，再描一圈黑边确保任何主题下都可见
    painter.line(points + [points[0]], fill=(0, 0, 0), width=max(1, int(round(size / 7))))
    return np.asarray(canvas, dtype=np.uint8)


def _to_tensor(image: Image.Image, input_size: int, mean, std, train: bool) -> torch.Tensor:
    """PIL 图片 -> 归一化后的 (3, H, W) float 张量。"""
    image = image.convert("RGB").resize((input_size, input_size), Image.BILINEAR)
    return _array_to_tensor(np.asarray(image, dtype=np.uint8), mean, std, train)


def _array_to_tensor(arr: np.ndarray, mean, std, train: bool) -> torch.Tensor:
    """uint8 (H, W, 3) RGB 数组 -> 归一化后的 (3, H, W) float 张量。"""
    if train:
        # 只做不移动像素位置的增强：轻微颜色/亮度抖动，提高对渲染差异的鲁棒性
        arr = arr.astype(np.float32)
        arr *= np.float32(np.random.uniform(0.9, 1.1))  # 亮度
        arr = (arr - arr.mean()) * np.float32(np.random.uniform(0.9, 1.1)) + arr.mean()  # 对比度
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    out = arr.astype(np.float32) / 255.0
    out = (out - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    out = np.transpose(out, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))


class DesktopScreenshotDataset(Dataset):
    """桌面截图数据集。

    每个 ``__getitem__`` 返回::

        {
          "image":  FloatTensor[3, S, S],   # ImageNet 归一化
          "action": LongTensor[],           # 动作类别索引
          "point":  FloatTensor[2],         # 归一化目标点 (x, y) ∈ [0,1]
          "sample_id": str,
          "image_size": FloatTensor[2],     # (W, H)，collate 后为 (B, 2)
        }

    Args:
        preload: 初始化时把所有截图解码并缩放为 ``input_size`` 的 uint8 数组常驻内存。
            499 张截图约占用 75 MB，但能避免每个 epoch 重复解码 4K PNG，
            在 CPU 训练时可以把单轮耗时降低一半以上。
        draw_cursor: 在缩放后的图像上于目标点绘制箭头光标。数据集本身没有光标像素，
            开启后坐标回归才成为良定义的任务（见 :func:`draw_cursor`）。
            注意光标位置由标签决定且每个样本固定，所以在 ``preload`` 阶段画一次即可，
            训练循环里没有任何额外开销。
        augment_shift: 平移增强的最大像素数（输入空间）。图像被平移多少，目标点标签
            就同步平移多少，因此标签语义不变、却能显著提升对界面布局偏移的鲁棒性。
            0 表示关闭。仅在 ``train=True`` 时生效。
    """

    def __init__(
        self,
        samples: Sequence[DesktopSample],
        label_map: Dict[str, int],
        input_size: int = 224,
        train: bool = False,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        preload: bool = True,
        verbose: bool = True,
        draw_cursor: bool = False,
        augment_shift: int = 0,
    ) -> None:
        self.samples = list(samples)
        self.label_map = dict(label_map)
        self.input_size = int(input_size)
        self.train = train
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.draw_cursor = bool(draw_cursor)
        self.augment_shift = max(int(augment_shift), 0)
        self._cache: Optional[List[np.ndarray]] = None

        if preload:
            cache: List[np.ndarray] = []
            for i, sample in enumerate(self.samples, 1):
                with Image.open(sample.image_path) as im:
                    im = im.convert("RGB").resize((self.input_size, self.input_size), Image.BILINEAR)
                    arr = np.asarray(im, dtype=np.uint8)
                cache.append(self._maybe_draw_cursor(arr, sample))
                if verbose and (i % 100 == 0 or i == len(self.samples)):
                    print(f"  预加载图像 {i}/{len(self.samples)}", flush=True)
            self._cache = cache

    def _maybe_draw_cursor(self, arr: np.ndarray, sample: DesktopSample) -> np.ndarray:
        if not self.draw_cursor:
            return arr
        nx, ny = sample.norm_point
        size = self.input_size
        return draw_cursor(arr, nx * size, ny * size, max(9, round(CURSOR_SIZE * size / 224)))

    def _augment(self, arr: np.ndarray, norm_point: Tuple[float, float]
                 ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """随机平移图像，并让目标点标签同步平移。

        平移是**保标签**的几何增强：图像内容移动 (dx, dy) 像素，目标点也移动同样多，
        所以"该点哪里"这个语义完全不变，只是让模型见过更多布局位置。
        用边缘像素填充（np.pad 的 edge 模式）而不是补零，避免引入现实中不存在的黑边。

        目标点贴边时**放弃这次增强**而不是把标签夹到 [0,1]：内容已经整体平移，
        若只夹标签就会出现"图像移了、标签没跟上"的错标，等于往训练集里掺噪声。
        """
        if not (self.train and self.augment_shift > 0):
            return arr, norm_point

        limit = self.augment_shift
        size = float(self.input_size)
        dx = int(np.random.randint(-limit, limit + 1))
        dy = int(np.random.randint(-limit, limit + 1))
        if dx == 0 and dy == 0:
            return arr, norm_point

        shifted_point = (norm_point[0] + dx / size, norm_point[1] + dy / size)
        # 目标点必须仍落在画面内，否则这次平移会破坏标签
        margin = 1.0 / size  # 允许一个像素的余量
        if not (margin <= shifted_point[0] <= 1.0 - margin
                and margin <= shifted_point[1] <= 1.0 - margin):
            return arr, norm_point

        # shifted[i, j] 必须等于 arr[i - dy, j - dx]，这样"内容里的特征从 (r, c)
        # 移到 (r + dy, c + dx)"才与标签的 (+dx, +dy) 一致。
        # 注意 x 方向也是减 dx：写成加号会让内容向左、标签向右，标签就不守恒了。
        padded = np.pad(arr, ((limit, limit), (limit, limit), (0, 0)), mode="edge")
        offset = limit
        shifted = padded[offset - dy: offset - dy + arr.shape[0],
                         offset - dx: offset - dx + arr.shape[1]]
        return shifted, shifted_point

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        if self._cache is not None:
            arr = self._cache[index]
        else:
            with Image.open(sample.image_path) as im:
                im = im.convert("RGB").resize((self.input_size, self.input_size), Image.BILINEAR)
            arr = self._maybe_draw_cursor(np.asarray(im, dtype=np.uint8), sample)

        norm_point = sample.norm_point
        arr, norm_point = self._augment(arr, norm_point)
        image = _array_to_tensor(arr, self.mean, self.std, self.train)

        action_idx = self.label_map.get(sample.action, -1)
        nx, ny = norm_point
        return {
            "image": image,
            "action": torch.tensor(action_idx, dtype=torch.long),
            "point": torch.tensor([nx, ny], dtype=torch.float32),
            "sample_id": sample.sample_id,
            # 用张量而不是元组，DataLoader 才能把它拼成 (B, 2)
            "image_size": torch.tensor(sample.image_size, dtype=torch.float32),
        }


def dataset_summary(samples: Sequence[DesktopSample]) -> Dict[str, object]:
    """统计一个 split 的规模、类别分布与分辨率分布。"""
    from collections import Counter

    actions = Counter(s.action for s in samples)
    sizes = Counter(f"{w}x{h}" for (w, h) in (s.image_size for s in samples))
    return {
        "num_samples": len(samples),
        "num_classes": len(actions),
        "action_counts": dict(sorted(actions.items(), key=lambda kv: (-kv[1], kv[0]))),
        "resolution_counts": dict(sizes.most_common()),
    }


__all__ = [
    "ACTION_ZH",
    "CURSOR_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DesktopSample",
    "DesktopScreenshotDataset",
    "action_zh",
    "build_label_map",
    "cursor_points",
    "dataset_summary",
    "draw_cursor",
    "load_label_map",
    "save_label_map",
    "scan_split",
]
