"""
卷积神经网络模型 - 用于屏幕图像理解和动作预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Dict, Any, Optional, Sequence
import logging

import numpy as np

logger = logging.getLogger(__name__)


def decode_points(model: "nn.Module", raw: torch.Tensor) -> torch.Tensor:
    """把 ``mouse_positions`` 的原始输出解码成 [0, 1] 归一化坐标。

    - ``spatial_coord=True``：soft-argmax 出的坐标本来就在 [0, 1]，直接用。
    - ``spatial_coord=False``：线性头的裸输出，需要 sigmoid 压到 [0, 1]。

    统一走这个函数，是为了避免训练 / 评估 / 推理三处各写一遍 sigmoid 而漏改某一处。
    """
    if getattr(model, "spatial_coord", False):
        return raw.clamp(0.0, 1.0)
    return torch.sigmoid(raw)


def spatial_coord_from_state(state_dict) -> bool:
    """从 checkpoint 的 key 判断它是不是用空间坐标头训练的。"""
    return any(str(k).startswith("coord_head.") for k in state_dict.keys())

class EnhancedCNN(nn.Module):
    """增强的CNN模型，用于屏幕理解和多任务预测

    Args:
        spatial_coord: 使用带空间信息的坐标头（soft-argmax 热力图），而不是在
            全局平均池化后的特征上做线性回归。**定位任务必须开启**：ResNet 自带的
            ``avgpool`` 会把空间维度压成 1x1，线性层无法从全局描述子里恢复出坐标。
            默认 False 只是为了兼容旧 checkpoint。
    """

    def __init__(self, num_actions: int, num_apps: int = 10, screen_size: Tuple[int, int] = (224, 224),
                 pretrained: bool = True, spatial_coord: bool = False):
        super(EnhancedCNN, self).__init__()
        
        # 确保num_actions是整数
        self.num_actions = int(num_actions)
        self.spatial_coord = bool(spatial_coord)
        
        # 使用预训练的ResNet作为backbone
        # pretrained=False 时使用随机初始化，便于离线/快速单元测试
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.resnet = models.resnet50(weights=weights)
        
        # 修改第一层适应屏幕截图（3通道RGB）
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 获取特征维度
        num_features = self.resnet.fc.in_features
        
        # 多任务预测头
        # 1. 动作预测
        self.action_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_actions)
        )
        
        # 2. 应用程序识别
        self.app_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_apps)
        )
        
        # 3. UI元素检测（边界框回归）
        self.bbox_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)  # [x, y, width, height]
        )
        
        # 4. 屏幕语义分割
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(num_features, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(32, 10, kernel_size=1)  # 10个UI元素类别
        )
        
        # 5. 鼠标位置预测
        # spatial_coord=True 时改用下面的空间热力图头；这里的线性头保留给旧 checkpoint
        self.mouse_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # (x, y)坐标
        )
        
        # 6. 空间坐标头（soft-argmax）
        # 在 layer4 的特征图上生成单通道热力图，再用 softmax 求期望坐标。
        # 这样坐标精度受特征图分辨率而非全局池化限制，是定位任务唯一可行的做法。
        if self.spatial_coord:
            self.coord_head = nn.Sequential(
                nn.Conv2d(num_features, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1, kernel_size=1),
            )
        
        # 移除ResNet的原始全连接层
        self.resnet.fc = nn.Identity()
        
        # 屏幕尺寸
        self.screen_size = screen_size
    
    def _run_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """跑一遍 ResNet 主干，返回 layer4 的空间特征图 (N, 2048, H/32, W/32)。

        注意不能直接用 ``self.resnet(x)``：ResNet 的 forward 末尾自带
        ``avgpool``，会把空间维度压成 1x1，定位与分割都无法工作。
        """
        f = self.resnet.conv1(x)
        f = self.resnet.bn1(f)
        f = self.resnet.relu(f)
        f = self.resnet.maxpool(f)
        f = self.resnet.layer1(f)
        f = self.resnet.layer2(f)
        f = self.resnet.layer3(f)
        f = self.resnet.layer4(f)
        return f
    
    def _soft_argmax(self, feature_map: torch.Tensor) -> torch.Tensor:
        """热力图 -> 归一化坐标 (N, 2)。"""
        heat = self.coord_head(feature_map)
        # 上采样到 1/4 输入分辨率，把坐标量化误差从 1/7 降到约 4 像素
        heat = F.interpolate(heat, size=(heat.size(2) * 8, heat.size(3) * 8),
                             mode="bilinear", align_corners=False)
        n, _, hh, ww = heat.shape
        prob = F.softmax(heat.view(n, -1), dim=1).view(n, 1, hh, ww)
        # 用 (i + 0.5) / size 与标签的归一化方式保持一致（x_px / W_orig）
        xs = ((torch.arange(ww, device=heat.device, dtype=heat.dtype) + 0.5) / ww).view(1, 1, 1, ww)
        ys = ((torch.arange(hh, device=heat.device, dtype=heat.dtype) + 0.5) / hh).view(1, 1, hh, 1)
        cx = (prob * xs).sum(dim=(2, 3))
        cy = (prob * ys).sum(dim=(2, 3))
        return torch.cat([cx, cy], dim=1)
        
    def forward(self, x: torch.Tensor, return_segmentation: bool = True) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            x: 输入图像张量 (N, 3, H, W)
            return_segmentation: 是否计算分割分支。该分支包含一整套 ConvTranspose2d，
                约占整体计算量的一半，且当前数据集没有分割标签，
                因此训练时可以传 False 显著提速；推理/导出保持默认 True。
        """
        # 主干只跑一次，空间特征图同时供坐标头 / 分割头使用
        feature_map = self._run_backbone(x)
        features = torch.flatten(F.adaptive_avg_pool2d(feature_map, (1, 1)), 1)
        
        # 多任务预测
        action_logits = self.action_head(features)
        app_logits = self.app_head(features)
        bbox_pred = self.bbox_head(features)
        mouse_pred = self._soft_argmax(feature_map) if self.spatial_coord else self.mouse_head(features)
        
        if return_segmentation:
            segmentation = self.segmentation_head(feature_map)
            segmentation = F.interpolate(segmentation, size=(self.screen_size[0], self.screen_size[1]), mode='bilinear', align_corners=False)
        else:
            segmentation = None
        
        return {
            'actions': action_logits,
            'applications': app_logits,
            'bounding_boxes': bbox_pred,
            'mouse_positions': mouse_pred,
            'segmentation': segmentation
        }
    
    def _to_input_tensor(self, image) -> torch.Tensor:
        """把 numpy 图像或任意形状张量整理成模型输入 ``(1, 3, S, S)``。

        GUI 之前直接把 OpenCV 的 ``(H, W, 3) uint8`` 数组传进 predict()，
        而 nn.Module 只接受张量，于是必然抛
        ``TypeError: conv2d() received numpy.ndarray``。这里统一做转换。
        """
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            # OpenCV 给的是 BGR；调用方若已是 RGB 可先自行转换
            tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0
        elif isinstance(image, torch.Tensor):
            tensor = image.detach().float()
            if tensor.max() > 1.5:  # 看起来是 0-255 的原始像素
                tensor = tensor / 255.0
        else:
            raise TypeError(f"不支持的输入类型: {type(image)!r}（应为 numpy 数组或 torch.Tensor）")

        while tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[1] != 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)

        target = (self.screen_size[0], self.screen_size[1])
        if (tensor.shape[2], tensor.shape[3]) != target:
            tensor = F.interpolate(tensor, size=target, mode="bilinear", align_corners=False)
        return tensor

    @torch.no_grad()
    def predict(self, image, topk: int = 3, action_names: Optional[Sequence[str]] = None,
                with_segmentation: bool = False) -> Dict[str, Any]:
        """推理预测。

        Args:
            image: numpy 图像（HWC, uint8, BGR）或 torch 张量。
            topk: 返回前 k 个动作。
            action_names: 类别索引 -> 动作名。传入后额外返回 ``action_name``，
                否则调用方只拿到无意义的整数索引。
            with_segmentation: 是否计算分割图。默认关闭：该分支含 4 层
                ConvTranspose2d，代价约等于再跑一遍主干，而训练数据并没有分割标签。

        Returns:
            ``mouse_position`` 是 **[0,1] 归一化坐标**（不再是乘过 224 的像素值）；
            要落到真实屏幕像素请乘上屏幕宽高，用 :func:`decode_points` 的语义保持一致。
        """
        self.eval()
        tensor = self._to_input_tensor(image)
        outputs = self.forward(tensor, return_segmentation=with_segmentation)

        k = max(1, min(int(topk), outputs["actions"].shape[1]))
        action_probs = F.softmax(outputs["actions"], dim=1)
        top_actions = torch.topk(action_probs, k)
        app_probs = F.softmax(outputs["applications"], dim=1)
        top_apps = torch.topk(app_probs, min(2, app_probs.shape[1]))
        bboxes = torch.sigmoid(outputs["bounding_boxes"])
        # 必须走 decode_points：spatial_coord 模型的坐标已在 [0,1]，
        # 再套一次 sigmoid 会被压向 0.5
        mouse_pos = decode_points(self, outputs["mouse_positions"])

        def name_of(idx: int) -> Optional[str]:
            if action_names is None or not (0 <= idx < len(action_names)):
                return None
            return str(action_names[idx])

        result: Dict[str, Any] = {
            "top_actions": [
                {"action": idx.item(), "action_name": name_of(idx.item()), "confidence": prob.item()}
                for idx, prob in zip(top_actions.indices[0], top_actions.values[0])
            ],
            "top_applications": [
                {"app": idx.item(), "confidence": prob.item()}
                for idx, prob in zip(top_apps.indices[0], top_apps.values[0])
            ],
            "bounding_boxes": bboxes[0].tolist(),
            # 归一化坐标；乘屏幕宽高即得真实像素
            "mouse_position": mouse_pos[0].tolist(),
            "confidence_scores": {
                "action_max": action_probs.max().item(),
                "app_max": app_probs.max().item(),
            },
        }
        if with_segmentation and outputs["segmentation"] is not None:
            result["segmentation_map"] = torch.argmax(outputs["segmentation"], dim=1)[0].cpu().numpy()
        return result

class MultiModalTransformer(nn.Module):
    """多模态Transformer模型，结合图像和文本特征"""
    
    def __init__(self, num_actions: int, text_vocab_size: int = 50000, d_model: int = 512):
        super(MultiModalTransformer, self).__init__()
        
        # 图像编码器
        self.image_encoder = models.resnet34(pretrained=True)
        num_image_features = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Linear(num_image_features, d_model)
        
        # 文本编码器
        self.text_embedding = nn.Embedding(text_vocab_size, d_model)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8),
            num_layers=3
        )
        
        # 多模态融合
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8),
            num_layers=2
        )
        
        # 动作预测头
        self.action_predictor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_actions)
        )
        
        # 上下文预测头
        self.context_predictor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 20)  # 20种上下文类型
        )
        
        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        
    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        batch_size = images.size(0)
        
        # 图像特征提取
        image_features = self.image_encoder(images)  # [batch, d_model]
        image_features = image_features.unsqueeze(1)  # [batch, 1, d_model]
        
        # 文本特征提取
        text_embeddings = self.text_embedding(text_tokens)  # [batch, seq_len, d_model]
        text_features = self.text_encoder(text_embeddings.transpose(0, 1)).transpose(0, 1)
        text_features = text_features.mean(dim=1).unsqueeze(1)  # [batch, 1, d_model]
        
        # 多模态融合
        combined_features = torch.cat([image_features, text_features], dim=1)  # [batch, 2, d_model]
        
        # 添加位置编码
        seq_len = combined_features.size(1)
        combined_features = combined_features + self.positional_encoding[:, :seq_len, :]
        
        # Transformer融合
        fused_features = self.fusion_transformer(combined_features.transpose(0, 1)).transpose(0, 1)
        fused_features = fused_features.mean(dim=1)  # [batch, d_model]
        
        # 预测
        action_logits = self.action_predictor(fused_features)
        context_logits = self.context_predictor(fused_features)
        
        return {
            'actions': action_logits,
            'context': context_logits,
            'features': fused_features
        }