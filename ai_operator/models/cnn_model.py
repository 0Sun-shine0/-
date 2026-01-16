"""
卷积神经网络模型 - 用于屏幕图像理解和动作预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EnhancedCNN(nn.Module):
    """增强的CNN模型，用于屏幕理解和多任务预测"""
    
    def __init__(self, num_actions: int, num_apps: int = 10, screen_size: Tuple[int, int] = (224, 224)):
        super(EnhancedCNN, self).__init__()
        
        # 确保num_actions是整数
        self.num_actions = int(num_actions)
        
        # 使用预训练的ResNet作为backbone
        self.resnet = models.resnet50(pretrained=True)
        
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
        self.mouse_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # (x, y)坐标
        )
        
        # 移除ResNet的原始全连接层
        self.resnet.fc = nn.Identity()
        
        # 屏幕尺寸
        self.screen_size = screen_size
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 提取特征
        features = self.resnet(x)
        
        # 全局平均池化
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
        
        # 多任务预测
        action_logits = self.action_head(features)
        app_logits = self.app_head(features)
        bbox_pred = self.bbox_head(features)
        mouse_pred = self.mouse_head(features)
        
        # 对于分割，需要不同的特征处理
        # 使用ResNet的中间层特征进行分割
        x_for_seg = self.resnet.conv1(x)
        x_for_seg = self.resnet.bn1(x_for_seg)
        x_for_seg = self.resnet.relu(x_for_seg)
        x_for_seg = self.resnet.maxpool(x_for_seg)
        
        x_for_seg = self.resnet.layer1(x_for_seg)
        x_for_seg = self.resnet.layer2(x_for_seg)
        x_for_seg = self.resnet.layer3(x_for_seg)
        x_for_seg = self.resnet.layer4(x_for_seg)
        
        segmentation = self.segmentation_head(x_for_seg)
        segmentation = F.interpolate(segmentation, size=(self.screen_size[0], self.screen_size[1]), mode='bilinear', align_corners=False)
        
        return {
            'actions': action_logits,
            'applications': app_logits,
            'bounding_boxes': bbox_pred,
            'mouse_positions': mouse_pred,
            'segmentation': segmentation
        }
    
    def predict(self, screen_tensor: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        """推理预测"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(screen_tensor)
            
            # 处理动作预测
            action_probs = F.softmax(outputs['actions'], dim=1)
            top_actions = torch.topk(action_probs, 3)  # 获取top-3动作
            
            # 处理应用程序识别
            app_probs = F.softmax(outputs['applications'], dim=1)
            top_apps = torch.topk(app_probs, 2)
            
            # 处理边界框（归一化到[0, 1]）
            bboxes = torch.sigmoid(outputs['bounding_boxes'])
            
            # 处理鼠标位置（归一化到屏幕尺寸）
            mouse_pos = torch.sigmoid(outputs['mouse_positions'])
            mouse_pos = mouse_pos * torch.tensor([self.screen_size[1], self.screen_size[0]])
            
            # 处理分割结果
            segmentation = torch.argmax(outputs['segmentation'], dim=1)
            
            return {
                'top_actions': [
                    {
                        'action': idx.item(),
                        'confidence': prob.item()
                    }
                    for idx, prob in zip(top_actions.indices[0], top_actions.values[0])
                ],
                'top_applications': [
                    {
                        'app': idx.item(),
                        'confidence': prob.item()
                    }
                    for idx, prob in zip(top_apps.indices[0], top_apps.values[0])
                ],
                'bounding_boxes': bboxes[0].tolist(),
                'mouse_position': mouse_pos[0].tolist(),
                'segmentation_map': segmentation[0].cpu().numpy(),
                'confidence_scores': {
                    'action_max': action_probs.max().item(),
                    'app_max': app_probs.max().item()
                }
            }

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