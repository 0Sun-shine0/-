"""
AI桌面操作员 - 模型微调脚本
"""
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
from PIL import Image

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from ai_operator.models.cnn_model import EnhancedCNN
from ai_operator.core.config import Config
from ai_operator.data.data_processor import DataProcessor

class AIDeskOperatorDataset(Dataset):
    """AI桌面操作员数据集类"""
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.image_paths = []
        self.annotation_paths = []
        self.data_processor = DataProcessor(Config())
        
        # 收集所有图像和标注文件路径
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_path / split
            images_dir = split_path / 'images'
            annotations_dir = split_path / 'annotations'
            
            if images_dir.exists() and annotations_dir.exists():
                # 遍历图像子目录
                for sub_dir in images_dir.iterdir():
                    if sub_dir.is_dir():
                        for img_file in sub_dir.glob("*.png"):
                            annotation_file = annotations_dir / f"{sub_dir.name}.json"
                            if annotation_file.exists():
                                self.image_paths.append(img_file)
                                self.annotation_paths.append(annotation_file)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))  # 统一尺寸
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # 转换为PyTorch格式 (C, H, W)
        
        # 加载标注
        annotation_path = self.annotation_paths[idx]
        import json
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)
        
        # 转换标注为动作向量
        action_vector = self.data_processor._action_to_vector(annotation)
        action_tensor = torch.from_numpy(action_vector).float()
        
        return img_tensor, action_tensor

def fine_tune_model():
    """模型微调函数"""
    print("开始模型微调...")
    
    # 加载预训练模型
    model = EnhancedCNN(14)  # 14维输出
    model_path = "ai_operator_dataset/models/ai_operator_model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"已加载预训练模型: {model_path}")
    else:
        print("未找到预训练模型，使用随机初始化")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"使用设备: {device}")
    
    # 创建数据集和数据加载器
    dataset = AIDeskOperatorDataset("ai_operator_dataset")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 降低学习率进行微调
    criterion = nn.MSELoss()
    
    # 训练参数
    epochs = 20  # 微调轮数
    model.train()
    
    print(f"开始微调，共 {epochs} 轮")
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, actions) in enumerate(dataloader):
            images, actions = images.to(device), actions.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs['actions'], actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'批次 {batch_idx}/{len(dataloader)}, 损失: {loss.item():.6f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'轮次 {epoch+1}/{epochs}, 平均损失: {avg_loss:.6f}')
    
    # 保存微调后的模型
    fine_tuned_model_path = "ai_operator_dataset/models/ai_operator_model_finetuned.pth"
    os.makedirs(os.path.dirname(fine_tuned_model_path), exist_ok=True)
    torch.save(model.state_dict(), fine_tuned_model_path)
    print(f"微调完成，模型已保存到: {fine_tuned_model_path}")

if __name__ == "__main__":
    fine_tune_model()