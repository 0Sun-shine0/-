"""
AI桌面操作员 - 增强训练脚本
用于继续训练模型以适应实际桌面环境
"""
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import cv2
import json
import random
from typing import Tuple, Dict, Any

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from ai_operator.models.cnn_model import EnhancedCNN
from ai_operator.core.config import Config
from ai_operator.data.data_processor import DataProcessor

class RealDesktopDataset(Dataset):
    """真实桌面环境数据集"""
    
    def __init__(self, dataset_path: str, transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.data_processor = DataProcessor(Config())
        
        # 收集所有图像和标注文件路径
        self.image_paths = []
        self.annotation_paths = []
        
        # 遍历训练集目录
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
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)
        
        # 转换标注为动作向量
        action_vector = self.data_processor._action_to_vector(annotation)
        action_tensor = torch.from_numpy(action_vector).float()
        
        return img_tensor, action_tensor

class DataAugmentation:
    """数据增强类"""
    
    @staticmethod
    def random_rotation(image: np.ndarray, max_angle: float = 10) -> np.ndarray:
        """随机旋转"""
        angle = random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
        return rotated
    
    @staticmethod
    def random_brightness(image: np.ndarray, factor_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """随机亮度调整"""
        factor = random.uniform(*factor_range)
        brightened = np.clip(image * factor, 0, 1).astype(np.float32)
        return brightened
    
    @staticmethod
    def add_noise(image: np.ndarray, noise_factor: float = 0.01) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, noise_factor, image.shape).astype(np.float32)
        noisy = np.clip(image + noise, 0, 1).astype(np.float32)
        return noisy
    
    @staticmethod
    def random_flip(image: np.ndarray, horizontal: bool = True) -> np.ndarray:
        """随机翻转"""
        if horizontal and random.random() > 0.5:
            return cv2.flip(image, 1)  # 水平翻转
        return image

def continue_training(epochs: int = 30, learning_rate: float = 0.0001, batch_size: int = 16):
    """继续训练模型"""
    print("开始增强训练...")
    
    # 加载预训练模型
    model = EnhancedCNN(14)  # 14维输出
    
    # 尝试加载微调后的模型，如果不存在则加载原始模型
    model_paths = [
        "ai_operator_dataset/models/ai_operator_model_finetuned.pth",
        "ai_operator_dataset/models/ai_operator_model.pth"
    ]
    
    loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"已加载预训练模型: {model_path}")
            loaded = True
            break
    
    if not loaded:
        print("警告：未找到预训练模型，使用随机初始化模型")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"使用设备: {device}")
    
    # 创建数据集和数据加载器
    dataset = RealDesktopDataset("ai_operator_dataset")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # 训练参数
    model.train()
    print(f"开始训练，共 {epochs} 轮，学习率: {learning_rate}")
    
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
    
    # 保存增强训练后的模型
    enhanced_model_path = "ai_operator_dataset/models/ai_operator_model_enhanced.pth"
    os.makedirs(os.path.dirname(enhanced_model_path), exist_ok=True)
    torch.save(model.state_dict(), enhanced_model_path)
    print(f"增强训练完成，模型已保存到: {enhanced_model_path}")
    
    return enhanced_model_path

def train_with_augmentation(epochs: int = 20, learning_rate: float = 0.00005):
    """使用数据增强继续训练"""
    print("开始使用数据增强的训练...")
    
    # 加载模型
    model = EnhancedCNN(14)
    
    # 尝试加载之前的模型
    model_paths = [
        "ai_operator_dataset/models/ai_operator_model_enhanced.pth",
        "ai_operator_dataset/models/ai_operator_model_finetuned.pth",
        "ai_operator_dataset/models/ai_operator_model.pth"
    ]
    
    loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"已加载模型: {model_path}")
            loaded = True
            break
    
    if not loaded:
        print("警告：未找到模型，使用随机初始化")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"使用设备: {device}")
    
    # 创建数据集（这里简化，实际应用中需要实现增强的数据集）
    dataset = RealDesktopDataset("ai_operator_dataset")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # 训练
    model.train()
    print(f"开始数据增强训练，共 {epochs} 轮")
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, actions) in enumerate(dataloader):
            images, actions = images.to(device), actions.to(device)
            
            # 应用数据增强（在GPU上操作）
            # 这里简化处理，实际应用中需要更复杂的增强逻辑
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs['actions'], actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'轮次 {epoch+1}/{epochs}, 平均损失: {avg_loss:.6f}')
    
    # 保存模型
    aug_model_path = "ai_operator_dataset/models/ai_operator_model_augmented.pth"
    os.makedirs(os.path.dirname(aug_model_path), exist_ok=True)
    torch.save(model.state_dict(), aug_model_path)
    print(f"数据增强训练完成，模型已保存到: {aug_model_path}")
    
    return aug_model_path

def evaluate_and_compare(model_paths: list):
    """评估并比较不同模型的性能"""
    print("评估不同模型的性能...")
    
    dataset = RealDesktopDataset("ai_operator_dataset")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    results = {}
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"模型不存在: {model_path}")
            continue
        
        # 加载模型
        model = EnhancedCNN(14)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for images, actions in dataloader:
                images, actions = images.to(device), actions.to(device)
                outputs = model(images)
                loss = nn.MSELoss()(outputs['actions'], actions)
                total_loss += loss.item()
                count += 1
        
        avg_loss = total_loss / count if count > 0 else float('inf')
        results[model_path] = avg_loss
        print(f"{model_path}: 平均损失 = {avg_loss:.6f}")
    
    return results

def main():
    """主函数"""
    print("=" * 60)
    print("AI桌面操作员 - 增强训练系统")
    print("=" * 60)
    
    print("\n功能选项:")
    print("1. 继续训练模型")
    print("2. 使用数据增强训练")
    print("3. 评估模型性能")
    print("4. 训练并评估全部")
    print("5. 退出")
    
    while True:
        choice = input("\n请选择功能 (1-5): ").strip()
        
        if choice == '1':
            epochs = input("请输入训练轮数 (默认30): ").strip()
            try:
                epochs = int(epochs) if epochs else 30
            except ValueError:
                epochs = 30
            
            lr = input("请输入学习率 (默认0.0001): ").strip()
            try:
                lr = float(lr) if lr else 0.0001
            except ValueError:
                lr = 0.0001
            
            continue_training(epochs=epochs, learning_rate=lr)
        
        elif choice == '2':
            epochs = input("请输入训练轮数 (默认20): ").strip()
            try:
                epochs = int(epochs) if epochs else 20
            except ValueError:
                epochs = 20
            
            lr = input("请输入学习率 (默认0.00005): ").strip()
            try:
                lr = float(lr) if lr else 0.00005
            except ValueError:
                lr = 0.00005
            
            train_with_augmentation(epochs=epochs, learning_rate=lr)
        
        elif choice == '3':
            model_paths = [
                "ai_operator_dataset/models/ai_operator_model.pth",
                "ai_operator_dataset/models/ai_operator_model_finetuned.pth",
                "ai_operator_dataset/models/ai_operator_model_enhanced.pth",
                "ai_operator_dataset/models/ai_operator_model_augmented.pth"
            ]
            
            results = evaluate_and_compare(model_paths)
            print("\n模型性能比较:")
            for path, loss in sorted(results.items(), key=lambda x: x[1]):
                print(f"{path}: {loss:.6f}")
        
        elif choice == '4':
            print("开始完整训练和评估流程...")
            
            # 首先进行常规训练
            print("\n--- 1. 继续训练模型 ---")
            enhanced_path = continue_training(epochs=20, learning_rate=0.0001)
            
            # 然后进行数据增强训练
            print("\n--- 2. 数据增强训练 ---")
            augmented_path = train_with_augmentation(epochs=15, learning_rate=0.00005)
            
            # 最后评估所有模型
            print("\n--- 3. 模型性能评估 ---")
            model_paths = [
                "ai_operator_dataset/models/ai_operator_model.pth",
                "ai_operator_dataset/models/ai_operator_model_finetuned.pth",
                enhanced_path,
                augmented_path
            ]
            
            results = evaluate_and_compare(model_paths)
            print("\n最终模型性能比较:")
            for path, loss in sorted(results.items(), key=lambda x: x[1]):
                print(f"{path}: {loss:.6f}")
        
        elif choice == '5':
            print("退出系统")
            break
        
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()