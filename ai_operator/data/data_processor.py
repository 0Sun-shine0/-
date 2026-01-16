"""
数据处理模块
处理屏幕截图、用户操作数据和训练数据的预处理
"""
import os
import json
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
import pickle
from datetime import datetime
import hashlib

from ..core.config import Config


class DataProcessor:
    """数据处理类"""
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessing_steps = []
        
    def preprocess_screen_image(self, image_path: str) -> np.ndarray:
        """
        预处理屏幕截图图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            预处理后的图像数组
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整大小到模型输入尺寸
        target_size = self.config.SCREEN_ANALYSIS_CONFIG['input_size']
        image = cv2.resize(image, target_size)
        
        # 归一化像素值到[0,1]范围
        image = image.astype(np.float32) / 255.0
        
        # 添加批次维度
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def preprocess_multiple_images(self, image_paths: List[str]) -> np.ndarray:
        """
        预处理多个图像
        
        Args:
            image_paths: 图像文件路径列表
            
        Returns:
            预处理后的图像数组(batch)
        """
        processed_images = []
        
        for path in image_paths:
            try:
                processed_img = self.preprocess_screen_image(path)
                processed_images.append(processed_img[0])  # 移除批次维度，稍后重新堆叠
            except Exception as e:
                print(f"处理图像 {path} 时出错: {e}")
                continue
        
        if not processed_images:
            raise ValueError("没有成功处理任何图像")
        
        # 堆叠成批次
        batch_images = np.stack(processed_images, axis=0)
        
        return batch_images
    
    def extract_ui_elements(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        从图像中提取UI元素
        
        Args:
            image: 输入图像数组
            
        Returns:
            UI元素列表
        """
        # 这里实现UI元素检测逻辑
        # 简化版本：检测边缘和基本形状
        if len(image.shape) == 4:  # 如果是批次数据，取第一张
            image = image[0]
        
        # 转换回OpenCV格式 (RGB to BGR)
        img_bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # 转换为灰度图
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ui_elements = []
        min_area = 100  # 最小面积阈值
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # 计算边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算中心点
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 估算元素类型（基于形状和大小）
                aspect_ratio = w / h
                element_type = self._classify_element_type(w, h, aspect_ratio)
                
                ui_element = {
                    'type': element_type,
                    'bbox': [x, y, w, h],
                    'center': [center_x, center_y],
                    'area': area,
                    'aspect_ratio': aspect_ratio
                }
                
                ui_elements.append(ui_element)
        
        return ui_elements
    
    def _classify_element_type(self, width: int, height: int, aspect_ratio: float) -> str:
        """
        根据尺寸和宽高比分类UI元素类型
        
        Args:
            width: 宽度
            height: 高度
            aspect_ratio: 宽高比
            
        Returns:
            元素类型字符串
        """
        area = width * height
        
        if aspect_ratio > 3:  # 很宽的元素，可能是文本框或菜单栏
            return 'text_field' if area < 5000 else 'menu_bar'
        elif aspect_ratio < 0.3:  # 很高的元素，可能是滚动条
            return 'scroll_bar'
        elif 0.8 <= aspect_ratio <= 1.2:  # 接近正方形，可能是按钮
            if area < 1000:
                return 'icon'
            else:
                return 'button'
        elif aspect_ratio > 1:  # 宽大于高，可能是按钮或输入框
            if area < 2000:
                return 'button'
            else:
                return 'text_field'
        else:  # 高大于宽
            return 'button' if area < 2000 else 'container'
    
    def _action_to_vector(self, action_data: Dict[str, Any]) -> np.ndarray:
        """
        将动作数据转换为向量（用于训练）
        
        Args:
            action_data: 动作数据字典
            
        Returns:
            动作向量
        """
        # 处理可能的嵌套结构，提取坐标和动作类型
        coords = [0, 0]
        action_type = 'click'
        
        # 尝试从不同可能的结构中提取坐标
        if isinstance(action_data, dict):
            if 'coordinates' in action_data:
                coords = action_data['coordinates']
            elif 'bbox' in action_data:
                bbox = action_data['bbox']
                coords = [bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2]  # 中心点
            elif 'center' in action_data:
                coords = action_data['center']
            
            if 'action_type' in action_data:
                action_type = action_data['action_type']
            elif 'type' in action_data:
                action_type = action_data['type']
            elif 'element_type' in action_data:
                action_type = action_data['element_type']
        
        # 确保坐标是数字列表
        if not isinstance(coords, list) or len(coords) < 2:
            coords = [0, 0]
        
        # 标准化坐标
        coords = [float(c) for c in coords[:2]]
        
        # 简单编码动作类型
        action_types = ['click', 'double_click', 'drag', 'keypress', 'type_text', 'move', 'button', 'text_field', 'icon', 'menu_bar', 'scroll_bar', 'container']
        type_encoding = [1 if action_type == t else 0 for t in action_types]
        
        # 归一化坐标
        max_coord = 1920  # 假设最大坐标为1920x1080
        norm_coords = [coord / max_coord for coord in coords]
        
        # 组合向量
        vector = norm_coords + type_encoding
        
        return np.array(vector, dtype=np.float32)
    
    def process_user_action_data(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理用户操作数据
        
        Args:
            action_data: 原始操作数据
            
        Returns:
            处理后的操作数据
        """
        processed_data = {
            'timestamp': action_data.get('timestamp', datetime.now().isoformat()),
            'action_type': action_data.get('action_type', ''),
            'coordinates': action_data.get('coordinates', [0, 0]),
            'screen_context': action_data.get('screen_context', {}),
            'application_state': action_data.get('application_state', {}),
            'intent': action_data.get('intent', ''),
            'success': action_data.get('success', True),
            'duration': action_data.get('duration', 0)
        }
        
        # 标准化坐标
        if 'coordinates' in processed_data:
            coords = processed_data['coordinates']
            if isinstance(coords, list) and len(coords) >= 2:
                # 确保坐标在合理范围内
                processed_data['coordinates'] = [
                    max(0, min(coords[0], self.config.ACTION_CONFIG['max_coordinate'])),
                    max(0, min(coords[1], self.config.ACTION_CONFIG['max_coordinate']))
                ]
        
        # 处理屏幕上下文
        if 'screen_context' in processed_data:
            screen_ctx = processed_data['screen_context']
            if 'screenshot_path' in screen_ctx:
                # 计算截图的哈希值用于唯一标识
                if os.path.exists(screen_ctx['screenshot_path']):
                    with open(screen_ctx['screenshot_path'], 'rb') as f:
                        content = f.read()
                        hash_value = hashlib.md5(content).hexdigest()
                        processed_data['screen_hash'] = hash_value
        
        return processed_data
    
    def create_training_dataset(self, 
                              screen_samples: List[str], 
                              action_samples: List[Dict[str, Any]],
                              output_path: str) -> bool:
        """
        创建训练数据集
        
        Args:
            screen_samples: 屏幕截图路径列表
            action_samples: 操作样本列表
            output_path: 输出路径
            
        Returns:
            是否成功创建数据集
        """
        if len(screen_samples) != len(action_samples):
            raise ValueError("屏幕样本和操作样本数量不匹配")
        
        dataset = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'sample_count': len(screen_samples),
                'screen_size': self.config.SCREEN_ANALYSIS_CONFIG['input_size']
            },
            'samples': []
        }
        
        for screen_path, action_data in zip(screen_samples, action_samples):
            try:
                # 预处理屏幕图像
                processed_image = self.preprocess_screen_image(screen_path)
                
                # 处理操作数据
                processed_action = self.process_user_action_data(action_data)
                
                # 创建样本
                sample = {
                    'screen_input': processed_image.tolist(),  # 转换为JSON可序列化格式
                    'action_output': processed_action,
                    'screen_path': screen_path
                }
                
                dataset['samples'].append(sample)
                
            except Exception as e:
                print(f"处理样本时出错 (screen: {screen_path}): {e}")
                continue
        
        # 保存数据集
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"成功创建包含 {len(dataset['samples'])} 个样本的训练数据集: {output_path}")
        return True
    
    def load_training_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        加载训练数据集
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            加载的数据集
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"成功加载数据集，包含 {len(dataset.get('samples', []))} 个样本")
        return dataset
    
    def batch_process_dataset(self, dataset_path: str, batch_size: int = 32) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        批量处理数据集
        
        Args:
            dataset_path: 数据集路径
            batch_size: 批次大小
            
        Returns:
            (批次图像, 批次操作数据) 元组
        """
        dataset = self.load_training_dataset(dataset_path)
        samples = dataset.get('samples', [])
        
        batch_images = []
        batch_actions = []
        
        for i, sample in enumerate(samples):
            # 转换图像数据
            img_data = np.array(sample['screen_input'])
            batch_images.append(img_data[0])  # 移除批次维度
            
            # 保存操作数据
            batch_actions.append(sample['action_output'])
            
            # 如果达到批次大小或处理完所有样本，返回批次
            if len(batch_images) >= batch_size or i == len(samples) - 1:
                images_array = np.array(batch_images)
                yield images_array, batch_actions
                
                # 重置批次
                batch_images = []
                batch_actions = []
    
    def augment_screen_data(self, image: np.ndarray, 
                           rotation_range: int = 10, 
                           brightness_range: float = 0.2) -> List[np.ndarray]:
        """
        增强屏幕数据（数据增强）
        
        Args:
            image: 输入图像
            rotation_range: 旋转范围（度）
            brightness_range: 亮度变化范围
            
        Returns:
            增强后的图像列表
        """
        if len(image.shape) == 4:  # 如果是批次数据，取第一张
            image = image[0]
        
        augmented_images = [image]  # 原始图像
        
        # 旋转增强
        for angle in [-rotation_range, rotation_range]:
            rotation_matrix = cv2.getRotationMatrix2D(
                (image.shape[1]//2, image.shape[0]//2), angle, 1
            )
            rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            augmented_images.append(rotated)
        
        # 亮度增强
        for factor in [1-brightness_range, 1+brightness_range]:
            brightened = np.clip(image * factor, 0, 1).astype(np.float32)
            augmented_images.append(brightened)
        
        # 高斯噪声
        noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
        noisy = np.clip(image + noise, 0, 1).astype(np.float32)
        augmented_images.append(noisy)
        
        return augmented_images
    
    def extract_features_from_image(self, image: np.ndarray) -> np.ndarray:
        """
        从图像中提取特征
        
        Args:
            image: 输入图像
            
        Returns:
            特征向量
        """
        # 简化的特征提取（实际应用中可能使用预训练模型）
        if len(image.shape) == 4:  # 如果是批次数据，取第一张
            image = image[0]
        
        # 转换为灰度图
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # 计算直方图作为特征
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # 归一化
        
        # 计算梯度直方图
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 将特征连接成一个向量
        features = np.concatenate([hist, [grad_magnitude.mean(), grad_magnitude.std()]])
        
        return features


# 使用示例
def example_usage():
    """使用示例"""
    config = Config()
    processor = DataProcessor(config)
    
    # 示例：处理单个屏幕截图
    # 注意：这里需要实际的图像文件路径
    # processed_img = processor.preprocess_screen_image("path/to/screenshot.png")
    
    # 示例：提取UI元素（需要实际图像数据）
    # ui_elements = processor.extract_ui_elements(processed_img)
    
    print("数据处理器初始化完成")


if __name__ == "__main__":
    example_usage()