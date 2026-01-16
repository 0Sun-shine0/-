"""
AI桌面操作员 - 模型集成脚本
将训练好的模型集成到主程序中，实现实时屏幕分析和操作预测
"""
import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
import time
import pyautogui
from typing import Dict, Tuple, Any

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from ai_operator.models.cnn_model import EnhancedCNN
from ai_operator.core.config import Config
from ai_operator.data.data_processor import DataProcessor
from ai_operator.automation.mouse_controller import MouseController
from ai_operator.automation.keyboard_controller import KeyboardController

class AIDeskOperator:
    """AI桌面操作员主类"""
    
    def __init__(self):
        self.config = Config()
        self.data_processor = DataProcessor(self.config)
        self.mouse_controller = MouseController()
        self.keyboard_controller = KeyboardController()
        
        # 初始化模型
        self.model = EnhancedCNN(14)  # 14维输出
        self.load_model()
        
        # 操作类型映射
        self.action_types = [
            'click', 'double_click', 'drag', 'keypress', 
            'type_text', 'move', 'button', 'text_field', 
            'icon', 'menu_bar', 'scroll_bar', 'container'
        ]
        
        # 是否启用安全确认
        self.safety_enabled = True
        
    def load_model(self):
        """加载训练好的模型"""
        model_path = "ai_operator_dataset/models/ai_operator_model.pth"
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            print(f"模型已加载: {model_path}")
        else:
            print("警告：未找到训练好的模型，使用随机初始化模型")
            print("请先运行训练流程生成模型")
    
    def capture_screen(self) -> np.ndarray:
        """捕获当前屏幕"""
        screenshot = pyautogui.screenshot()
        screen_array = np.array(screenshot)
        screen_array = cv2.resize(screen_array, (224, 224))  # 调整到模型输入尺寸
        screen_array = screen_array.astype(np.float32) / 255.0  # 归一化
        screen_array = np.transpose(screen_array, (2, 0, 1))  # 调整维度顺序 (C, H, W)
        return screen_array
    
    def preprocess_screen(self, screen_array: np.ndarray) -> torch.Tensor:
        """预处理屏幕数据"""
        # 添加批次维度
        screen_tensor = torch.from_numpy(screen_array).unsqueeze(0)
        return screen_tensor
    
    def predict_action(self, screen_tensor: torch.Tensor) -> Dict[str, Any]:
        """预测操作"""
        with torch.no_grad():
            outputs = self.model(screen_tensor)
            action_output = outputs['actions'].squeeze(0).numpy()
        
        # 解析动作向量
        coords = action_output[:2]  # 前2个是坐标
        action_probs = action_output[2:]  # 后12个是动作类型概率
        
        # 归一化坐标到屏幕尺寸
        screen_width, screen_height = pyautogui.size()
        x = int(coords[0] * screen_width)
        y = int(coords[1] * screen_height)
        
        # 获取最可能的动作类型
        action_type_idx = np.argmax(action_probs)
        action_type = self.action_types[action_type_idx]
        confidence = action_probs[action_type_idx]
        
        return {
            'coordinates': (x, y),
            'action_type': action_type,
            'confidence': confidence,
            'all_probabilities': action_probs
        }
    
    def execute_action(self, prediction: Dict[str, Any]) -> bool:
        """执行预测的操作"""
        if self.safety_enabled:
            print(f"预测操作: {prediction['action_type']} 在 ({prediction['coordinates'][0]}, {prediction['coordinates'][1]})")
            print(f"置信度: {prediction['confidence']:.3f}")
            
            # 安全确认
            if prediction['confidence'] < 0.5:
                print("置信度过低，跳过操作")
                return False
            
            confirm = input("是否执行此操作? (y/n): ").strip().lower()
            if confirm != 'y':
                print("操作已取消")
                return False
        
        x, y = prediction['coordinates']
        action_type = prediction['action_type']
        
        try:
            if action_type == 'click':
                pyautogui.click(x, y)
            elif action_type == 'double_click':
                pyautogui.doubleClick(x, y)
            elif action_type == 'drag':
                # 简化拖拽操作：从当前位置拖到目标位置
                current_x, current_y = pyautogui.position()
                pyautogui.dragTo(x, y, duration=0.5)
            elif action_type in ['keypress', 'type_text']:
                # 如果是文本输入，需要特殊处理
                if action_type == 'type_text':
                    # 这里可以添加文本输入逻辑
                    pass
                else:
                    # 简单按键操作
                    pass
            elif action_type == 'move':
                pyautogui.moveTo(x, y)
            else:
                # 对于UI元素类型，可能需要不同的处理方式
                print(f"UI元素类型操作: {action_type}")
                pyautogui.click(x, y)
            
            print(f"操作执行成功: {action_type} at ({x}, {y})")
            return True
            
        except Exception as e:
            print(f"执行操作时出错: {e}")
            return False
    
    def run_single_cycle(self) -> Dict[str, Any]:
        """运行单个预测-执行周期"""
        # 捕获屏幕
        screen_array = self.capture_screen()
        screen_tensor = self.preprocess_screen(screen_array)
        
        # 预测操作
        prediction = self.predict_action(screen_tensor)
        
        # 执行操作
        success = self.execute_action(prediction)
        
        return {
            'prediction': prediction,
            'execution_success': success,
            'timestamp': time.time()
        }
    
    def run_continuous(self, interval: float = 1.0):
        """连续运行AI桌面操作员"""
        print("AI桌面操作员启动...")
        print("按 Ctrl+C 停止")
        
        try:
            while True:
                result = self.run_single_cycle()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nAI桌面操作员已停止")

def main():
    """主函数"""
    print("=" * 60)
    print("AI桌面操作员 - 模型集成演示")
    print("=" * 60)
    
    # 创建AI桌面操作员实例
    operator = AIDeskOperator()
    
    print("\n选项:")
    print("1. 单次预测和执行")
    print("2. 连续运行模式")
    print("3. 退出")
    
    while True:
        choice = input("\n请选择 (1-3): ").strip()
        
        if choice == '1':
            print("执行单次预测...")
            result = operator.run_single_cycle()
            print(f"执行结果: {'成功' if result['execution_success'] else '失败'}")
        
        elif choice == '2':
            interval = input("请输入执行间隔（秒，默认1.0）: ").strip()
            try:
                interval = float(interval) if interval else 1.0
            except ValueError:
                interval = 1.0
            print(f"开始连续运行，间隔 {interval} 秒...")
            operator.run_continuous(interval)
            break
        
        elif choice == '3':
            print("退出")
            break
        
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()