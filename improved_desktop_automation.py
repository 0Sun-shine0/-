"""
AI桌面操作员 - 改进版桌面自动化任务控制器
修复坐标范围问题，增加更多实用功能
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
import json
from typing import Dict, Tuple, Any, List, Optional
import threading
import queue

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from ai_operator.models.cnn_model import EnhancedCNN
from ai_operator.core.config import Config
from ai_operator.data.data_processor import DataProcessor
from ai_operator.automation.mouse_controller import AdvancedMouseController
from ai_operator.automation.keyboard_controller import KeyboardController

class ImprovedDesktopAutomation:
    """改进版桌面自动化控制器"""
    
    def __init__(self):
        self.config = Config()
        self.data_processor = DataProcessor(self.config)
        self.mouse_controller = AdvancedMouseController()
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
        
        # 安全机制
        self.safety_enabled = True
        self.confidence_threshold = 0.5
        self.emergency_stop = False
        
        # 任务队列
        self.task_queue = queue.Queue()
        self.running = False
        
        # 坐标范围修复参数
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"检测到屏幕尺寸: {self.screen_width}x{self.screen_height}")
    
    def load_model(self):
        """加载训练好的模型"""
        # 尝试加载微调后的模型，如果不存在则加载原始模型
        model_paths = [
            "ai_operator_dataset/models/ai_operator_model_finetuned.pth",
            "ai_operator_dataset/models/ai_operator_model.pth"
        ]
        
        loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
                print(f"模型已加载: {model_path}")
                loaded = True
                break
        
        if not loaded:
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
        
        # 归一化坐标到屏幕尺寸，并修复范围问题
        # 确保坐标在[0, 1]范围内
        x_norm = np.clip(coords[0], 0.0, 1.0)
        y_norm = np.clip(coords[1], 0.0, 1.0)
        
        x = int(x_norm * self.screen_width)
        y = int(y_norm * self.screen_height)
        
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
        if self.emergency_stop:
            print("紧急停止激活，跳过操作")
            return False
        
        x, y = prediction['coordinates']
        action_type = prediction['action_type']
        confidence = prediction['confidence']
        
        print(f"预测操作: {action_type} 在 ({x}, {y})，置信度: {confidence:.3f}")
        
        # 安全检查
        if confidence < self.confidence_threshold:
            print(f"置信度 {confidence:.3f} 低于阈值 {self.confidence_threshold}，跳过操作")
            return False
        
        # 边界检查（使用实际屏幕尺寸）
        if not (0 <= x <= self.screen_width and 0 <= y <= self.screen_height):
            print(f"坐标 ({x}, {y}) 超出屏幕范围 [0,0] - [{self.screen_width},{self.screen_height}]，跳过操作")
            return False
        
        try:
            if action_type == 'click':
                pyautogui.click(x, y)
            elif action_type == 'double_click':
                pyautogui.doubleClick(x, y)
            elif action_type == 'drag':
                # 从当前位置拖拽到目标位置
                current_x, current_y = pyautogui.position()
                pyautogui.dragTo(x, y, duration=0.5)
            elif action_type == 'keypress':
                # 简单按键操作
                pyautogui.press('enter')  # 示例操作
            elif action_type == 'type_text':
                # 文本输入操作
                pyautogui.typewrite("Hello World")  # 示例文本
            elif action_type == 'move':
                pyautogui.moveTo(x, y)
            elif action_type in ['button', 'text_field', 'icon', 'menu_bar', 'scroll_bar', 'container']:
                # UI元素类型操作，通常需要点击
                pyautogui.click(x, y)
            else:
                # 默认点击操作
                pyautogui.click(x, y)
            
            print(f"操作执行成功: {action_type} at ({x}, {y})")
            return True
            
        except Exception as e:
            print(f"执行操作时出错: {e}")
            return False
    
    def run_single_cycle(self) -> Dict[str, Any]:
        """运行单个预测-执行周期"""
        if self.emergency_stop:
            return {'prediction': None, 'execution_success': False, 'timestamp': time.time()}
        
        try:
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
        except Exception as e:
            print(f"执行周期出错: {e}")
            return {
                'prediction': None,
                'execution_success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def run_continuous(self, interval: float = 1.0):
        """连续运行桌面自动化"""
        print("桌面自动化系统启动...")
        print("按 Ctrl+C 或使用紧急停止命令停止")
        
        self.running = True
        try:
            while self.running and not self.emergency_stop:
                result = self.run_single_cycle()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n桌面自动化系统已停止")
        finally:
            self.running = False
    
    def emergency_stop_system(self):
        """紧急停止系统"""
        self.emergency_stop = True
        print("紧急停止已激活！")
    
    def reset_system(self):
        """重置系统"""
        self.emergency_stop = False
        print("系统已重置")
    
    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.confidence_threshold = threshold
        print(f"置信度阈值已设置为: {threshold}")
    
    def add_task_to_queue(self, task: Dict[str, Any]):
        """添加任务到队列"""
        self.task_queue.put(task)
        print(f"任务已添加到队列: {task['type']}")
    
    def process_task_queue(self):
        """处理任务队列"""
        while not self.task_queue.empty():
            task = self.task_queue.get()
            print(f"处理任务: {task['type']}")
            
            if task['type'] == 'click':
                pyautogui.click(task['x'], task['y'])
            elif task['type'] == 'type':
                pyautogui.typewrite(task['text'])
            elif task['type'] == 'press':
                pyautogui.press(task['key'])
            elif task['type'] == 'move':
                pyautogui.moveTo(task['x'], task['y'])
            elif task['type'] == 'custom':
                # 执行自定义操作
                self.execute_custom_action(task['action'])
    
    def execute_custom_action(self, action: str):
        """执行自定义操作"""
        print(f"执行自定义操作: {action}")
        # 这里可以根据需要扩展自定义操作
        if action == 'open_notepad':
            pyautogui.press('win')
            time.sleep(0.5)
            pyautogui.typewrite('notepad')
            time.sleep(0.5)
            pyautogui.press('enter')
        elif action == 'screenshot':
            timestamp = int(time.time())
            screenshot = pyautogui.screenshot()
            screenshot.save(f'screenshot_{timestamp}.png')
            print(f"截图已保存: screenshot_{timestamp}.png")
        elif action == 'open_chrome':
            pyautogui.press('win')
            time.sleep(0.5)
            pyautogui.typewrite('chrome')
            time.sleep(0.5)
            pyautogui.press('enter')
        elif action == 'open_explorer':
            pyautogui.hotkey('win', 'e')
    
    def interactive_mode(self):
        """交互模式 - 允许用户指定具体操作"""
        print("进入交互模式...")
        print("可用命令:")
        print("  click x y - 点击指定坐标")
        print("  type 'text' - 输入文本")
        print("  press key - 按键")
        print("  move x y - 移动鼠标")
        print("  custom action - 执行自定义操作")
        print("  quit - 退出交互模式")
        
        while True:
            command = input("\n请输入命令: ").strip()
            
            if command.lower() == 'quit':
                print("退出交互模式")
                break
            
            parts = command.split(' ', 2)  # 最多分割成3部分
            if len(parts) < 1:
                continue
            
            cmd_type = parts[0].lower()
            
            if cmd_type == 'click' and len(parts) >= 3:
                try:
                    x, y = int(parts[1]), int(parts[2])
                    if 0 <= x <= self.screen_width and 0 <= y <= self.screen_height:
                        pyautogui.click(x, y)
                        print(f"已点击 ({x}, {y})")
                    else:
                        print(f"坐标超出范围: ({x}, {y})")
                except ValueError:
                    print("坐标必须是数字")
            
            elif cmd_type == 'type' and len(parts) >= 2:
                text = parts[1].strip("'\"")  # 去除引号
                pyautogui.typewrite(text)
                print(f"已输入: {text}")
            
            elif cmd_type == 'press' and len(parts) >= 2:
                key = parts[1]
                pyautogui.press(key)
                print(f"已按键: {key}")
            
            elif cmd_type == 'move' and len(parts) >= 3:
                try:
                    x, y = int(parts[1]), int(parts[2])
                    if 0 <= x <= self.screen_width and 0 <= y <= self.screen_height:
                        pyautogui.moveTo(x, y)
                        print(f"已移动到 ({x}, {y})")
                    else:
                        print(f"坐标超出范围: ({x}, {y})")
                except ValueError:
                    print("坐标必须是数字")
            
            elif cmd_type == 'custom' and len(parts) >= 2:
                action = parts[1]
                self.execute_custom_action(action)
            
            else:
                print("无效命令或参数不足")

def main():
    """主函数"""
    print("=" * 60)
    print("AI桌面操作员 - 改进版桌面自动化任务")
    print("=" * 60)
    
    # 创建控制器实例
    controller = ImprovedDesktopAutomation()
    
    print("\n功能选项:")
    print("1. 单次预测和执行")
    print("2. 连续运行模式")
    print("3. 任务队列处理")
    print("4. 系统设置")
    print("5. 紧急停止")
    print("6. 交互模式")
    print("7. 退出")
    
    while True:
        choice = input("\n请选择功能 (1-7): ").strip()
        
        if choice == '1':
            print("执行单次预测...")
            result = controller.run_single_cycle()
            print(f"执行结果: {'成功' if result['execution_success'] else '失败'}")
        
        elif choice == '2':
            interval = input("请输入执行间隔（秒，默认1.0）: ").strip()
            try:
                interval = float(interval) if interval else 1.0
            except ValueError:
                interval = 1.0
            print(f"开始连续运行，间隔 {interval} 秒...")
            print("注意：按 Ctrl+C 停止运行")
            controller.run_continuous(interval)
        
        elif choice == '3':
            print("任务队列功能演示...")
            # 添加一些示例任务
            controller.add_task_to_queue({'type': 'click', 'x': 100, 'y': 100})
            controller.add_task_to_queue({'type': 'type', 'text': 'Hello World'})
            controller.add_task_to_queue({'type': 'custom', 'action': 'open_notepad'})
            
            # 处理任务队列
            controller.process_task_queue()
        
        elif choice == '4':
            print("\n系统设置:")
            print("1. 设置置信度阈值")
            print("2. 查看当前设置")
            
            sub_choice = input("请选择 (1-2): ").strip()
            if sub_choice == '1':
                try:
                    threshold = float(input("请输入置信度阈值 (0.0-1.0): "))
                    if 0.0 <= threshold <= 1.0:
                        controller.set_confidence_threshold(threshold)
                    else:
                        print("阈值必须在0.0到1.0之间")
                except ValueError:
                    print("请输入有效的数字")
            elif sub_choice == '2':
                print(f"当前置信度阈值: {controller.confidence_threshold}")
                print(f"安全机制: {'启用' if controller.safety_enabled else '禁用'}")
                print(f"紧急停止: {'激活' if controller.emergency_stop else '未激活'}")
                print(f"屏幕尺寸: {controller.screen_width}x{controller.screen_height}")
        
        elif choice == '5':
            controller.emergency_stop_system()
            reset = input("是否重置系统? (y/n): ").strip().lower()
            if reset == 'y':
                controller.reset_system()
        
        elif choice == '6':
            print("启动交互模式...")
            controller.interactive_mode()
        
        elif choice == '7':
            print("退出系统")
            break
        
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()