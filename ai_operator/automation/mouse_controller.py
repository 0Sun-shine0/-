"""
高级鼠标控制器 - 支持多种鼠标操作和手势识别
"""
import pyautogui
import time
import math
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from enum import Enum

class MouseButton(Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"

@dataclass
class MouseGesture:
    """鼠标手势定义"""
    name: str
    pattern: List[Tuple[int, int]]  # 相对移动轨迹
    action: str  # 对应的操作

class AdvancedMouseController:
    """高级鼠标控制器"""
    
    def __init__(self, smoothness: float = 0.7, speed: float = 1.0):
        self.smoothness = smoothness  # 平滑度 (0-1)
        self.speed = speed  # 移动速度
        self.current_position = pyautogui.position()
        self.gestures = self._initialize_gestures()
        self.is_dragging = False
        
    def _initialize_gestures(self) -> List[MouseGesture]:
        """初始化预定义手势"""
        return [
            MouseGesture(
                name="circle_clockwise",
                pattern=[(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)],
                action="undo"
            ),
            MouseGesture(
                name="circle_counter_clockwise",
                pattern=[(0, 0), (0, 20), (20, 20), (20, 0), (0, 0)],
                action="redo"
            ),
            MouseGesture(
                name="check_mark",
                pattern=[(0, 0), (10, 10), (20, 0)],
                action="confirm"
            ),
            MouseGesture(
                name="x_mark",
                pattern=[(0, 0), (20, 20), (0, 20), (20, 0)],
                action="cancel"
            )
        ]
    
    def move_to(self, target_x: int, target_y: int, duration: float = 0.5):
        """平滑移动到指定位置"""
        start_x, start_y = self.current_position
        
        # 计算贝塞尔曲线控制点
        control_points = self._calculate_bezier_points(
            start_x, start_y, target_x, target_y
        )
        
        # 生成平滑轨迹
        trajectory = self._generate_trajectory(control_points, duration)
        
        # 执行移动
        for point in trajectory:
            pyautogui.moveTo(point[0], point[1], duration=0)
            time.sleep(duration / len(trajectory))
        
        self.current_position = (target_x, target_y)
    
    def _calculate_bezier_points(self, start_x: int, start_y: int, 
                                target_x: int, target_y: int) -> List[Tuple[int, int]]:
        """计算贝塞尔曲线控制点"""
        # 计算中间控制点
        distance = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
        
        # 根据距离调整曲线强度
        curve_strength = min(100, distance * 0.3)
        
        # 计算垂直方向的控制点
        if abs(target_x - start_x) > abs(target_y - start_y):
            # 水平移动为主，垂直方向加曲线
            control1 = (start_x + (target_x - start_x) // 3, 
                       start_y + curve_strength)
            control2 = (start_x + 2 * (target_x - start_x) // 3,
                       target_y - curve_strength)
        else:
            # 垂直移动为主，水平方向加曲线
            control1 = (start_x + curve_strength,
                       start_y + (target_y - start_y) // 3)
            control2 = (target_x - curve_strength,
                       start_y + 2 * (target_y - start_y) // 3)
        
        return [(start_x, start_y), control1, control2, (target_x, target_y)]
    
    def _generate_trajectory(self, control_points: List[Tuple[int, int]], 
                            duration: float) -> List[Tuple[int, int]]:
        """生成平滑轨迹"""
        num_points = max(10, int(duration * 100 * self.speed))
        trajectory = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # 三次贝塞尔曲线
            x = (1 - t)**3 * control_points[0][0] + \
                3 * (1 - t)**2 * t * control_points[1][0] + \
                3 * (1 - t) * t**2 * control_points[2][0] + \
                t**3 * control_points[3][0]
            
            y = (1 - t)**3 * control_points[0][1] + \
                3 * (1 - t)**2 * t * control_points[1][1] + \
                3 * (1 - t) * t**2 * control_points[2][1] + \
                t**3 * control_points[3][1]
            
            trajectory.append((int(x), int(y)))
        
        return trajectory
    
    def click(self, x: Optional[int] = None, y: Optional[int] = None, 
              button: MouseButton = MouseButton.LEFT, clicks: int = 1, hold_time: float = 0.0):
        """点击操作，支持按钮、点击次数和按住时长"""
        if x is not None and y is not None:
            self.move_to(x, y)

        btn = button.value

        # 简单点击
        if hold_time <= 0:
            pyautogui.click(button=btn, clicks=clicks)
            return

        # 按住一定时长（用于拖拽前的按住或长按）
        pyautogui.mouseDown(button=btn)
        time.sleep(hold_time)
        pyautogui.mouseUp(button=btn)
    
    def double_click(self, x: Optional[int] = None, y: Optional[int] = None):
        """双击操作"""
        self.click(x, y, MouseButton.LEFT, clicks=2)

    def right_double_click(self, x: Optional[int] = None, y: Optional[int] = None):
        """右键双击（某些环境用于特殊操作）"""
        self.click(x, y, MouseButton.RIGHT, clicks=2)

    def click_and_hold(self, x: Optional[int] = None, y: Optional[int] = None,
                       button: MouseButton = MouseButton.LEFT, hold_time: float = 1.0):
        """在位置点击并保持（可用于拖拽起始）"""
        self.click(x, y, button=button, clicks=1, hold_time=hold_time)
    
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, 
            duration: float = 0.5):
        """拖拽操作"""
        self.move_to(start_x, start_y)
        
        # 按下鼠标
        pyautogui.mouseDown()
        self.is_dragging = True
        
        # 移动到目标位置
        self.move_to(end_x, end_y, duration)
        
        # 释放鼠标
        pyautogui.mouseUp()
        self.is_dragging = False

    def right_click_and_hold(self, x: Optional[int] = None, y: Optional[int] = None, hold_time: float = 1.0):
        """右键按住"""
        self.click_and_hold(x, y, button=MouseButton.RIGHT, hold_time=hold_time)
    
    def scroll(self, amount: int, direction: str = "vertical"):
        """滚动操作"""
        if direction == "vertical":
            pyautogui.scroll(amount)
        elif direction == "horizontal":
            pyautogui.hscroll(amount)

    def smooth_scroll(self, total_amount: int, steps: int = 10, direction: str = "vertical", delay: float = 0.01):
        """平滑滚动：将滚动分解为多个小步，模拟自然滚轮动作"""
        if steps <= 0:
            steps = 1

        step_amount = int(total_amount / steps)
        for i in range(steps):
            if direction == "vertical":
                pyautogui.scroll(step_amount)
            else:
                pyautogui.hscroll(step_amount)
            time.sleep(delay)

    def scroll_to(self, target: int, current: Optional[int] = None, steps: int = 20, direction: str = "vertical"):
        """滚动到相对目标（以当前为基准，如需要可先测量）"""
        if current is None:
            # 无法获取屏幕滚动位置时使用平滑滚动近似
            self.smooth_scroll(target, steps=steps, direction=direction)
            return

        delta = target - current
        self.smooth_scroll(delta, steps=steps, direction=direction)
    
    def gesture_recognition(self, points: List[Tuple[int, int]]) -> Optional[str]:
        """手势识别"""
        if len(points) < 3:
            return None
        
        # 归一化点序列
        normalized = self._normalize_points(points)
        
        # 匹配手势
        for gesture in self.gestures:
            if self._match_gesture(normalized, gesture.pattern):
                return gesture.action
        
        return None
    
    def _normalize_points(self, points: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """归一化点序列"""
        if not points:
            return []
        
        # 转换为相对坐标
        start_x, start_y = points[0]
        normalized = []
        
        for x, y in points:
            normalized.append((x - start_x, y - start_y))
        
        # 归一化到单位正方形
        max_x = max(abs(x) for x, _ in normalized) or 1
        max_y = max(abs(y) for _, y in normalized) or 1
        
        return [(x / max_x, y / max_y) for x, y in normalized]
    
    def _match_gesture(self, points1: List[Tuple[float, float]], 
                      points2: List[Tuple[float, float]], 
                      threshold: float = 0.3) -> bool:
        """匹配手势"""
        # 动态时间规整(DTW)距离
        distance = self._dtw_distance(points1, points2)
        
        # 归一化距离
        normalized_distance = distance / max(len(points1), len(points2))
        
        return normalized_distance < threshold
    
    def _dtw_distance(self, seq1: List[Tuple[float, float]], 
                     seq2: List[Tuple[float, float]]) -> float:
        """计算动态时间规整距离"""
        n, m = len(seq1), len(seq2)
        dtw = np.zeros((n + 1, m + 1))
        dtw[1:, 0] = float('inf')
        dtw[0, 1:] = float('inf')
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self._euclidean_distance(seq1[i-1], seq2[j-1])
                dtw[i, j] = cost + min(dtw[i-1, j],    # 插入
                                      dtw[i, j-1],    # 删除
                                      dtw[i-1, j-1])  # 匹配
        
        return dtw[n, m]
    
    def _euclidean_distance(self, p1: Tuple[float, float], 
                           p2: Tuple[float, float]) -> float:
        """计算欧几里得距离"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def smart_click(self, element_type: str, confidence: float = 0.8):
        """智能点击 - 根据元素类型调整点击方式"""
        current_x, current_y = self.current_position
        
        if element_type == "button":
            # 按钮：点击中心
            self.click(current_x, current_y)
            time.sleep(0.1)
            
        elif element_type == "checkbox":
            # 复选框：可能需要精确点击
            self.click(current_x, current_y)
            
        elif element_type == "link":
            # 链接：快速单击
            self.click(current_x, current_y)
            
        elif element_type == "icon":
            # 图标：可能双击
            if confidence > 0.9:
                self.double_click(current_x, current_y)
            else:
                self.click(current_x, current_y)
                
        elif element_type == "slider":
            # 滑块：可能需要拖拽
            self.click(current_x, current_y)
            
        else:
            # 默认点击
            self.click(current_x, current_y)
    
    def record_movement(self, duration: float = 5.0) -> List[Tuple[int, int]]:
        """记录鼠标移动轨迹"""
        points = []
        end_time = time.time() + duration
        
        while time.time() < end_time:
            x, y = pyautogui.position()
            points.append((x, y))
            time.sleep(0.01)
        
        return points
    
    def replay_movement(self, points: List[Tuple[int, int]], 
                       speed_factor: float = 1.0):
        """重放鼠标移动轨迹"""
        if not points:
            return
        
        total_points = len(points)
        interval = 0.01 / speed_factor
        
        for i, (x, y) in enumerate(points):
            # 使用平滑移动
            self.move_to(x, y, duration=interval)
            time.sleep(interval)
            
            # 显示进度
            if i % 10 == 0:
                progress = (i + 1) / total_points * 100
                print(f"重放进度: {progress:.1f}%")