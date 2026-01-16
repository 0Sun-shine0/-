import sys
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                             QComboBox, QSlider, QGroupBox, QScrollArea)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np
from PIL import ImageGrab
import io

from ..core.config import Config
from ..agents.deepseek_agent import DeepSeekAgent
from ..models.cnn_model import EnhancedCNN
from ..automation.mouse_controller import AdvancedMouseController
from ..automation.keyboard_controller import KeyboardController
from ..automation.web_controller import WebController, MultiAppController, DataProcessor
from ..automation.workflow_manager import WorkflowManager, SmartDecisionEngine, Task
from ..automation.advanced_content_generator import AdvancedContentGenerator, SmartErrorHandler, PerformanceMonitor


class TaskWorker(QThread):
    """后台任务工作线程"""
    task_finished = pyqtSignal(object, str)  # 结果, 任务类型
    task_error = pyqtSignal(str, str)      # 错误信息, 任务类型
    
    def __init__(self, agent, task_text, task_type="deepseek"):
        super().__init__()
        self.agent = agent
        self.task_text = task_text
        self.task_type = task_type
    
    def run(self):
        try:
            if self.task_type == "deepseek":
                result = self.agent.process_task(self.task_text)
                self.task_finished.emit(result, self.task_type)
            elif self.task_type == "prediction":
                # 这里可以添加其他类型的预测任务
                pass
        except Exception as e:
            self.task_error.emit(str(e), self.task_type)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.deepseek_agent = DeepSeekAgent(self.config)
        self.cnn_model = EnhancedCNN(len(self.config.AVAILABLE_ACTIONS))
        self.mouse_controller = AdvancedMouseController()
        self.keyboard_controller = KeyboardController()
        
        # 新增功能控制器
        self.web_controller = WebController()
        self.multi_app_controller = MultiAppController()
        self.data_processor = DataProcessor()
        self.workflow_manager = WorkflowManager()
        self.smart_decision_engine = SmartDecisionEngine()
        
        # 高级功能
        self.advanced_content_generator = AdvancedContentGenerator()
        self.smart_error_handler = SmartErrorHandler()
        self.performance_monitor = PerformanceMonitor()
        
        # 初始化后台任务线程
        self.task_worker = None
        
        self.setup_ui()
        self.setup_timers()
        
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("AI智能桌面操作员")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧：屏幕捕获和显示区域
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 3)
        
        # 右侧：控制和信息面板
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)
        
    def create_left_panel(self):
        """创建左侧屏幕捕获面板"""
        left_group = QGroupBox("屏幕捕获")
        left_layout = QVBoxLayout()
        
        # 屏幕截图显示
        self.screen_label = QLabel()
        self.screen_label.setMinimumSize(600, 400)
        self.screen_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.screen_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.screen_label)
        
        # 屏幕捕获按钮
        capture_layout = QHBoxLayout()
        self.capture_button = QPushButton("捕获屏幕")
        self.capture_button.clicked.connect(self.capture_screen)
        capture_layout.addWidget(self.capture_button)
        
        self.auto_capture_checkbox = QPushButton("自动捕获")
        self.auto_capture_checkbox.setCheckable(True)
        self.auto_capture_checkbox.clicked.connect(self.toggle_auto_capture)
        capture_layout.addWidget(self.auto_capture_checkbox)
        
        left_layout.addLayout(capture_layout)
        
        # 预测结果显示
        prediction_group = QGroupBox("预测结果")
        prediction_layout = QVBoxLayout()
        
        self.prediction_text = QTextEdit()
        self.prediction_text.setMaximumHeight(150)
        self.prediction_text.setReadOnly(True)
        prediction_layout.addWidget(self.prediction_text)
        
        prediction_group.setLayout(prediction_layout)
        left_layout.addWidget(prediction_group)
        
        left_group.setLayout(left_layout)
        return left_group
        
    def create_right_panel(self):
        """创建右侧控制面板"""
        right_group = QGroupBox("控制面板")
        right_layout = QVBoxLayout()
        
        # 任务输入区域
        task_group = QGroupBox("任务输入")
        task_layout = QVBoxLayout()
        
        self.task_input = QTextEdit()
        self.task_input.setMaximumHeight(100)
        self.task_input.setPlaceholderText("请输入您的任务描述...")
        task_layout.addWidget(self.task_input)
        
        self.execute_button = QPushButton("执行任务")
        self.execute_button.clicked.connect(self.execute_task)
        task_layout.addWidget(self.execute_button)
        
        # 新增高级任务按钮
        advanced_task_layout = QHBoxLayout()
        self.workflow_button = QPushButton("执行工作流")
        self.workflow_button.clicked.connect(self.execute_workflow)
        advanced_task_layout.addWidget(self.workflow_button)
        
        self.smart_decision_button = QPushButton("智能决策")
        self.smart_decision_button.clicked.connect(self.execute_smart_decision)
        advanced_task_layout.addWidget(self.smart_decision_button)
        
        task_layout.addLayout(advanced_task_layout)
        
        # 新增高级功能按钮
        advanced_features_layout = QHBoxLayout()
        self.content_gen_button = QPushButton("高级内容生成")
        self.content_gen_button.clicked.connect(self.execute_advanced_content_generation)
        advanced_features_layout.addWidget(self.content_gen_button)
        
        self.performance_report_button = QPushButton("性能报告")
        self.performance_report_button.clicked.connect(self.show_performance_report)
        advanced_features_layout.addWidget(self.performance_report_button)
        
        task_layout.addLayout(advanced_features_layout)
        
        task_group.setLayout(task_layout)
        right_layout.addWidget(task_group)
        
        # 模型配置区域
        model_group = QGroupBox("模型配置")
        model_layout = QVBoxLayout()
        
        # 模型选择
        model_combo_layout = QHBoxLayout()
        model_combo_layout.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["EnhancedCNN", "Transformer", "Hybrid"])
        model_combo_layout.addWidget(self.model_combo)
        model_layout.addLayout(model_combo_layout)
        
        # 置信度阈值
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("置信度阈值:"))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(70)
        self.confidence_label = QLabel("0.70")
        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_label)
        
        self.confidence_slider.valueChanged.connect(
            lambda value: self.confidence_label.setText(f"{value/100:.2f}")
        )
        model_layout.addLayout(confidence_layout)
        
        model_group.setLayout(model_layout)
        right_layout.addWidget(model_group)
        
        # 操作历史
        history_group = QGroupBox("操作历史")
        history_layout = QVBoxLayout()
        
        self.history_text = QTextEdit()
        self.history_text.setMaximumHeight(200)
        self.history_text.setReadOnly(True)
        history_layout.addWidget(self.history_text)
        
        history_group.setLayout(history_layout)
        right_layout.addWidget(history_group)
        
        # 状态信息
        status_group = QGroupBox("状态")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("就绪")
        status_layout.addWidget(self.status_label)
        
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)
        
        # 添加弹性空间
        right_layout.addStretch()
        
        right_group.setLayout(right_layout)
        return right_group
        
    def setup_timers(self):
        """设置定时器"""
        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self.capture_screen)
        
    def capture_screen(self):
        """捕获屏幕"""
        try:
            # 使用PIL截取屏幕
            screenshot = ImageGrab.grab()
            screenshot = screenshot.convert('RGB')
            
            # 转换为OpenCV格式
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 调整大小以适应显示
            height, width = frame.shape[:2]
            scale = min(600/width, 400/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            
            # 转换为QPixmap并显示
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.screen_label.setPixmap(pixmap)
            
            # 使用CNN模型进行预测
            prediction = self.cnn_model.predict(frame)
            self.update_prediction_display(prediction)
            
        except Exception as e:
            self.status_label.setText(f"捕获屏幕失败: {str(e)}")
            
            # 即使捕获失败，也要提供一个默认的预测结果
            # 以便AI可以继续处理任务
            default_prediction = {
                'top_actions': [{'action': 0, 'confidence': 0.1}],
                'top_applications': [{'app': 0, 'confidence': 0.1}],
                'bounding_boxes': [0, 0, 0, 0],
                'mouse_position': [0, 0],
                'segmentation_map': np.zeros((10, 10)),
                'confidence_scores': {
                    'action_max': 0.1,
                    'app_max': 0.1
                }
            }
            self.update_prediction_display(default_prediction)
            
    def update_prediction_display(self, prediction):
        """更新预测结果显示"""
        try:
            prediction_str = json.dumps(prediction, indent=2, ensure_ascii=False)
            self.prediction_text.setPlainText(prediction_str)
        except Exception as e:
            self.prediction_text.setPlainText(f"显示预测结果失败: {str(e)}")
            
    def toggle_auto_capture(self, checked):
        """切换自动捕获"""
        if checked:
            self.capture_timer.start(1000)  # 每秒捕获一次
            self.status_label.setText("自动捕获已启用")
        else:
            self.capture_timer.stop()
            self.status_label.setText("自动捕获已禁用")
            
    def execute_task(self):
        """执行用户任务"""
        task_text = self.task_input.toPlainText().strip()
        if not task_text:
            self.status_label.setText("请输入任务描述")
            return
            
        self.status_label.setText("正在处理任务...")
        
        # 创建后台任务线程
        self.task_worker = TaskWorker(self.deepseek_agent, task_text, "deepseek")
        self.task_worker.task_finished.connect(self.on_task_finished)
        self.task_worker.task_error.connect(self.on_task_error)
        self.task_worker.start()
    
    def execute_workflow(self):
        """执行工作流"""
        task_text = self.task_input.toPlainText().strip()
        if not task_text:
            self.status_label.setText("请输入任务描述")
            return
            
        self.status_label.setText("正在执行工作流...")
        
        # 根据任务描述创建工作流
        try:
            # 这里可以根据任务描述动态创建工作流
            # 简单示例：创建一个数据处理工作流
            tasks = [
                Task(
                    id="extract_data",
                    name="提取数据",
                    action="extract_data",
                    parameters={"text": task_text},
                    dependencies=[]
                ),
                Task(
                    id="process_data",
                    name="处理数据",
                    action="process_data",
                    parameters={},
                    dependencies=["extract_data"]
                ),
                Task(
                    id="generate_report",
                    name="生成报告",
                    action="generate_report",
                    parameters={"title": "任务处理报告"},
                    dependencies=["process_data"]
                )
            ]
            
            for task in tasks:
                self.workflow_manager.add_task(task)
            
            result = self.workflow_manager.execute_workflow()
            self.status_label.setText(f"工作流执行完成: {result['status']}")
            
            # 记录操作历史
            self.history_text.append(f"工作流任务: {task_text}\n结果: {result}\n---")
            
        except Exception as e:
            self.status_label.setText(f"执行工作流失败: {str(e)}")
    
    def execute_smart_decision(self):
        """执行智能决策"""
        task_text = self.task_input.toPlainText().strip()
        if not task_text:
            self.status_label.setText("请输入任务描述")
            return
            
        self.status_label.setText("正在执行智能决策...")
        
        try:
            # 使用智能决策引擎处理任务
            tasks = self.smart_decision_engine.process_user_request(task_text)
            
            if tasks:
                # 执行生成的任务
                for task_info in tasks:
                    self.execute_task_from_info(task_info)
                
                self.status_label.setText("智能决策执行完成")
                
                # 记录操作历史
                self.history_text.append(f"智能决策任务: {task_text}\n生成任务数: {len(tasks)}\n---")
            else:
                self.status_label.setText("未能生成可执行任务")
                
        except Exception as e:
            self.status_label.setText(f"执行智能决策失败: {str(e)}")
    
    def execute_advanced_content_generation(self):
        """执行高级内容生成"""
        task_text = self.task_input.toPlainText().strip()
        if not task_text:
            self.status_label.setText("请输入任务描述")
            return
            
        self.status_label.setText("正在生成高级内容...")
        
        try:
            # 解析任务，确定内容类型和主题
            content_type = "report"  # 默认类型
            if "邮件" in task_text or "email" in task_text.lower():
                content_type = "email"
            elif "故事" in task_text or "小说" in task_text or "story" in task_text.lower():
                content_type = "story"
            
            # 提取主题
            topic = task_text.split("关于")[1] if "关于" in task_text else task_text.split("写")[1] if "写" in task_text else task_text
            
            # 生成内容
            generated_content = self.advanced_content_generator.generate_content(content_type, topic, "medium")
            
            # 尝试启动Word并输入内容
            self.execute_launch_application("启动Word")
            import time
            time.sleep(3)  # 等待Word启动
            
            # 确保Word窗口获得焦点
            import pyautogui
            try:
                pyautogui.click(100, 100)  # 点击Word窗口区域以获得焦点
                time.sleep(0.5)
            except:
                pass  # 如果pyautogui不可用，跳过焦点设置
            
            # 直接使用pyautogui输入内容，绕过键盘控制器
            try:
                pyautogui.write(generated_content, interval=0.02)
            except:
                # 如果write失败，尝试使用typewrite
                try:
                    pyautogui.typewrite(generated_content, interval=0.02)
                except:
                    # 如果还是失败，分段输入
                    for char in generated_content:
                        try:
                            pyautogui.write(char, interval=0.02)
                        except:
                            # 如果单个字符输入也失败，跳过
                            continue
            
            self.status_label.setText("高级内容生成完成")
            
            # 记录操作历史
            self.history_text.append(f"高级内容生成: {task_text}\n内容类型: {content_type}\n---")
            
        except Exception as e:
            self.status_label.setText(f"高级内容生成失败: {str(e)}")
    
    def show_performance_report(self):
        """显示性能报告"""
        self.status_label.setText("正在生成性能报告...")
        
        try:
            report = self.performance_monitor.get_performance_report()
            
            # 显示性能报告
            report_text = f"""
性能报告:
- 运行时间: {report['uptime']:.2f} 秒
- 总操作数: {report['total_operations']}
- 成功操作数: {report['successful_operations']}
- 成功率: {report['success_rate']:.2%}
- 平均执行时间: {report['average_execution_time']:.3f} 秒

操作统计:
"""
            for op, count in report['operation_breakdown'].items():
                op_efficiency = self.performance_monitor.get_operation_efficiency(op)
                report_text += f"- {op}: {count} 次 (成功率: {op_efficiency['success_rate']:.2%}, 平均时间: {op_efficiency['avg_time']:.3f}秒)\n"
            
            # 在历史记录中显示报告
            self.history_text.append(f"性能报告:\n{report_text}\n---")
            self.status_label.setText("性能报告已生成")
            
        except Exception as e:
            self.status_label.setText(f"生成性能报告失败: {str(e)}")
    
    def execute_task_from_info(self, task_info):
        """根据任务信息执行任务"""
        task_id = task_info.get("id", "")
        action = task_info.get("action", "")
        params = task_info.get("parameters", {})
        
        if action == "start_app":
            app_name = params.get("app_name", "")
            self.multi_app_controller.start_application(app_name)
        elif action == "type_text":
            text = params.get("text", "")
            import pyautogui
            pyautogui.write(text, interval=0.02)
        elif action == "click":
            x = params.get("x", 0)
            y = params.get("y", 0)
            import pyautogui
            pyautogui.click(x, y)
    
    def on_task_finished(self, result, task_type):
        """任务完成回调"""
        task_text = self.task_input.toPlainText().strip()
            
        # 记录操作历史
        self.history_text.append(f"任务: {task_text}\n结果: {result}\n---")
            
        # 执行预测的动作
        self.execute_predicted_actions(result)
            
        self.status_label.setText("任务执行完成")
            
        # 清理线程对象
        if self.task_worker:
            self.task_worker = None
        
    def on_task_error(self, error_msg, task_type):
        """任务错误回调"""
        self.status_label.setText(f"执行任务失败: {error_msg}")
            
        # 清理线程对象
        if self.task_worker:
            self.task_worker = None
            
    def execute_predicted_actions(self, result):
        """执行预测的动作"""
        try:
            # 解析DeepSeek的结果并执行相应的操作
            if isinstance(result, dict):
                # 检查是否是DeepSeek的分析结果格式
                if 'task_analysis' in result and 'steps' in result:
                    # 这是DeepSeek的分析结果，需要解析步骤并执行
                    self.execute_deepseek_analysis(result)
                else:
                    # 这是标准的动作格式
                    actions = result.get('actions', [])
                    for action in actions:
                        self.execute_single_action(action)
            elif isinstance(result, str):
                # 如果是字符串，尝试解析为JSON
                try:
                    parsed_result = json.loads(result)
                    if 'task_analysis' in parsed_result and 'steps' in parsed_result:
                        # 这是DeepSeek的分析结果
                        self.execute_deepseek_analysis(parsed_result)
                    else:
                        # 这是标准的动作格式
                        actions = parsed_result.get('actions', [])
                        for action in actions:
                            self.execute_single_action(action)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，可能是一个简单的指令
                    self.execute_simple_action(result)
        except Exception as e:
            # 使用智能错误处理器处理错误
            error_result = self.smart_error_handler.handle_error("execution_error", str(e), {
                "result": result,
                "operation": "execute_predicted_actions"
            })
            self.status_label.setText(f"执行动作失败: {str(e)}")
            
    def execute_deepseek_analysis(self, analysis_result):
        """执行DeepSeek分析结果中的步骤或内容"""
        try:
            task_analysis = analysis_result.get('task_analysis', '')
            content = analysis_result.get('content', None)
            steps = analysis_result.get('steps', [])
            
            # 检查是否有直接生成的内容
            if content is not None:
                # 这是内容生成任务，直接处理生成的内容
                self.handle_generated_content(content)
            elif steps:
                # 这是操作任务，执行步骤
                self.execute_operation_steps(steps)
            else:
                self.status_label.setText("没有找到可执行的步骤或内容")
                
        except Exception as e:
            # 使用智能错误处理器处理错误
            error_result = self.smart_error_handler.handle_error("analysis_error", str(e), {
                "analysis_result": analysis_result,
                "operation": "execute_deepseek_analysis"
            })
            self.status_label.setText(f"执行分析结果失败: {str(e)}")
    
    def handle_generated_content(self, content):
        """处理生成的内容"""
        try:
            self.status_label.setText("正在处理生成的内容...")
            QApplication.processEvents()  # 更新UI
            
            # 如果内容是字符串，直接处理
            if isinstance(content, str):
                # 尝试启动Word并输入内容
                self.execute_launch_application("启动Word")
                import time
                time.sleep(5)  # 增加等待时间，确保Word完全加载
                
                # 确保Word窗口获得焦点
                import pyautogui
                try:
                    pyautogui.click(100, 100)  # 点击Word窗口区域以获得焦点
                    time.sleep(0.5)
                except:
                    pass  # 如果pyautogui不可用，跳过焦点设置
                
                # 直接使用pyautogui输入内容，绕过键盘控制器
                try:
                    pyautogui.write(content, interval=0.02)
                except:
                    # 如果write失败，尝试使用typewrite
                    try:
                        pyautogui.typewrite(content, interval=0.02)
                    except:
                        # 如果还是失败，分段输入
                        for char in content:
                            try:
                                pyautogui.write(char, interval=0.02)
                            except:
                                # 如果单个字符输入也失败，跳过
                                continue
                
            # 如果内容是字典或其他格式，根据具体格式处理
            elif isinstance(content, dict):
                text_content = content.get('text', '') or content.get('content', '') or str(content)
                self.execute_launch_application("启动Word")
                import time
                time.sleep(5)  # 增加等待时间，确保Word完全加载
                
                # 确保Word窗口获得焦点
                import pyautogui
                try:
                    pyautogui.click(100, 100)  # 点击Word窗口区域以获得焦点
                    time.sleep(0.5)
                except:
                    pass  # 如果pyautogui不可用，跳过焦点设置
                
                # 直接使用pyautogui输入内容，绕过键盘控制器
                try:
                    pyautogui.write(text_content, interval=0.02)
                except:
                    # 如果write失败，尝试使用typewrite
                    try:
                        pyautogui.typewrite(text_content, interval=0.02)
                    except:
                        # 如果还是失败，分段输入
                        for char in text_content:
                            try:
                                pyautogui.write(char, interval=0.02)
                            except:
                                # 如果单个字符输入也失败，跳过
                                continue
            
            self.status_label.setText("内容已处理完成")
        except Exception as e:
            # 使用智能错误处理器处理错误
            error_result = self.smart_error_handler.handle_error("content_handling_error", str(e), {
                "content": content,
                "operation": "handle_generated_content"
            })
            self.status_label.setText(f"处理内容失败: {str(e)}")
    
    def execute_operation_steps(self, steps):
        """执行操作步骤"""
        try:
            for step in steps:
                step_desc = step.get('description', '')
                self.status_label.setText(f"执行步骤: {step_desc}")
                QApplication.processEvents()  # 更新UI
                
                # 解析步骤描述并执行相应操作
                if '启动' in step_desc or '打开' in step_desc or 'word' in step_desc.lower() or 'Word' in step_desc:
                    self.execute_launch_application(step_desc)
                elif '新建' in step_desc or '新文档' in step_desc:
                    self.execute_new_document(step_desc)
                elif '输入' in step_desc or '标题' in step_desc:
                    self.execute_text_input(step_desc)
                elif '点击' in step_desc or '单击' in step_desc:
                    self.execute_click_operation(step_desc)
                elif '保存' in step_desc:
                    self.execute_save_operation(step_desc)
                elif '格式' in step_desc or '样式' in step_desc:
                    self.execute_format_operation(step_desc)
                
                # 延长延迟以确保操作完成，并确保Word完全加载
                import time
                time.sleep(2)
                
                # 确保Word窗口获得焦点
                try:
                    import pyautogui
                    # 尝试点击Word窗口中心以获得焦点
                    pyautogui.click(500, 300)  # 点击Word窗口区域
                    time.sleep(0.5)
                except:
                    pass  # 如果pyautogui不可用，跳过焦点设置
                
            self.status_label.setText("所有步骤执行完成")
        except Exception as e:
            # 使用智能错误处理器处理错误
            error_result = self.smart_error_handler.handle_error("operation_steps_error", str(e), {
                "steps": steps,
                "operation": "execute_operation_steps"
            })
            self.status_label.setText(f"执行操作步骤失败: {str(e)}")
    
    def execute_launch_application(self, description):
        """执行启动应用程序操作"""
        if 'word' in description.lower() or 'Word' in description:
            # 尝试启动Word
            import subprocess
            try:
                # 尝试多种启动Word的方式
                word_paths = [
                    r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
                    r"C:\Program Files (x86)\Microsoft Office\root\Office16\WINWORD.EXE",
                    r"C:\Program Files\Microsoft Office\Office16\WINWORD.EXE",
                    r"C:\Program Files (x86)\Microsoft Office\Office16\WINWORD.EXE",
                    r"C:\Program Files\Microsoft Office\root\Office15\WINWORD.EXE",
                    r"C:\Program Files (x86)\Microsoft Office\root\Office15\WINWORD.EXE",
                    r"C:\Program Files\Microsoft Office\Office15\WINWORD.EXE",
                    r"C:\Program Files (x86)\Microsoft Office\Office15\WINWORD.EXE",
                ]
                
                for path in word_paths:
                    try:
                        subprocess.Popen([path])
                        self.status_label.setText("Word已启动")
                        import time
                        time.sleep(3)  # 等待Word启动
                        return
                    except FileNotFoundError:
                        continue
                
                # 如果找不到路径，尝试使用系统命令启动
                try:
                    subprocess.Popen(['start', 'winword'], shell=True)
                    self.status_label.setText("Word已启动")
                    import time
                    time.sleep(3)  # 等待Word启动
                except Exception:
                    self.status_label.setText("无法启动Word，请确认是否已安装")
            except Exception as e:
                # 使用智能错误处理器处理错误
                error_result = self.smart_error_handler.handle_error("app_launch_error", str(e), {
                    "description": description,
                    "operation": "execute_launch_application"
                })
                self.status_label.setText(f"启动Word失败: {str(e)}")
    
    def execute_new_document(self, description):
        """执行新建文档操作"""
        try:
            import pyautogui
            import time
            
            # 使用快捷键Ctrl+N新建文档
            pyautogui.hotkey('ctrl', 'n')
            time.sleep(1)  # 等待新文档创建
            
            time.sleep(2)  # 等待新文档完全创建
            self.status_label.setText("新文档已创建")
        except Exception as e:
            # 使用智能错误处理器处理错误
            error_result = self.smart_error_handler.handle_error("new_document_error", str(e), {
                "description": description,
                "operation": "execute_new_document"
            })
            self.status_label.setText(f"新建文档失败: {str(e)}")
    
    def execute_text_input(self, description):
        """执行文本输入操作"""
        import time
        time.sleep(0.5)  # 确保焦点在正确位置
        
        # 检查是否需要输入标题
        if '标题' in description:
            # 从描述中提取标题
            import re
            title_match = re.search(r'《([^》]+)》', description)
            if title_match:
                title = f"《{title_match.group(1)}》"
            else:
                # 如果没有找到特定标题，使用通用标题
                title = "文档标题"
            
            # 直接使用pyautogui输入标题，绕过键盘控制器
            import pyautogui
            try:
                pyautogui.write(title, interval=0.02)
            except:
                # 如果write失败，尝试使用typewrite
                try:
                    pyautogui.typewrite(title, interval=0.02)
                except:
                    # 如果还是失败，分段输入
                    for char in title:
                        try:
                            pyautogui.write(char, interval=0.02)
                        except:
                            continue
            
            # 按回车键
            pyautogui.press('enter')
            pyautogui.press('enter')  # 空一行
    
    def execute_click_operation(self, description):
        """执行点击操作"""
        # 根据描述执行点击操作
        pass
    
    def execute_save_operation(self, description):
        """执行保存操作"""
        try:
            import pyautogui
            import time
            
            # 使用快捷键Ctrl+S保存
            pyautogui.hotkey('ctrl', 's')
            time.sleep(1)
            
            # 输入文件名并保存
            pyautogui.typewrite('我的前半生小说.docx')
            time.sleep(0.5)
            pyautogui.press('enter')
        except ImportError:
            # 如果pyautogui未安装，使用键盘控制器
            self.status_label.setText("pyautogui未安装，使用键盘控制器进行保存")
        except Exception as e:
            # 使用智能错误处理器处理错误
            error_result = self.smart_error_handler.handle_error("save_operation_error", str(e), {
                "description": description,
                "operation": "execute_save_operation"
            })
            self.status_label.setText(f"保存操作失败: {str(e)}")
    
    def execute_format_operation(self, description):
        """执行格式化操作"""
        try:
            import pyautogui
            import time
            
            # 根据描述执行不同的格式化操作
            if '居中' in description:
                pyautogui.hotkey('ctrl', 'e')  # 居中对齐
            elif '加粗' in description:
                pyautogui.hotkey('ctrl', 'b')  # 加粗
            elif '斜体' in description:
                pyautogui.hotkey('ctrl', 'i')  # 斜体
            elif '下划线' in description:
                pyautogui.hotkey('ctrl', 'u')  # 下划线
            elif '字体' in description:
                pyautogui.hotkey('ctrl', 'd')  # 字体设置
            elif '编号' in description:
                pyautogui.hotkey('ctrl', 'shift', 'l')  # 编号列表
            elif '项目符号' in description:
                pyautogui.hotkey('ctrl', 'shift', '8')  # 项目符号
            
            time.sleep(0.5)
            
            time.sleep(1)  # 等待格式化操作完成
            self.status_label.setText("格式化操作完成")
        except Exception as e:
            # 使用智能错误处理器处理错误
            error_result = self.smart_error_handler.handle_error("format_operation_error", str(e), {
                "description": description,
                "operation": "execute_format_operation"
            })
            self.status_label.setText(f"格式化操作失败: {str(e)}")
    
    def execute_single_action(self, action):
        """执行单个动作"""
        action_type = action.get('type', '')
        params = action.get('params', {})
        
        if action_type == 'click':
            x, y = params.get('x', 0), params.get('y', 0)
            self.mouse_controller.click(x, y)
        elif action_type == 'double_click':
            x, y = params.get('x', 0), params.get('y', 0)
            self.mouse_controller.double_click(x, y)
        elif action_type == 'drag':
            start_x, start_y = params.get('start_x', 0), params.get('start_y', 0)
            end_x, end_y = params.get('end_x', 0), params.get('end_y', 0)
            self.mouse_controller.drag(start_x, start_y, end_x, end_y)
        elif action_type == 'keypress':
            key = params.get('key', '')
            # 使用pyautogui直接输入，绕过键盘控制器
            import pyautogui
            try:
                pyautogui.write(key, interval=0.02)
            except:
                try:
                    pyautogui.typewrite(key, interval=0.02)
                except:
                    for char in key:
                        try:
                            pyautogui.write(char, interval=0.02)
                        except:
                            continue
        elif action_type == 'type_text':
            text = params.get('text', '')
            # 使用pyautogui直接输入，绕过键盘控制器
            import pyautogui
            try:
                pyautogui.write(text, interval=0.02)
            except:
                try:
                    pyautogui.typewrite(text, interval=0.02)
                except:
                    for char in text:
                        try:
                            pyautogui.write(char, interval=0.02)
                        except:
                            continue
        # 添加更多动作类型...
        
    def execute_simple_action(self, action_str):
        """执行简单动作字符串"""
        # 这里可以根据简单的指令字符串执行操作
        # 例如：解析自然语言指令
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())