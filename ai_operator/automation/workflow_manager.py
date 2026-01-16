"""
工作流管理器 - 实现跨应用的复杂任务自动化流程
"""
import time
import json
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """任务定义"""
    id: str
    name: str
    action: str  # 任务类型，如 'web_action', 'app_action', 'data_action'
    parameters: Dict[str, Any]
    dependencies: List[str]  # 依赖的其他任务ID
    condition: str = None  # 执行条件


class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self):
        self.tasks = {}
        self.status = {}
        self.results = {}
        self.web_controller = None
        self.multi_app_controller = None
        self.data_processor = None
        
        # 导入控制器
        from .web_controller import WebController, MultiAppController, DataProcessor
        self.web_controller = WebController()
        self.multi_app_controller = MultiAppController()
        self.data_processor = DataProcessor()
    
    def add_task(self, task: Task):
        """添加任务到工作流"""
        self.tasks[task.id] = task
        self.status[task.id] = TaskStatus.PENDING
        self.results[task.id] = None
    
    def execute_task(self, task_id: str) -> bool:
        """执行单个任务"""
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        self.status[task_id] = TaskStatus.RUNNING
        
        try:
            # 检查依赖
            for dep_id in task.dependencies:
                if self.status[dep_id] != TaskStatus.COMPLETED:
                    print(f"任务 {task_id} 依赖的任务 {dep_id} 未完成")
                    self.status[task_id] = TaskStatus.FAILED
                    return False
            
            # 检查条件
            if task.condition and not self.evaluate_condition(task.condition):
                print(f"任务 {task_id} 条件不满足: {task.condition}")
                self.status[task_id] = TaskStatus.CANCELLED
                return True  # 条件不满足不算是失败
            
            # 执行任务
            result = self._perform_action(task.action, task.parameters)
            self.results[task_id] = result
            self.status[task_id] = TaskStatus.COMPLETED
            print(f"任务 {task_id} 执行成功")
            return True
            
        except Exception as e:
            print(f"任务 {task_id} 执行失败: {e}")
            self.status[task_id] = TaskStatus.FAILED
            return False
    
    def _perform_action(self, action: str, parameters: Dict[str, Any]):
        """执行具体动作"""
        if action == "open_url":
            # 打开网页
            url = parameters.get("url")
            return self.web_controller.open_url(url)
        
        elif action == "click_element":
            # 点击网页元素
            selector = parameters.get("selector")
            by = parameters.get("by", "css")
            return self.web_controller.click_element(selector, by)
        
        elif action == "input_text":
            # 输入文本到网页元素
            selector = parameters.get("selector")
            text = parameters.get("text")
            by = parameters.get("by", "css")
            return self.web_controller.input_text(selector, text, by)
        
        elif action == "start_app":
            # 启动应用
            app_name = parameters.get("app_name")
            return self.multi_app_controller.start_application(app_name)
        
        elif action == "extract_data":
            # 提取数据
            text = parameters.get("text")
            return self.data_processor.extract_data_from_text(text)
        
        elif action == "process_data":
            # 处理数据
            data = parameters.get("data")
            if data is None:
                # 如果没有提供数据，尝试从之前的任务结果中获取
                data = self.results.get("extract", {})
            return self.data_processor.process_data(data)
        
        elif action == "generate_report":
            # 生成报告
            data = parameters.get("data")
            if data is None:
                # 如果没有提供数据，尝试从之前的任务结果中获取
                data = self.results.get("process", {})
            title = parameters.get("title", "数据报告")
            return self.data_processor.generate_report(data, title)
        
        elif action == "type_text":
            # 输入文本到当前焦点
            text = parameters.get("text")
            import pyautogui
            pyautogui.write(text, interval=0.02)
            return True
        
        elif action == "click":
            # 点击屏幕坐标
            x = parameters.get("x", 0)
            y = parameters.get("y", 0)
            import pyautogui
            pyautogui.click(x, y)
            return True
        
        else:
            raise ValueError(f"未知的动作类型: {action}")
    
    def evaluate_condition(self, condition: str) -> bool:
        """评估条件"""
        # 简单的条件评估，实际应用中可能需要更复杂的逻辑
        return True
    
    def execute_workflow(self) -> Dict[str, Any]:
        """执行整个工作流"""
        print("开始执行工作流...")
        
        # 按依赖关系排序任务
        execution_order = self._get_execution_order()
        
        for task_id in execution_order:
            if not self.execute_task(task_id):
                print(f"工作流执行失败，任务 {task_id} 执行不成功")
                return {
                    "status": "failed",
                    "completed_tasks": [tid for tid, status in self.status.items() if status == TaskStatus.COMPLETED],
                    "failed_task": task_id
                }
        
        print("工作流执行完成")
        return {
            "status": "completed",
            "completed_tasks": [tid for tid, status in self.status.items() if status == TaskStatus.COMPLETED],
            "results": self.results
        }
    
    def _get_execution_order(self) -> List[str]:
        """获取任务执行顺序"""
        # 简单的拓扑排序
        order = []
        visited = set()
        
        def visit(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            
            task = self.tasks[task_id]
            for dep_id in task.dependencies:
                visit(dep_id)
            
            order.append(task_id)
        
        for task_id in self.tasks:
            visit(task_id)
        
        return order


class SmartDecisionEngine:
    """智能决策引擎"""
    
    def __init__(self):
        self.context = {}
        self.decision_rules = []
    
    def add_decision_rule(self, condition: Callable, action: Callable):
        """添加决策规则"""
        self.decision_rules.append({
            "condition": condition,
            "action": action
        })
    
    def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """基于上下文做出决策"""
        self.context = context
        
        for rule in self.decision_rules:
            if rule["condition"](context):
                return rule["action"](context)
        
        # 默认决策
        return self._default_decision(context)
    
    def _default_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """默认决策逻辑"""
        return {
            "action": "analyze_context",
            "reason": "未找到匹配的决策规则，执行默认分析",
            "next_steps": ["context_analysis"]
        }
    
    def process_user_request(self, user_request: str) -> List[Dict[str, Any]]:
        """处理用户请求"""
        # 分析用户请求
        request_analysis = self._analyze_request(user_request)
        
        # 根据分析结果生成任务
        tasks = self._generate_tasks_from_analysis(request_analysis)
        
        return tasks
    
    def _analyze_request(self, request: str) -> Dict[str, Any]:
        """分析用户请求"""
        analysis = {
            "request": request,
            "intent": self._identify_intent(request),
            "entities": self._extract_entities(request),
            "required_apps": self._identify_required_apps(request),
            "required_data": self._identify_required_data(request)
        }
        return analysis
    
    def _identify_intent(self, request: str) -> str:
        """识别用户意图"""
        request_lower = request.lower()
        
        if any(keyword in request_lower for keyword in ["打开", "启动", "运行", "open", "start", "launch"]):
            return "launch_app"
        elif any(keyword in request_lower for keyword in ["搜索", "查找", "search", "find", "browse"]):
            return "web_search"
        elif any(keyword in request_lower for keyword in ["写", "创建", "生成", "write", "create", "generate"]):
            return "content_creation"
        elif any(keyword in request_lower for keyword in ["提取", "分析", "处理", "extract", "analyze", "process"]):
            return "data_processing"
        else:
            return "unknown"
    
    def _extract_entities(self, request: str) -> Dict[str, str]:
        """提取实体"""
        import re
        
        entities = {}
        
        # 提取URL
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', request)
        if urls:
            entities["urls"] = urls
        
        # 提取应用名称
        apps = []
        if "word" in request.lower():
            apps.append("word")
        if "excel" in request.lower():
            apps.append("excel")
        if "powerpoint" in request.lower():
            apps.append("powerpoint")
        if "chrome" in request.lower() or "浏览器" in request.lower():
            apps.append("browser")
        if apps:
            entities["apps"] = apps
        
        # 提取文档类型
        if any(keyword in request.lower() for keyword in ["报告", "report", "文档", "document"]):
            entities["doc_type"] = "report"
        elif any(keyword in request.lower() for keyword in ["表格", "spreadsheet", "excel"]):
            entities["doc_type"] = "spreadsheet"
        
        return entities
    
    def _identify_required_apps(self, request: str) -> List[str]:
        """识别需要的应用"""
        apps = []
        request_lower = request.lower()
        
        if any(keyword in request_lower for keyword in ["word", "文档", "写作"]):
            apps.append("word")
        if any(keyword in request_lower for keyword in ["excel", "表格", "数据"]):
            apps.append("excel")
        if any(keyword in request_lower for keyword in ["powerpoint", "演示", "ppt"]):
            apps.append("powerpoint")
        if any(keyword in request_lower for keyword in ["浏览器", "搜索", "网页"]):
            apps.append("browser")
        
        return apps
    
    def _identify_required_data(self, request: str) -> List[str]:
        """识别需要的数据类型"""
        data_types = []
        request_lower = request.lower()
        
        if any(keyword in request_lower for keyword in ["统计", "分析", "数字"]):
            data_types.append("numerical")
        if any(keyword in request_lower for keyword in ["联系", "邮箱", "电话"]):
            data_types.append("contact")
        if any(keyword in request_lower for keyword in ["时间", "日期"]):
            data_types.append("temporal")
        
        return data_types
    
    def _generate_tasks_from_analysis(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据分析结果生成任务"""
        tasks = []
        intent = analysis["intent"]
        
        if intent == "launch_app":
            # 启动应用任务
            apps = analysis["entities"].get("apps", [])
            for i, app in enumerate(apps):
                task_id = f"launch_app_{i}"
                tasks.append({
                    "id": task_id,
                    "name": f"启动{app}应用",
                    "action": "start_app",
                    "parameters": {"app_name": app},
                    "dependencies": []
                })
        
        elif intent == "content_creation":
            # 内容创建任务
            # 1. 启动Word
            tasks.append({
                "id": "launch_word",
                "name": "启动Word",
                "action": "start_app",
                "parameters": {"app_name": "word"},
                "dependencies": []
            })
            
            # 2. 输入内容
            tasks.append({
                "id": "input_content",
                "name": "输入内容",
                "action": "type_text",
                "parameters": {"text": "这是自动生成的内容"},
                "dependencies": ["launch_word"]
            })
        
        elif intent == "data_processing":
            # 数据处理任务
            # 1. 提取数据
            tasks.append({
                "id": "extract_data",
                "name": "提取数据",
                "action": "extract_data",
                "parameters": {"text": analysis["request"]},
                "dependencies": []
            })
            
            # 2. 处理数据
            tasks.append({
                "id": "process_data",
                "name": "处理数据",
                "action": "process_data",
                "parameters": {"data": {}},  # 这里需要根据实际情况填充
                "dependencies": ["extract_data"]
            })
        
        return tasks


# 测试函数
def test_workflow_manager():
    """测试工作流管理器"""
    print("测试工作流管理器...")
    manager = WorkflowManager()
    
    # 创建一个简单的数据处理工作流
    tasks = [
        Task(
            id="extract",
            name="提取数据",
            action="extract_data",
            parameters={"text": "联系方式：contact@example.com 价格：$99.99"},
            dependencies=[]
        ),
        Task(
            id="process",
            name="处理数据",
            action="process_data",
            parameters={},
            dependencies=["extract"]
        ),
        Task(
            id="report",
            name="生成报告",
            action="generate_report",
            parameters={"title": "数据处理报告"},
            dependencies=["process"]
        )
    ]
    
    for task in tasks:
        manager.add_task(task)
    
    result = manager.execute_workflow()
    print(f"工作流执行结果: {result}")


def test_smart_decision_engine():
    """测试智能决策引擎"""
    print("测试智能决策引擎...")
    engine = SmartDecisionEngine()
    
    # 测试不同类型的请求
    requests = [
        "帮我打开Word并写一个报告",
        "分析这些数据：联系方式 contact@example.com 价格 $99.99",
        "启动浏览器并搜索人工智能"
    ]
    
    for req in requests:
        print(f"\n分析请求: {req}")
        tasks = engine.process_user_request(req)
        print(f"生成的任务: {tasks}")


if __name__ == "__main__":
    test_workflow_manager()
    test_smart_decision_engine()