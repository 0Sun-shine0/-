"""
DeepSeek思维链智能体 - 处理复杂任务的分解和规划
"""
import json
import re
from typing import Dict, List, Tuple, Any, Optional
import openai
from openai import OpenAI
import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging

from ai_operator.core.config import config

logger = logging.getLogger(__name__)

@dataclass
class ThoughtStep:
    """思维链步骤"""
    step_id: int
    description: str
    action_type: str  # think, analyze, plan, execute, verify
    content: str
    timestamp: datetime
    status: str = "pending"  # pending, in_progress, completed, failed

class DeepSeekAgent:
    """DeepSeek智能体，负责复杂任务分解和思维链生成"""
    
    def __init__(self, config=None, api_key: Optional[str] = None):
        # 如果传入了config参数，使用它；否则使用全局配置
        if config is not None:
            self.config = config
        else:
            from ai_operator.core.config import Config
            self.config = Config()
        
        self.api_key = api_key or self.config.DEEPSEEK_API_KEY
        if not self.api_key:
            # 如果没有API密钥，可以设置为模拟模式
            logger.warning("DeepSeek API key未设置，将使用模拟模式")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=config.DEEPSEEK_API_URL
        )
        
        self.thought_chain: List[ThoughtStep] = []
        
        # 系统提示词
        self.system_prompt = """你是AI桌面操作员的高级思维链处理器。你的任务是根据用户请求的类型，智能地决定响应方式：

对于需要桌面操作的任务（如打开应用、点击按钮、输入文本等）：
1. 将请求分解为可执行的桌面操作步骤
2. 为每个步骤生成详细的操作指令

对于需要内容生成的任务（如写小说、创作文章、回答问题等）：
1. 直接生成所需内容
2. 提供完整、高质量的内容输出

判断标准：
- 如果用户请求是创建、写作、生成内容，则直接生成内容
- 如果用户请求是执行操作、打开程序、点击等，则分解为操作步骤

响应格式必须是JSON，包含以下字段：
- "task_analysis": 对任务的分析和理解
- "content": 生成的具体内容（对于内容生成任务）
- "steps": 步骤列表，每个步骤包含（对于操作任务）:
  - "step_id": 步骤编号
  - "description": 步骤描述
- "thought_chain": 详细的思维链分析
- "estimated_time": 预计完成时间（分钟）
- "prerequisites": 前提条件
- "expected_outcomes": 预期结果
"""
    
    def create_thought_chain(self, user_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """创建思维链来处理复杂任务"""
        
        prompt = f"""
用户请求: {user_query}

当前上下文:
{json.dumps(context, ensure_ascii=False, indent=2)}

请按照以下步骤处理这个请求：
1. 分析请求的深层含义和目标
2. 识别完成任务所需的所有桌面操作
3. 考虑操作的顺序和依赖关系
4. 预测可能的障碍和解决方案
5. 生成详细的执行计划

请返回JSON格式的结果，包含task_analysis, steps, thought_chain, estimated_time, prerequisites, 和expected_outcomes字段。
"""
        
        try:
            response = self.client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=config.MAX_TOKENS,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            self._parse_thought_chain(result, user_query)
            return result
            
        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {e}")
            return self._fallback_response(user_query, context)
    
    def _parse_thought_chain(self, result: Dict[str, Any], user_query: str):
        """解析思维链结果"""
        self.thought_chain.clear()
        
        # 创建思维步骤
        steps = result.get("steps", [])
        thought_analysis = result.get("thought_chain", "")
        
        # 解析思维链中的思考步骤
        thought_steps = self._extract_thought_steps(thought_analysis)
        
        for i, step_text in enumerate(thought_steps):
            step = ThoughtStep(
                step_id=i,
                description=f"思维分析步骤 {i+1}",
                action_type="think",
                content=step_text,
                timestamp=datetime.now()
            )
            self.thought_chain.append(step)
        
        # 添加执行步骤
        for i, step_data in enumerate(steps):
            step = ThoughtStep(
                step_id=len(self.thought_chain) + i,
                description=step_data.get("description", ""),
                action_type="execute",
                content=f"执行: {step_data.get('description', '')}",
                timestamp=datetime.now(),
                status="pending"
            )
            self.thought_chain.append(step)
    
    def _extract_thought_steps(self, thought_analysis: str) -> List[str]:
        """从思维链文本中提取步骤"""
        # 按数字或项目符号分割
        steps = []
        lines = thought_analysis.split('\n')
        
        current_step = ""
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+[\.\)]', line) or re.match(r'^[•\-*]', line):
                if current_step:
                    steps.append(current_step)
                current_step = line
            elif line:
                current_step += " " + line
        
        if current_step:
            steps.append(current_step)
        
        return steps if steps else [thought_analysis]
    
    def _fallback_response(self, user_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """备用响应，当API调用失败时使用"""
        return {
            "task_analysis": f"分析用户请求: {user_query}",
            "steps": [
                {"step_id": 1, "description": "理解任务需求"},
                {"step_id": 2, "description": "规划执行步骤"},
                {"step_id": 3, "description": "执行具体操作"},
                {"step_id": 4, "description": "验证完成结果"}
            ],
            "thought_chain": "由于API连接问题，使用简化的思维链分析。",
            "estimated_time": 5,
            "prerequisites": ["网络连接正常", "相关软件已安装"],
            "expected_outcomes": ["任务完成", "结果符合预期"]
        }
    
    async def process_complex_task(self, task_description: str) -> Dict[str, Any]:
        """异步处理复杂任务"""
        context = {
            "current_app": self._detect_current_application(),
            "screen_state": self._get_screen_state(),
        }
        
        # 思维链分析
        thought_result = self.create_thought_chain(task_description, context)
        
        # 生成详细的操作序列
        operation_sequence = await self._generate_operation_sequence(thought_result)
        
        return {
            "thought_chain": thought_result,
            "operation_sequence": operation_sequence,
            "monitoring_metrics": self._generate_monitoring_metrics()
        }
    
    async def _generate_operation_sequence(self, thought_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成详细的操作序列"""
        operation_sequence = []
        
        for step in thought_result.get("steps", []):
            operations = await self._expand_step_to_operations(step)
            operation_sequence.extend(operations)
        
        return operation_sequence
    
    async def _expand_step_to_operations(self, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将步骤扩展为具体的操作"""
        description = step.get("description", "")
        
        prompt = f"""
将以下任务步骤分解为具体的桌面操作：
步骤: {description}

请生成JSON格式的操作序列，包含json_object格式，每个操作包含：
- action: 操作类型（如click, type, hotkey等）
- target: 目标描述
- params: 参数
"""
        
        try:
            response = self.client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "你是桌面操作分解专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("operations", [])
            
        except Exception as e:
            logger.error(f"操作分解失败: {e}")
            # 返回默认操作
            return [{
                "action": "keyboard_input",
                "target": "未知目标",
                "params": {"text": f"执行: {description}"}
            }]
    
    def _detect_current_application(self) -> str:
        """检测当前活跃应用程序"""
        try:
            import pygetwindow as gw
            active_window = gw.getActiveWindow()
            if active_window:
                return active_window.title
        except:
            pass
        return "unknown"
        
    def _get_screen_state(self) -> Dict[str, Any]:
        """获取当前屏幕状态"""
        try:
            import pygetwindow as gw
            import pyautogui
                
            # 获取屏幕分辨率
            screen_width, screen_height = pyautogui.size()
                
            # 获取当前活动窗口
            active_window = "unknown"
            try:
                active_window_obj = gw.getActiveWindow()
                if active_window_obj:
                    active_window = active_window_obj.title
            except:
                pass
                
            # 获取鼠标位置
            mouse_x, mouse_y = pyautogui.position()
                
            return {
                "resolution": f"{screen_width}x{screen_height}",
                "active_window": active_window,
                "mouse_position": (mouse_x, mouse_y)
            }
        except:
            # 如果无法获取详细信息，返回默认值
            return {
                "resolution": "1920x1080",
                "active_window": "unknown",
                "mouse_position": (0, 0)
            }
    
    def _generate_monitoring_metrics(self) -> Dict[str, Any]:
        """生成监控指标"""
        return {
            "thought_depth": len(self.thought_chain),
            "complexity_score": self._calculate_complexity(),
            "confidence_level": 0.85,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_complexity(self) -> float:
        """计算任务复杂度"""
        if not self.thought_chain:
            return 0.0
        
        total_steps = len(self.thought_chain)
        thought_steps = sum(1 for step in self.thought_chain if step.action_type == "think")
        
        return min(1.0, (thought_steps * 0.3 + total_steps * 0.7) / 10.0)
    
    def process_task(self, task_description: str) -> Dict[str, Any]:
        """处理任务的主要方法"""
        context = {
            "current_app": self._detect_current_application(),
            "screen_state": self._get_screen_state(),
        }
        
        # 使用思维链处理任务
        result = self.create_thought_chain(task_description, context)
        return result