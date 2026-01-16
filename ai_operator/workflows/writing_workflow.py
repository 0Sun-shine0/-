"""
写作工作流模块
实现复杂的文档编写、编辑和格式化工作流
"""
import time
import json
from typing import Dict, List, Any, Optional
from enum import Enum

from ..core.config import Config
from ..agents.deepseek_agent import DeepSeekAgent
from ..automation.mouse_controller import AdvancedMouseController
from ..automation.keyboard_controller import KeyboardController


class WritingStage(Enum):
    """写作阶段枚举"""
    RESEARCH = "research"
    OUTLINE = "outline"
    DRAFT = "draft"
    EDIT = "edit"
    FORMAT = "format"
    REVIEW = "review"


class WritingWorkflow:
    """写作工作流类"""
    
    def __init__(self, config: Config):
        self.config = config
        self.deepseek_agent = DeepSeekAgent(config)
        self.mouse_controller = AdvancedMouseController()
        self.keyboard_controller = KeyboardController()
        
        self.current_stage = WritingStage.RESEARCH
        self.workflow_state = {
            'topic': '',
            'research_data': {},
            'outline': [],
            'draft_content': '',
            'edits_made': [],
            'final_document': ''
        }
        
    def execute_writing_task(self, topic: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行完整的写作任务
        
        Args:
            topic: 写作主题
            requirements: 写作要求字典
            
        Returns:
            包含写作结果的字典
        """
        self.workflow_state['topic'] = topic
        
        try:
            # 1. 研究阶段
            self.current_stage = WritingStage.RESEARCH
            self._research_phase(topic, requirements)
            
            # 2. 大纲阶段
            self.current_stage = WritingStage.OUTLINE
            self._outline_phase(topic, requirements)
            
            # 3. 草稿阶段
            self.current_stage = WritingStage.DRAFT
            self._draft_phase(topic, requirements)
            
            # 4. 编辑阶段
            self.current_stage = WritingStage.EDIT
            self._edit_phase(requirements)
            
            # 5. 格式化阶段
            self.current_stage = WritingStage.FORMAT
            self._format_phase(requirements)
            
            # 6. 审查阶段
            self.current_stage = WritingStage.REVIEW
            self._review_phase()
            
            return {
                'success': True,
                'final_document': self.workflow_state['final_document'],
                'workflow_state': self.workflow_state,
                'summary': self._generate_summary()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'workflow_state': self.workflow_state
            }
    
    def _research_phase(self, topic: str, requirements: Dict[str, Any]):
        """研究阶段：收集相关信息"""
        print(f"开始研究阶段 - 主题: {topic}")
        
        # 使用DeepSeek进行研究
        research_prompt = f"""
        请为以下主题进行深入研究：{topic}
        研究要求：{json.dumps(requirements, ensure_ascii=False)}
        
        请提供：
        1. 关键概念和定义
        2. 相关数据和统计信息
        3. 重要观点和论据
        4. 参考资料和来源
        
        以JSON格式返回研究结果。
        """
        
        research_result = self.deepseek_agent.process_task(research_prompt)
        self.workflow_state['research_data'] = research_result
        
        print("研究阶段完成")
    
    def _outline_phase(self, topic: str, requirements: Dict[str, Any]):
        """大纲阶段：创建文档结构"""
        print(f"开始大纲阶段 - 主题: {topic}")
        
        outline_prompt = f"""
        基于以下研究数据，为'{topic}'创建一个详细的文档大纲：
        {json.dumps(self.workflow_state['research_data'], ensure_ascii=False)}
        
        大纲要求：{json.dumps(requirements, ensure_ascii=False)}
        
        请返回一个包含章节、子章节和要点的详细大纲，使用JSON格式。
        """
        
        outline_result = self.deepseek_agent.process_task(outline_prompt)
        
        if isinstance(outline_result, dict) and 'outline' in outline_result:
            self.workflow_state['outline'] = outline_result['outline']
        elif isinstance(outline_result, list):
            self.workflow_state['outline'] = outline_result
        else:
            # 尝试解析为JSON
            try:
                parsed_outline = json.loads(str(outline_result))
                self.workflow_state['outline'] = parsed_outline.get('outline', [])
            except:
                self.workflow_state['outline'] = []
        
        print("大纲阶段完成")
    
    def _draft_phase(self, topic: str, requirements: Dict[str, Any]):
        """草稿阶段：撰写初稿"""
        print(f"开始草稿阶段 - 主题: {topic}")
        
        # 在文本编辑器中打开新文档
        self._open_text_editor()
        
        # 逐段撰写内容
        draft_content = []
        
        for section in self.workflow_state['outline']:
            section_content = self._generate_section_content(topic, section, requirements)
            draft_content.append(section_content)
            
            # 输入内容到文档
            self.keyboard_controller.type_text(section_content + "\n\n")
            time.sleep(0.5)  # 短暂暂停
        
        self.workflow_state['draft_content'] = "\n\n".join(draft_content)
        print("草稿阶段完成")
    
    def _edit_phase(self, requirements: Dict[str, Any]):
        """编辑阶段：修改和完善内容"""
        print("开始编辑阶段")
        
        # 全选文档内容
        self.keyboard_controller.press_shortcut(['ctrl', 'a'])
        time.sleep(0.5)
        
        # 复制内容以便AI分析
        self.keyboard_controller.press_shortcut(['ctrl', 'c'])
        time.sleep(0.5)
        
        # 使用DeepSeek进行内容编辑
        edit_prompt = f"""
        请编辑以下文档内容：
        {self.workflow_state['draft_content']}
        
        编辑要求：{json.dumps(requirements, ensure_ascii=False)}
        
        请提供改进后的内容。
        """
        
        edited_content = self.deepseek_agent.process_task(edit_prompt)
        
        # 清除当前内容并粘贴编辑后的内容
        self.keyboard_controller.press_key('delete')
        time.sleep(0.2)
        
        if isinstance(edited_content, dict):
            edited_text = edited_content.get('content', str(edited_content))
        else:
            edited_text = str(edited_content)
            
        self.keyboard_controller.type_text(edited_text)
        
        self.workflow_state['edits_made'].append({
            'stage': 'edit',
            'original': self.workflow_state['draft_content'],
            'edited': edited_text,
            'timestamp': time.time()
        })
        
        self.workflow_state['draft_content'] = edited_text
        print("编辑阶段完成")
    
    def _format_phase(self, requirements: Dict[str, Any]):
        """格式化阶段：应用样式和格式"""
        print("开始格式化阶段")
        
        # 全选文档
        self.keyboard_controller.press_shortcut(['ctrl', 'a'])
        time.sleep(0.5)
        
        # 应用基本格式（根据要求）
        formatting_instructions = requirements.get('formatting', {})
        
        # 应用字体和大小
        if 'font' in formatting_instructions:
            font_name = formatting_instructions['font']
            # 通过快捷键或菜单应用字体（简化处理）
            print(f"应用字体: {font_name}")
        
        if 'size' in formatting_instructions:
            font_size = formatting_instructions['size']
            print(f"应用字号: {font_size}")
        
        # 应用标题格式
        self._apply_heading_formats()
        
        # 应用段落格式
        self._apply_paragraph_formats()
        
        print("格式化阶段完成")
    
    def _review_phase(self):
        """审查阶段：最终检查和确认"""
        print("开始审查阶段")
        
        # 获取最终文档内容
        self.keyboard_controller.press_shortcut(['ctrl', 'a'])
        time.sleep(0.2)
        self.keyboard_controller.press_shortcut(['ctrl', 'c'])
        time.sleep(0.2)
        
        # 使用AI进行最终审查
        review_prompt = f"""
        请审查以下文档：
        {self.workflow_state['draft_content']}
        
        请检查：
        1. 内容完整性
        2. 逻辑连贯性
        3. 语法和拼写
        4. 格式一致性
        
        提供审查报告和改进建议。
        """
        
        review_result = self.deepseek_agent.process_task(review_prompt)
        
        # 保存最终文档
        self.workflow_state['final_document'] = self.workflow_state['draft_content']
        
        self.workflow_state['review'] = review_result
        print("审查阶段完成")
    
    def _generate_section_content(self, topic: str, section: Any, requirements: Dict[str, Any]) -> str:
        """生成单个章节内容"""
        section_prompt = f"""
        为'{topic}'的以下章节撰写内容：
        {json.dumps(section, ensure_ascii=False)}
        
        写作要求：{json.dumps(requirements, ensure_ascii=False)}
        
        请提供详细、准确且连贯的内容。
        """
        
        content = self.deepseek_agent.process_task(section_prompt)
        
        if isinstance(content, dict):
            return content.get('content', str(content))
        else:
            return str(content)
    
    def _open_text_editor(self):
        """打开文本编辑器"""
        # 模拟打开文本编辑器（如Word、记事本等）
        # 这里简化为打开记事本
        self.keyboard_controller.press_shortcut(['win', 'r'])
        time.sleep(0.5)
        self.keyboard_controller.type_text('notepad')
        self.keyboard_controller.press_key('enter')
        time.sleep(2)  # 等待程序打开
    
    def _apply_heading_formats(self):
        """应用标题格式"""
        # 模拟应用标题格式（简化处理）
        # 在实际实现中，这将通过应用程序的菜单或快捷键来完成
        pass
    
    def _apply_paragraph_formats(self):
        """应用段落格式"""
        # 模拟应用段落格式（简化处理）
        pass
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成工作流摘要"""
        return {
            'topic': self.workflow_state['topic'],
            'total_sections': len(self.workflow_state['outline']),
            'total_edits': len(self.workflow_state['edits_made']),
            'time_stages': {
                'research': 'N/A',
                'outline': 'N/A', 
                'draft': 'N/A',
                'edit': 'N/A',
                'format': 'N/A',
                'review': 'N/A'
            }
        }


# 使用示例
def example_usage():
    """使用示例"""
    config = Config()
    workflow = WritingWorkflow(config)
    
    topic = "人工智能在医疗领域的应用"
    requirements = {
        "length": "1000字",
        "tone": "专业",
        "audience": "医疗专业人士",
        "formatting": {
            "font": "宋体",
            "size": 12,
            "spacing": "1.5倍行距"
        },
        "sections": ["引言", "主要应用", "挑战", "未来展望", "结论"]
    }
    
    result = workflow.execute_writing_task(topic, requirements)
    
    if result['success']:
        print("写作任务完成！")
        print(f"最终文档长度: {len(result['final_document'])} 字符")
    else:
        print(f"写作任务失败: {result['error']}")


if __name__ == "__main__":
    example_usage()