"""
高级内容生成器 - 提升内容质量和多样性
"""
import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import random
import time


@dataclass
class ContentTemplate:
    """内容模板定义"""
    name: str
    type: str  # 'document', 'report', 'email', 'article', 'story'
    structure: List[str]  # 内容结构，如 ['title', 'introduction', 'body', 'conclusion']
    keywords: List[str]
    examples: List[str]


class AdvancedContentGenerator:
    """高级内容生成器"""
    
    def __init__(self):
        self.templates = self._load_templates()
        self.knowledge_base = self._load_knowledge_base()
        
    def _load_templates(self) -> Dict[str, ContentTemplate]:
        """加载内容模板"""
        templates = {
            "report": ContentTemplate(
                name="标准报告",
                type="report",
                structure=["title", "executive_summary", "introduction", "analysis", "findings", "conclusion", "recommendations"],
                keywords=["分析", "数据", "结论", "建议", "趋势", "预测"],
                examples=[
                    "根据最新市场数据，我们对Q4销售业绩进行了全面分析...",
                    "本报告旨在评估项目执行效果并提出改进建议..."
                ]
            ),
            "email": ContentTemplate(
                name="商务邮件",
                type="email",
                structure=["greeting", "subject", "body", "action_items", "closing", "signature"],
                keywords=["尊敬", "感谢", "附件", "会议", "确认", "跟进"],
                examples=[
                    "尊敬的客户，感谢您选择我们的服务...",
                    "关于昨天讨论的项目，我想跟进一下进度..."
                ]
            ),
            "story": ContentTemplate(
                name="故事创作",
                type="story",
                structure=["title", "setting", "characters", "plot", "conflict", "resolution", "conclusion"],
                keywords=["从前", "然后", "但是", "最终", "结局", "道理"],
                examples=[
                    "从前有一个小村庄，住着一位善良的少年...",
                    "在遥远的国度里，有一个神秘的传说..."
                ]
            )
        }
        return templates
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """加载知识库"""
        return {
            "business_terms": [
                "ROI", "KPI", "B2B", "B2C", "SaaS", "PaaS", "IaaS",
                "市场渗透率", "客户获取成本", "用户生命周期价值"
            ],
            "common_phrases": [
                "基于以上分析", "综上所述", "值得注意的是",
                "为了进一步验证", "根据数据显示"
            ],
            "transition_words": [
                "首先", "其次", "最后", "另外", "同时", "然而",
                "因此", "所以", "由于", "尽管", "虽然"
            ]
        }
    
    def generate_content(self, 
                        content_type: str, 
                        topic: str, 
                        length: str = "medium", 
                        style: str = "professional") -> str:
        """生成内容"""
        if content_type not in self.templates:
            return self._generate_generic_content(topic, length, style)
        
        template = self.templates[content_type]
        return self._generate_from_template(template, topic, length, style)
    
    def _generate_generic_content(self, topic: str, length: str, style: str) -> str:
        """生成通用内容"""
        length_map = {
            "short": 100,
            "medium": 300,
            "long": 600
        }
        target_length = length_map.get(length, 300)
        
        content_parts = []
        
        # 添加标题
        content_parts.append(f"关于{topic}的分析")
        content_parts.append("")
        
        # 添加引言
        content_parts.append("引言:")
        content_parts.append(f"关于{topic}，这是一个非常重要的话题。")
        content_parts.append("")
        
        # 添加主体内容
        content_parts.append("主体:")
        content_parts.append(f"在讨论{topic}时，我们需要考虑多个方面。")
        
        # 根据长度添加更多内容
        if length in ["medium", "long"]:
            content_parts.append(f"首先，{topic}具有重要意义。")
            content_parts.append(f"其次，我们需要关注{topic}的相关因素。")
            
            if length == "long":
                content_parts.append(f"此外，{topic}还涉及多个层面。")
                content_parts.append(f"最后，我们可以得出关于{topic}的结论。")
        
        content_parts.append("")
        
        # 添加结论
        content_parts.append("结论:")
        content_parts.append(f"综上所述，{topic}是一个值得深入探讨的话题。")
        
        content = "\n".join(content_parts)
        
        # 如果生成的内容长度不够，扩展内容
        while len(content) < target_length:
            content += f"\n\n关于{topic}的进一步分析显示，这个问题比我们想象的更加复杂。"
        
        return content
    
    def _generate_from_template(self, 
                               template: ContentTemplate, 
                               topic: str, 
                               length: str, 
                               style: str) -> str:
        """根据模板生成内容"""
        content_parts = []
        
        for section in template.structure:
            section_content = self._generate_section(section, topic, length, style)
            if section_content:
                content_parts.append(section_content)
                content_parts.append("")  # 添加空行分隔
        
        return "\n".join(content_parts).strip()
    
    def _generate_section(self, section: str, topic: str, length: str, style: str) -> str:
        """生成内容段落"""
        section_map = {
            "title": f"《{topic}》",
            "greeting": "尊敬的收件人，",
            "subject": f"主题：关于{topic}的报告",
            "introduction": f"关于{topic}，这是一个非常重要的话题。",
            "executive_summary": f"本摘要旨在概述关于{topic}的关键发现。",
            "analysis": f"对{topic}的深入分析显示了以下趋势...",
            "findings": f"关于{topic}的主要发现包括...",
            "conclusion": f"综上所述，关于{topic}的结论如下...",
            "recommendations": f"基于对{topic}的分析，我们建议...",
            "body": f"关于{topic}的详细信息如下...",
            "action_items": f"关于{topic}，需要采取以下行动...",
            "closing": f"如果您对{topic}有任何疑问，请随时联系我们。",
            "signature": f"此致\n敬礼",
            "setting": f"故事发生在{random.choice(['一个遥远的村庄', '繁华的都市', '神秘的森林'])}...",
            "characters": f"主要角色包括{random.choice(['勇敢的少年', '智慧的长者', '神秘的旅者'])}...",
            "plot": f"故事的主线围绕{topic}展开...",
            "conflict": f"故事的冲突点在于{topic}的挑战...",
            "resolution": f"最终，{topic}的问题得到了解决..."
        }
        
        base_content = section_map.get(section, f"{section}: 关于{topic}的内容")
        
        # 根据长度调整内容
        if length == "long":
            base_content += f" 这是一个关于{topic}的详细分析。"
            base_content += f" 在深入研究{topic}的过程中，我们发现了许多有趣的现象。"
        
        return base_content
    
    def enhance_content(self, content: str) -> str:
        """增强现有内容"""
        # 添加专业术语
        enhanced = self._add_business_terms(content)
        
        # 添加过渡词
        enhanced = self._add_transition_words(enhanced)
        
        # 改进句子结构
        enhanced = self._improve_sentence_structure(enhanced)
        
        return enhanced
    
    def _add_business_terms(self, content: str) -> str:
        """添加专业术语"""
        terms = self.knowledge_base["business_terms"]
        # 随机选择一些术语添加到内容中
        for _ in range(min(3, len(terms))):
            term = random.choice(terms)
            if random.random() > 0.7:  # 30%概率添加术语
                content += f" ({term})"
        
        return content
    
    def _add_transition_words(self, content: str) -> str:
        """添加过渡词"""
        words = self.knowledge_base["transition_words"]
        sentences = content.split('.')
        
        enhanced_sentences = []
        for i, sentence in enumerate(sentences):
            if i > 0 and random.random() > 0.8:  # 20%概率添加过渡词
                transition = random.choice(words)
                enhanced_sentences.append(f"{transition}，{sentence.strip()}")
            else:
                enhanced_sentences.append(sentence.strip())
        
        return '. '.join(enhanced_sentences)
    
    def _improve_sentence_structure(self, content: str) -> str:
        """改进句子结构"""
        # 简单的句子结构改进
        content = re.sub(r'([。！？])\s*([^\n])', r'\1\n\2', content)  # 在句号后添加换行
        return content


class SmartErrorHandler:
    """智能错误处理与恢复系统"""
    
    def __init__(self):
        self.error_history = []
        self.recovery_strategies = {
            "app_not_responding": self._handle_app_not_responding,
            "element_not_found": self._handle_element_not_found,
            "access_denied": self._handle_access_denied,
            "timeout": self._handle_timeout,
            "connection_error": self._handle_connection_error
        }
    
    def handle_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理错误"""
        error_info = {
            "type": error_type,
            "message": error_message,
            "timestamp": time.time(),
            "context": context or {}
        }
        
        self.error_history.append(error_info)
        
        # 尝试恢复
        recovery_result = self._attempt_recovery(error_type, error_message, context)
        
        return {
            "handled": True,
            "recovery_attempted": recovery_result is not None,
            "recovery_success": recovery_result or False,
            "suggested_action": self._get_suggested_action(error_type)
        }
    
    def _attempt_recovery(self, error_type: str, error_message: str, context: Dict[str, Any]) -> Optional[bool]:
        """尝试恢复"""
        if error_type in self.recovery_strategies:
            try:
                return self.recovery_strategies[error_type](context)
            except Exception as e:
                print(f"恢复尝试失败: {e}")
                return False
        return None
    
    def _handle_app_not_responding(self, context: Dict[str, Any]) -> bool:
        """处理应用无响应"""
        print("检测到应用无响应，尝试重启应用...")
        # 实际实现中会尝试重启应用
        return True
    
    def _handle_element_not_found(self, context: Dict[str, Any]) -> bool:
        """处理元素未找到"""
        print("页面元素未找到，尝试等待或查找替代元素...")
        # 实际实现中会尝试其他方法查找元素
        return True
    
    def _handle_access_denied(self, context: Dict[str, Any]) -> bool:
        """处理访问被拒绝"""
        print("访问被拒绝，检查权限设置...")
        # 实际实现中会检查权限
        return False  # 权限问题通常无法自动恢复
    
    def _handle_timeout(self, context: Dict[str, Any]) -> bool:
        """处理超时"""
        print("操作超时，增加等待时间...")
        # 实际实现中会增加超时时间
        return True
    
    def _handle_connection_error(self, context: Dict[str, Any]) -> bool:
        """处理连接错误"""
        print("连接错误，检查网络连接...")
        # 实际实现中会尝试重新连接
        return True
    
    def _get_suggested_action(self, error_type: str) -> str:
        """获取建议操作"""
        actions = {
            "app_not_responding": "重启应用程序",
            "element_not_found": "检查页面元素选择器",
            "access_denied": "检查权限设置",
            "timeout": "增加操作等待时间",
            "connection_error": "检查网络连接"
        }
        return actions.get(error_type, "检查系统状态")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计"""
        if not self.error_history:
            return {"total_errors": 0}
        
        error_types = {}
        for error in self.error_history:
            error_type = error["type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "recent_errors": self.error_history[-5:]  # 最近5个错误
        }


class PerformanceMonitor:
    """性能监控与分析"""
    
    def __init__(self):
        self.metrics = {
            "execution_times": [],
            "success_rates": [],
            "resource_usage": [],
            "operation_counts": {}
        }
        self.start_time = time.time()
    
    def record_operation(self, operation: str, execution_time: float, success: bool):
        """记录操作"""
        self.metrics["execution_times"].append({
            "operation": operation,
            "time": execution_time,
            "success": success,
            "timestamp": time.time()
        })
        
        # 更新操作计数
        self.metrics["operation_counts"][operation] = self.metrics["operation_counts"].get(operation, 0) + 1
        
        # 更新成功率
        if success:
            self.metrics["success_rates"].append(1)
        else:
            self.metrics["success_rates"].append(0)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        total_ops = len(self.metrics["execution_times"])
        successful_ops = sum(self.metrics["success_rates"]) if self.metrics["success_rates"] else 0
        success_rate = successful_ops / total_ops if total_ops > 0 else 0
        
        avg_execution_time = 0
        if self.metrics["execution_times"]:
            avg_execution_time = sum(op["time"] for op in self.metrics["execution_times"]) / len(self.metrics["execution_times"])
        
        return {
            "uptime": time.time() - self.start_time,
            "total_operations": total_ops,
            "successful_operations": successful_ops,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "operation_breakdown": self.metrics["operation_counts"],
            "recent_operations": self.metrics["execution_times"][-10:]  # 最近10次操作
        }
    
    def get_operation_efficiency(self, operation: str) -> Dict[str, float]:
        """获取特定操作的效率"""
        ops = [op for op in self.metrics["execution_times"] if op["operation"] == operation]
        if not ops:
            return {"count": 0, "avg_time": 0, "success_rate": 0}
        
        total_ops = len(ops)
        successful_ops = sum(1 for op in ops if op["success"])
        avg_time = sum(op["time"] for op in ops) / total_ops
        
        return {
            "count": total_ops,
            "avg_time": avg_time,
            "success_rate": successful_ops / total_ops if total_ops > 0 else 0
        }


# 测试函数
def test_advanced_content_generator():
    """测试高级内容生成器"""
    print("=== 测试高级内容生成器 ===")
    generator = AdvancedContentGenerator()
    
    # 测试生成不同类型的内容
    content_types = ["report", "email", "story"]
    for content_type in content_types:
        content = generator.generate_content(content_type, "人工智能", "medium")
        print(f"{content_type}内容生成: {len(content)} 字符")
        print(f"内容预览: {content[:100]}...")
        print()
    
    # 测试内容增强
    basic_content = "这是一个关于项目管理的基本描述。"
    enhanced = generator.enhance_content(basic_content)
    print(f"内容增强: '{basic_content}' -> '{enhanced}'")
    print()


def test_smart_error_handler():
    """测试智能错误处理器"""
    print("=== 测试智能错误处理器 ===")
    handler = SmartErrorHandler()
    
    # 模拟不同类型的错误
    errors = [
        ("app_not_responding", "应用程序无响应"),
        ("element_not_found", "找不到指定元素"),
        ("timeout", "操作超时")
    ]
    
    for error_type, error_msg in errors:
        result = handler.handle_error(error_type, error_msg, {"operation": "click", "element": "button"})
        print(f"{error_type}: 恢复{'成功' if result['recovery_success'] else '失败'}")
        print(f"建议操作: {result['suggested_action']}")
        print()
    
    # 获取错误统计
    stats = handler.get_error_statistics()
    print(f"错误统计: {stats}")
    print()


def test_performance_monitor():
    """测试性能监控器"""
    print("=== 测试性能监控器 ===")
    monitor = PerformanceMonitor()
    
    # 模拟记录一些操作
    operations = ["click", "type", "click", "navigate", "type"]
    import random
    for op in operations:
        execution_time = random.uniform(0.1, 2.0)
        success = random.random() > 0.1  # 90%成功率
        monitor.record_operation(op, execution_time, success)
    
    # 获取性能报告
    report = monitor.get_performance_report()
    print(f"性能报告: {report}")
    
    # 获取特定操作效率
    for op in set(operations):
        efficiency = monitor.get_operation_efficiency(op)
        print(f"{op}效率: {efficiency}")
    print()


if __name__ == "__main__":
    test_advanced_content_generator()
    test_smart_error_handler()
    test_performance_monitor()
    print("高级功能测试完成！")