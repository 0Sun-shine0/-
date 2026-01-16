"""
高级键盘控制器 - 支持智能输入、快捷键和文本处理
"""
import pyautogui
import time
import pyperclip
from typing import List, Dict, Any, Optional, Union
import string
import random
from enum import Enum
import json

class InputMode(Enum):
    NORMAL = "normal"
    FAST = "fast"
    ACCURATE = "accurate"
    STEALTH = "stealth"  # 模拟人类输入

class KeyboardController:
    """高级键盘控制器"""
    
    def __init__(self, wpm: int = 60, accuracy: float = 0.99):
        self.wpm = wpm  # 每分钟字数
        self.accuracy = accuracy  # 输入准确率
        self.input_mode = InputMode.NORMAL
        self.input_history = []
        self.common_typos = self._load_common_typos()
        
    def _load_common_typos(self) -> Dict[str, str]:
        """加载常见拼写错误"""
        return {
            "the": ["teh", "hte"],
            "and": ["adn", "nad"],
            "that": ["taht", "thta"],
            "with": ["wiht", "wth"],
            "this": ["htis", "tihs"],
            "have": ["haev", "hvae"],
            "from": ["form", "frome"]
        }
    
    def _contains_chinese(self, text: str) -> bool:
        """检查文本是否包含中文字符"""
        for char in text:
            if 0x4e00 <= ord(char) <= 0x9fff:  # 中文字符范围
                return True
        return False
    
    def type_text(self, text: str, mode: Optional[InputMode] = None):
        """输入文本"""
        if mode is None:
            mode = self.input_mode
        
        self.input_history.append({
            "text": text,
            "mode": mode.value,
            "timestamp": time.time()
        })
        
        # 确保在输入前有短暂延迟，让目标应用准备就绪
        time.sleep(1.0)

        # 如果文本较长或包含非 ASCII（如中文），优先使用剪贴板粘贴以提高可靠性
        use_clipboard = False
        try:
            if len(text) > 120 or self._contains_chinese(text):
                use_clipboard = True
        except Exception:
            use_clipboard = False

        if use_clipboard:
            self.paste_via_clipboard(text)
            time.sleep(0.2)
            return

        if mode == InputMode.FAST:
            self._type_fast(text)
        elif mode == InputMode.ACCURATE:
            self._type_accurate(text)
        elif mode == InputMode.STEALTH:
            self._type_stealth(text)
        else:
            self._type_normal(text)

        # 输入完成后等待，确保文本完全输入
        time.sleep(0.2)

    def paste_via_clipboard(self, text: str):
        """通过剪贴板粘贴大段文本，作为对逐字符输入的可靠回退方案"""
        try:
            old_clip = None
            try:
                old_clip = pyperclip.paste()
            except Exception:
                old_clip = None

            pyperclip.copy(text)
            # 在 Windows/大多数系统上使用 Ctrl+V 粘贴
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(0.05)

            # 尝试恢复之前剪贴板内容（最好但不是必需）
            if old_clip is not None:
                try:
                    pyperclip.copy(old_clip)
                except Exception:
                    pass
        except Exception:
            # 万一剪贴板不可用，回退到逐字符输入
            self._type_normal(text)

    def key_down(self, key: str):
        """按下按键（不释放）"""
        pyautogui.keyDown(key)

    def key_up(self, key: str):
        """释放按键"""
        pyautogui.keyUp(key)
    
    def _type_accurate(self, text: str):
        """准确输入（无错误）"""
        # 对于所有文本，直接使用pyautogui.write一次性输入，避免逐字符问题
        # 这样可以确保中文和英文都能正确输入
        pyautogui.write(text, interval=0.01)

    def _type_normal(self, text: str):
        """正常速度输入"""
        # 对于所有文本，直接使用pyautogui.write一次性输入，避免逐字符问题
        # 这样可以确保中文和英文都能正确输入
        pyautogui.write(text, interval=0.05)

    def _type_fast(self, text: str):
        """快速输入"""
        # 对于所有文本，直接使用pyautogui.write一次性输入，避免逐字符问题
        # 这样可以确保中文和英文都能正确输入
        pyautogui.write(text, interval=0.01)

    def _type_stealth(self, text: str):
        """隐身模式输入（模拟人类）"""
        # 对于所有文本，使用逐字输入来模拟人类输入模式
        for char in text:
            # 模拟人类打字速度变化
            base_speed = random.uniform(0.05, 0.15)
            delay = base_speed * random.uniform(0.8, 1.2)
            pyautogui.write(char, interval=delay)
            
            # 偶尔暂停（模拟思考）
            if random.random() < 0.05:
                time.sleep(random.uniform(0.1, 0.5))
    
    def _correct_typo(self, typo: str, correct: str):
        """纠正拼写错误"""
        # 删除错误输入
        for _ in range(len(typo)):
            pyautogui.press('backspace')
            time.sleep(0.05)
        
        # 重新输入正确单词
        pyautogui.write(correct, interval=0.05)
    
    def hotkey(self, *keys: str):
        """执行快捷键"""
        pyautogui.hotkey(*keys)
        self.input_history.append({
            "action": "hotkey",
            "keys": keys,
            "timestamp": time.time()
        })
    
    def press(self, key: str, presses: int = 1):
        """按下按键"""
        pyautogui.press(key, presses=presses)
        self.input_history.append({
            "action": "press",
            "key": key,
            "presses": presses,
            "timestamp": time.time()
        })
    
    def write_code(self, code: str, language: str = "python"):
        """编写代码（智能缩进和语法感知）"""
        lines = code.split('\n')
        
        for line in lines:
            # 处理缩进
            indent_level = len(line) - len(line.lstrip())
            
            if indent_level > 0:
                # 添加缩进
                for _ in range(indent_level // 4):
                    pyautogui.press('tab')
            
            # 输入代码行
            self.type_text(line.strip(), mode=InputMode.ACCURATE)
            
            # 添加换行
            pyautogui.press('enter')
            
            # 语言特定的智能行为
            if language == "python":
                self._handle_python_specific(line)
            elif language == "javascript":
                self._handle_javascript_specific(line)
    
    def _handle_python_specific(self, line: str):
        """处理Python特定的智能行为"""
        line = line.strip()
        
        # 自动添加冒号后的缩进
        if line.endswith(':'):
            time.sleep(0.1)
            pyautogui.press('enter')
            pyautogui.press('tab')
        
        # 自动闭合括号
        if line.count('(') > line.count(')'):
            self.press(')', presses=line.count('(') - line.count(')'))
        
        # 自动闭合引号
        if line.count('"') % 2 == 1:
            self.press('"')
        if line.count("'") % 2 == 1:
            self.press("'")
    
    def _handle_javascript_specific(self, line: str):
        """处理JavaScript特定的智能行为"""
        line = line.strip()
        
        # 自动闭合大括号
        if line.endswith('{'):
            time.sleep(0.1)
            pyautogui.press('enter')
            pyautogui.press('tab')
            pyautogui.press('enter')
            pyautogui.hotkey('shift', 'tab')
            pyautogui.write('}')
            pyautogui.hotkey('shift', 'tab')
        
        # 自动添加分号（可选）
        if line and not line.endswith(';') and not line.endswith('{'):
            if 'if' not in line and 'for' not in line and 'while' not in line:
                pyautogui.write(';')
    
    def fill_form(self, form_data: Dict[str, str]):
        """智能填充表单"""
        for field, value in form_data.items():
            # 切换到下一个字段
            pyautogui.press('tab')
            time.sleep(0.1)
            
            # 输入值
            self.type_text(value, mode=InputMode.ACCURATE)
            
            # 根据字段类型添加特定处理
            if "email" in field.lower():
                self._validate_email(value)
            elif "phone" in field.lower():
                self._format_phone(value)
    
    def _validate_email(self, email: str):
        """验证邮箱格式"""
        if "@" not in email or "." not in email.split("@")[1]:
            print(f"警告: 邮箱格式可能不正确: {email}")
    
    def _format_phone(self, phone: str):
        """格式化电话号码"""
        # 移除非数字字符
        digits = ''.join(filter(str.isdigit, phone))
        
        if len(digits) == 11:  # 中国手机号
            formatted = f"{digits[:3]} {digits[3:7]} {digits[7:]}"
            
            # 删除并重新输入格式化后的号码
            for _ in range(len(phone)):
                pyautogui.press('backspace')
            
            self.type_text(formatted, mode=InputMode.ACCURATE)
    
    def execute_macro(self, macro_name: str, params: Dict[str, Any] = None):
        """执行预定义的宏"""
        macros = {
            "save_and_close": self._macro_save_and_close,
            "format_document": self._macro_format_document,
            "search_and_replace": self._macro_search_and_replace,
            "duplicate_line": self._macro_duplicate_line,
            "comment_selection": self._macro_comment_selection,
        }
        
        if macro_name in macros:
            macro_func = macros[macro_name]
            macro_func(params or {})
        else:
            print(f"未知宏: {macro_name}")
    
    def _macro_save_and_close(self, params: Dict[str, Any]):
        """保存并关闭文档宏"""
        self.hotkey('ctrl', 's')  # 保存
        time.sleep(0.5)
        self.hotkey('ctrl', 'w')  # 关闭
    
    def _macro_format_document(self, params: Dict[str, Any]):
        """格式化文档宏"""
        self.hotkey('ctrl', 'a')  # 全选
        time.sleep(0.1)
        
        # 根据应用使用不同的格式化快捷键
        app = params.get('app', 'vscode')
        if app == 'vscode':
            self.hotkey('shift', 'alt', 'f')
        elif app == 'word':
            self.hotkey('ctrl', 'e')  # 居中对齐（示例）
        else:
            self.hotkey('ctrl', 'shift', 'f')
    
    def _macro_search_and_replace(self, params: Dict[str, Any]):
        """查找和替换宏"""
        search_text = params.get('search', '')
        replace_text = params.get('replace', '')
        
        self.hotkey('ctrl', 'h')  # 打开替换对话框
        time.sleep(0.5)
        
        self.type_text(search_text)
        pyautogui.press('tab')
        self.type_text(replace_text)
        pyautogui.press('enter')  # 替换
    
    def _macro_duplicate_line(self, params: Dict[str, Any]):
        """复制当前行宏"""
        self.hotkey('ctrl', 'c')  # 复制
        pyautogui.press('end')   # 到行尾
        pyautogui.press('enter') # 新行
        self.hotkey('ctrl', 'v')  # 粘贴
    
    def _macro_comment_selection(self, params: Dict[str, Any]):
        """注释选中行宏"""
        language = params.get('language', 'python')
        
        if language == 'python':
            self.hotkey('ctrl', '/')  # 在大多数编辑器中有效
        elif language == 'html':
            self.hotkey('ctrl', 'shift', '/')
    
    def get_input_speed(self) -> float:
        """计算输入速度（WPM）"""
        if len(self.input_history) < 2:
            return 0.0
        
        # 分析最近10次输入
        recent_history = self.input_history[-10:]
        
        total_chars = 0
        total_time = 0
        
        for i in range(1, len(recent_history)):
            if 'text' in recent_history[i] and 'text' in recent_history[i-1]:
                time_diff = recent_history[i]['timestamp'] - recent_history[i-1]['timestamp']
                chars = len(recent_history[i-1]['text'])
                
                total_chars += chars
                total_time += time_diff
        
        if total_time > 0:
            # 计算WPM（假设平均单词长度5个字符）
            wpm = (total_chars / 5) / (total_time / 60)
            return wpm
        
        return 0.0
    
    def get_accuracy_score(self) -> float:
        """计算输入准确率"""
        if not self.input_history:
            return 1.0
        
        # 这里可以添加更复杂的准确率计算逻辑
        return self.accuracy