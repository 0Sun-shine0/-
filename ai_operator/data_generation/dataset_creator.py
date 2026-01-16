"""
专业级数据集生成器
"""
import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import time
from datetime import datetime
import cv2

@dataclass
class UIElement:
    """UI元素定义"""
    element_type: str  # button, input, icon, menu, etc.
    bounds: Tuple[int, int, int, int]  # x1, y1, x2, y2
    text: str = ""
    action: str = ""
    style: Dict[str, Any] = None

class ProfessionalDatasetCreator:
    """专业级数据集生成器"""
    
    def __init__(self, output_dir: str = "datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 屏幕分辨率配置
        self.resolutions = [
            (1920, 1080),  # 1080p
            (1366, 768),   # 常见笔记本
            (2560, 1440),  # 2K
            (3840, 2160),  # 4K
        ]
        
        # UI主题配置
        self.themes = {
            "windows_light": {
                "bg_color": (240, 240, 240),
                "button_color": (0, 120, 215),
                "input_color": (255, 255, 255),
                "text_color": (0, 0, 0),
                "border_color": (200, 200, 200)
            },
            "windows_dark": {
                "bg_color": (32, 32, 32),
                "button_color": (0, 90, 158),
                "input_color": (56, 56, 56),
                "text_color": (255, 255, 255),
                "border_color": (100, 100, 100)
            },
            "macos_light": {
                "bg_color": (251, 251, 251),
                "button_color": (0, 122, 255),
                "input_color": (242, 242, 247),
                "text_color": (0, 0, 0),
                "border_color": (209, 209, 214)
            },
            "macos_dark": {
                "bg_color": (28, 28, 30),
                "button_color": (10, 132, 255),
                "input_color": (44, 44, 46),
                "text_color": (255, 255, 255),
                "border_color": (72, 72, 74)
            }
        }
        
        # 应用程序模板
        self.app_templates = self._load_app_templates()
        
        # 操作定义
        self.actions = self._define_actions()
        
    def _load_app_templates(self) -> Dict[str, Any]:
        """加载应用程序模板"""
        return {
            "web_browser": {
                "name": ["Chrome", "Firefox", "Edge", "Safari"],
                "elements": ["address_bar", "tabs", "bookmarks", "settings"],
                "layouts": ["normal", "fullscreen", "dev_tools"]
            },
            "text_editor": {
                "name": ["VSCode", "Sublime", "Notepad++", "Word"],
                "elements": ["menu_bar", "toolbar", "editor", "sidebar"],
                "layouts": ["normal", "zen", "split_view"]
            },
            "file_explorer": {
                "name": ["Finder", "Explorer", "Nautilus"],
                "elements": ["sidebar", "file_list", "address_bar", "preview"],
                "layouts": ["details", "icons", "list", "tiles"]
            }
        }
    
    def _define_actions(self) -> Dict[str, List[str]]:
        """定义所有可能的操作"""
        return {
            "mouse": [
                "left_click", "right_click", "double_click",
                "middle_click", "drag_start", "drag_end",
                "hover", "wheel_up", "wheel_down"
            ],
            "keyboard": [
                "type_text", "press_enter", "press_tab",
                "press_escape", "press_backspace", "press_delete",
                "press_space", "press_arrow_up", "press_arrow_down",
                "press_arrow_left", "press_arrow_right"
            ],
            "hotkeys": [
                "ctrl_c", "ctrl_v", "ctrl_x", "ctrl_z", "ctrl_y",
                "ctrl_s", "ctrl_o", "ctrl_n", "ctrl_f", "ctrl_h",
                "alt_tab", "alt_f4", "win_d", "win_e"
            ],
            "system": [
                "window_minimize", "window_maximize", "window_close",
                "window_resize", "window_move", "switch_app",
                "show_desktop", "open_start_menu"
            ]
        }
    
    def create_basic_dataset(self, num_samples: int = 1000):
        """创建基础数据集"""
        print(f"开始创建基础数据集: {num_samples} 样本")
        
        dataset_info = {
            "name": "basic_desktop_operations_v1",
            "created_at": datetime.now().isoformat(),
            "num_samples": num_samples,
            "resolution": "1920x1080",
            "actions_covered": list(self.actions.keys()),
            "samples": []
        }
        
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"进度: {i}/{num_samples}")
            
            # 随机选择主题和分辨率
            theme_name = random.choice(list(self.themes.keys()))
            resolution = random.choice(self.resolutions)
            
            # 创建屏幕截图
            screenshot, elements = self.create_desktop_screenshot(
                theme_name, resolution
            )
            
            # 生成随机操作
            operation = self.generate_random_operation(elements)
            
            # 保存样本
            sample_id = f"basic_{i:06d}"
            self.save_sample(sample_id, screenshot, operation, elements)
            
            dataset_info["samples"].append({
                "id": sample_id,
                "theme": theme_name,
                "resolution": f"{resolution[0]}x{resolution[1]}",
                "operation": operation,
                "elements_count": len(elements)
            })
        
        # 保存数据集信息
        self.save_dataset_info(dataset_info, "basic")
        print(f"基础数据集创建完成: {num_samples} 样本")
    
    def create_app_specific_dataset(self, app_type: str, num_samples: int = 500):
        """创建应用程序特定数据集"""
        print(f"创建 {app_type} 数据集: {num_samples} 样本")
        
        dataset_info = {
            "name": f"{app_type}_operations_v1",
            "created_at": datetime.now().isoformat(),
            "num_samples": num_samples,
            "app_type": app_type,
            "samples": []
        }
        
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"进度: {i}/{num_samples}")
            
            # 创建应用程序界面
            screenshot, elements = self.create_app_interface(app_type)
            
            # 生成应用特定操作
            operation = self.generate_app_specific_operation(app_type, elements)
            
            # 保存样本
            sample_id = f"{app_type}_{i:06d}"
            self.save_sample(sample_id, screenshot, operation, elements)
            
            dataset_info["samples"].append({
                "id": sample_id,
                "app_name": random.choice(self.app_templates[app_type]["name"]),
                "operation": operation,
                "elements_count": len(elements)
            })
        
        self.save_dataset_info(dataset_info, app_type)
        print(f"{app_type} 数据集创建完成")
    
    def create_desktop_screenshot(self, theme_name: str, resolution: Tuple[int, int]):
        """创建桌面截图"""
        width, height = resolution
        theme = self.themes[theme_name]
        
        # 创建基础图像
        img = Image.new('RGB', (width, height), color=theme["bg_color"])
        draw = ImageDraw.Draw(img)
        
        # 添加桌面元素
        elements = []
        
        # 任务栏（底部）
        taskbar_height = 40
        draw.rectangle(
            [0, height - taskbar_height, width, height],
            fill=theme.get("taskbar_color", theme["border_color"])
        )
        
        # 开始按钮
        start_button = UIElement(
            element_type="button",
            bounds=(10, height - taskbar_height + 5, 80, height - 5),
            text="开始",
            action="open_start_menu",
            style={"color": theme["button_color"]}
        )
        elements.append(start_button)
        
        # 系统托盘
        tray_width = 200
        tray_bounds = (width - tray_width, height - taskbar_height, width, height)
        draw.rectangle(tray_bounds, fill=theme.get("tray_color", theme["border_color"]))
        
        # 桌面图标（随机位置）
        num_icons = random.randint(3, 8)
        for i in range(num_icons):
            icon_size = 60
            x = random.randint(50, width - icon_size - 50)
            y = random.randint(50, height - taskbar_height - icon_size - 50)
            
            icon = UIElement(
                element_type="icon",
                bounds=(x, y, x + icon_size, y + icon_size),
                text=f"图标{i+1}",
                action="double_click",
                style={"color": (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))}
            )
            elements.append(icon)
            
            # 绘制图标
            draw.rectangle(icon.bounds, fill=icon.style["color"])
            draw.text(
                (x + 5, y + icon_size + 5),
                icon.text,
                fill=theme["text_color"]
            )
        
        # 绘制所有元素
        for element in elements:
            self.draw_ui_element(draw, element, theme)
        
        return img, elements
    
    def create_app_interface(self, app_type: str):
        """创建应用程序界面"""
        width, height = 1600, 900
        theme_name = random.choice(list(self.themes.keys()))
        theme = self.themes[theme_name]
        
        img = Image.new('RGB', (width, height), color=theme["bg_color"])
        draw = ImageDraw.Draw(img)
        
        elements = []
        
        if app_type == "web_browser":
            elements = self.create_browser_interface(draw, theme, width, height)
        elif app_type == "text_editor":
            elements = self.create_editor_interface(draw, theme, width, height)
        elif app_type == "file_explorer":
            elements = self.create_explorer_interface(draw, theme, width, height)
        
        return img, elements
    
    def create_browser_interface(self, draw, theme, width, height):
        """创建浏览器界面"""
        elements = []
        
        # 地址栏
        address_bar = UIElement(
            element_type="input",
            bounds=(50, 10, width - 250, 45),
            text="https://www.example.com",
            action="click_input",
            style={"bg_color": theme["input_color"], "text_color": theme["text_color"]}
        )
        elements.append(address_bar)
        
        # 标签页
        tab_width = 150
        for i in range(3):
            tab = UIElement(
                element_type="tab",
                bounds=(50 + i * tab_width, 50, 50 + (i + 1) * tab_width, 80),
                text=f"标签页 {i+1}",
                action="click_tab",
                style={"bg_color": theme["button_color"] if i == 0 else theme["border_color"]}
            )
            elements.append(tab)
        
        # 书签栏
        bookmark_bar = UIElement(
            element_type="toolbar",
            bounds=(50, 85, width - 50, 110),
            text="",
            action="click_bookmark",
            style={"bg_color": theme["border_color"]}
        )
        elements.append(bookmark_bar)
        
        # 网页内容区域
        content_area = UIElement(
            element_type="content",
            bounds=(50, 115, width - 50, height - 50),
            text="",
            action="scroll",
            style={"bg_color": (255, 255, 255)}
        )
        elements.append(content_area)
        
        # 在内容区域添加一些元素
        button = UIElement(
            element_type="button",
            bounds=(100, 200, 250, 240),
            text="点击这里",
            action="left_click",
            style={"bg_color": theme["button_color"], "text_color": (255, 255, 255)}
        )
        elements.append(button)
        
        link = UIElement(
            element_type="link",
            bounds=(100, 300, 300, 320),
            text="了解更多",
            action="left_click",
            style={"text_color": (0, 0, 255), "underline": True}
        )
        elements.append(link)
        
        return elements
    
    def create_editor_interface(self, draw, theme, width, height):
        """创建编辑器界面"""
        elements = []
        
        # 菜单栏
        menu_bar = UIElement(
            element_type="menu",
            bounds=(0, 0, width, 30),
            text="文件 编辑 视图 帮助",
            action="click_menu",
            style={"bg_color": theme["border_color"]}
        )
        elements.append(menu_bar)
        
        # 工具栏
        toolbar = UIElement(
            element_type="toolbar",
            bounds=(0, 30, width, 70),
            text="",
            action="click_tool",
            style={"bg_color": theme["border_color"]}
        )
        elements.append(toolbar)
        
        # 侧边栏
        sidebar = UIElement(
            element_type="sidebar",
            bounds=(0, 70, 200, height),
            text="",
            action="click_sidebar",
            style={"bg_color": (45, 45, 45)}
        )
        elements.append(sidebar)
        
        # 编辑区域
        editor = UIElement(
            element_type="editor",
            bounds=(200, 70, width, height),
            text="# Welcome to Code Editor\nprint('Hello World')",
            action="type_text",
            style={"bg_color": (30, 30, 30), "text_color": (200, 200, 200)}
        )
        elements.append(editor)
        
        # 状态栏
        status_bar = UIElement(
            element_type="statusbar",
            bounds=(0, height - 30, width, height),
            text="Line 1, Column 1 | Python",
            action="",
            style={"bg_color": theme["border_color"]}
        )
        elements.append(status_bar)
        
        return elements
    
    def create_explorer_interface(self, draw, theme, width, height):
        """创建文件资源管理器界面"""
        elements = []
        
        # 地址栏
        address_bar = UIElement(
            element_type="input",
            bounds=(50, 10, width - 100, 45),
            text="C:\\Users\\Documents",
            action="click_input",
            style={"bg_color": theme["input_color"], "text_color": theme["text_color"]}
        )
        elements.append(address_bar)
        
        # 导航窗格
        nav_pane = UIElement(
            element_type="sidebar",
            bounds=(0, 50, 200, height - 50),
            text="",
            action="click_sidebar",
            style={"bg_color": (240, 240, 240)}
        )
        elements.append(nav_pane)
        
        # 文件列表
        file_list = UIElement(
            element_type="list",
            bounds=(200, 50, width, height - 50),
            text="",
            action="click_file",
            style={"bg_color": (255, 255, 255)}
        )
        elements.append(file_list)
        
        # 添加一些模拟文件
        for i in range(5):
            file_item = UIElement(
                element_type="file",
                bounds=(220, 70 + i * 30, 600, 100 + i * 30),
                text=f"文件{i+1}.txt",
                action="click_file",
                style={"bg_color": (255, 255, 255), "text_color": (0, 0, 0)}
            )
            elements.append(file_item)
        
        return elements
    
    def draw_ui_element(self, draw, element: UIElement, theme: Dict):
        """绘制UI元素"""
        x1, y1, x2, y2 = element.bounds
        
        # 根据元素类型绘制
        if element.element_type in ["button", "icon"]:
            # 绘制按钮/图标
            bg_color = element.style.get("bg_color", theme["button_color"])
            draw.rectangle([x1, y1, x2, y2], fill=bg_color, outline=theme["border_color"], width=2)
            
            if element.text:
                # 居中文本
                try:
                    draw.text((x1 + 5, y1 + (y2 - y1 - 20) // 2), element.text, fill=element.style.get("text_color", (255, 255, 255)))
                except:
                    # 如果字体有问题，使用默认字体
                    draw.text((x1 + 5, y1 + (y2 - y1 - 20) // 2), element.text, fill=element.style.get("text_color", (255, 255, 255)))
        
        elif element.element_type == "input":
            # 绘制输入框
            draw.rectangle([x1, y1, x2, y2], fill=theme["input_color"], outline=theme["border_color"], width=1)
            
            if element.text:
                draw.text((x1 + 5, y1 + (y2 - y1 - 20) // 2), element.text, fill=theme["text_color"])
        
        elif element.element_type in ["menu", "toolbar", "statusbar"]:
            # 绘制工具栏/菜单栏
            bg_color = element.style.get("bg_color", theme["border_color"])
            draw.rectangle([x1, y1, x2, y2], fill=bg_color)
            
            if element.text:
                draw.text((x1 + 10, y1 + (y2 - y1 - 20) // 2), element.text, fill=theme["text_color"])
        
        elif element.element_type == "link":
            # 绘制链接
            text_color = element.style.get("text_color", (0, 0, 255))
            draw.text((x1, y1), element.text, fill=text_color)
            
            if element.style.get("underline", False):
                text_width = 0
                try:
                    text_width = draw.textlength(element.text)
                except:
                    text_width = len(element.text) * 10  # 估算文本宽度
                draw.line([x1, y2-5, x1 + text_width, y2-5], fill=text_color, width=1)
    
    def generate_random_operation(self, elements: List[UIElement]) -> Dict[str, Any]:
        """生成随机操作"""
        if not elements:
            return {"action": "idle", "target": None, "params": {}}
        
        # 随机选择一个元素
        element = random.choice(elements)
        
        # 根据元素类型确定操作
        if element.element_type == "button":
            action = "left_click"
        elif element.element_type == "input":
            action = random.choice(["click_input", "type_text"])
        elif element.element_type == "icon":
            action = random.choice(["left_click", "double_click", "right_click"])
        elif element.element_type == "link":
            action = "left_click"
        elif element.element_type == "file":
            action = random.choice(["left_click", "right_click"])
        else:
            action = random.choice(["left_click", "right_click", "hover"])
        
        # 构建操作参数
        params = {}
        if action == "type_text":
            params["text"] = random.choice(["Hello World", "test", "username", "password", "search query"])
        elif action == "click_input":
            params["cursor_position"] = random.randint(0, len(element.text) if element.text else 0)
        
        return {
            "action": action,
            "target_element": element.element_type,
            "target_bounds": element.bounds,
            "target_text": element.text,
            "params": params,
            "timestamp": time.time()
        }
    
    def generate_app_specific_operation(self, app_type: str, elements: List[UIElement]) -> Dict[str, Any]:
        """生成应用程序特定操作"""
        if not elements:
            return {"action": "idle", "target": None, "params": {}}
        
        # 根据应用类型选择操作
        if app_type == "web_browser":
            possible_actions = [
                "click_link", "click_button", "click_tab", 
                "click_address_bar", "scroll_page"
            ]
        elif app_type == "text_editor":
            possible_actions = [
                "type_text", "click_menu", "click_toolbar", 
                "click_editor", "click_sidebar"
            ]
        elif app_type == "file_explorer":
            possible_actions = [
                "click_file", "click_folder", "click_address_bar",
                "click_sidebar", "scroll_list"
            ]
        else:
            possible_actions = ["left_click", "right_click", "hover"]
        
        action = random.choice(possible_actions)
        
        # 选择相关元素
        relevant_elements = [e for e in elements if e.element_type in [
            "button", "link", "input", "file", "folder", "menu", "toolbar", "editor", "sidebar"
        ]]
        
        if relevant_elements:
            element = random.choice(relevant_elements)
        else:
            element = random.choice(elements)
        
        params = {}
        if action == "type_text":
            params["text"] = random.choice(["Hello World", "test code", "document content"])
        
        return {
            "action": action,
            "target_element": element.element_type,
            "target_bounds": element.bounds,
            "target_text": element.text,
            "params": params,
            "timestamp": time.time()
        }
    
    def save_sample(self, sample_id: str, image: Image.Image, operation: Dict, elements: List[UIElement]):
        """保存样本"""
        # 创建样本目录
        sample_dir = self.output_dir / "samples" / sample_id
        sample_dir.mkdir(exist_ok=True, parents=True)
        
        # 保存图像
        image_path = sample_dir / "screenshot.png"
        image.save(image_path, "PNG")
        
        # 保存标注
        annotation = {
            "sample_id": sample_id,
            "image_path": str(image_path.relative_to(self.output_dir)),
            "operation": operation,
            "elements": [
                {
                    "type": elem.element_type,
                    "bounds": elem.bounds,
                    "text": elem.text,
                    "action": elem.action
                }
                for elem in elements
            ],
            "created_at": datetime.now().isoformat()
        }
        
        # 使用样本ID作为标注文件名，以便与图像文件对应
        annotation_path = sample_dir / f"{sample_id}.json"
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)
    
    def save_dataset_info(self, info: Dict, dataset_name: str):
        """保存数据集信息"""
        info_path = self.output_dir / f"{dataset_name}_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    
    def generate_synthetic_dataset(self, total_samples: int = 1000):
        """生成完整的合成数据集"""
        print(f"开始生成完整数据集: {total_samples} 样本")
        
        # 分配样本数量
        basic_samples = int(total_samples * 0.5)  # 50% 基础操作
        app_samples = int(total_samples * 0.5)    # 50% 应用操作 (平均分配给3个应用)
        
        app_samples_per_type = app_samples // 3
        
        # 生成基础数据集
        self.create_basic_dataset(basic_samples)
        
        # 生成应用特定数据集
        app_types = ["web_browser", "text_editor", "file_explorer"]
        for app_type in app_types:
            self.create_app_specific_dataset(app_type, app_samples_per_type)
        
        print(f"数据集生成完成!")
        print(f"总计生成约 {total_samples} 个样本")
        print(f"数据集位置: {self.output_dir.absolute()}")

def main():
    """主函数"""
    print("=" * 60)
    print("AI操作员数据集生成器")
    print("=" * 60)
    
    creator = ProfessionalDatasetCreator("ai_operator_dataset")
    
    print("\n请选择生成模式:")
    print("  1. 快速测试数据集 (100 样本)")
    print("  2. 小型训练数据集 (500 样本)")
    print("  3. 标准训练数据集 (1000 样本)")
    print("  4. 大型训练数据集 (2000 样本)")
    print("  5. 退出")
    
    choice = input("\n请选择 (1-5): ").strip()
    
    if choice == "1":
        creator.generate_synthetic_dataset(100)
    elif choice == "2":
        creator.generate_synthetic_dataset(500)
    elif choice == "3":
        creator.generate_synthetic_dataset(1000)
    elif choice == "4":
        creator.generate_synthetic_dataset(2000)
    else:
        print("退出")

if __name__ == "__main__":
    main()