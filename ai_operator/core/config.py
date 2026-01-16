"""
AI操作员项目 - 核心配置文件
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging

@dataclass
class Config:
    """项目配置类"""
    
    # 项目路径配置
    PROJECT_ROOT: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    WORKFLOWS_DIR: Path = PROJECT_ROOT / "workflows"
    
    # DeepSeek API配置（必须通过环境变量提供，避免在仓库中明文存储）
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY", None)
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-reasoner"
    MAX_TOKENS: int = 6000
    
    # 屏幕分析配置
    SCREEN_CAPTURE_INTERVAL: float = 0.5  # 秒
    SCREEN_REGION: Optional[Tuple[int, int, int, int]] = None  # (left, top, right, bottom)
    SCREEN_RESIZE: Tuple[int, int] = (640, 480)
    
    # 动作配置
    AVAILABLE_ACTIONS: List[str] = field(default_factory=lambda: [
        'left_click',           # 左键单击
        'left_double_click',    # 左键双击（显式区分）
        'left_click_and_hold',  # 左键点击并保持
        'left_long_press',      # 左键长按
        'right_click',          # 右键单击
        'right_double_click',   # 右键双击
        'right_click_and_hold', # 右键按住
        'right_long_press',     # 右键长按
        'middle_click',         # 中键点击
        'drag_start',           # 开始拖拽
        'drag_end',             # 结束拖拽
        'scroll_pixels',        # 滚动指定像素量
        'smooth_scroll',        # 平滑滚动
        'horizontal_scroll',    # 水平滚动
        'keyboard_input',       # 键盘输入
        'paste_via_clipboard',  # 通过剪贴板粘贴大块文本
        'hotkey_press',         # 快捷键
        'window_focus',         # 窗口聚焦
        'window_minimize',      # 最小化窗口
        'window_maximize',      # 最大化窗口
        'window_close',         # 关闭窗口
        'copy_selection',       # 复制选择
        'paste_content',        # 粘贴内容
        'undo_action',          # 撤销操作
        'redo_action',          # 重做操作
        'save_file',            # 保存文件
        'open_file',            # 打开文件
        'search_text',          # 搜索文本
        'navigate_back',        # 后退
        'navigate_forward',     # 前进
        'refresh_page',         # 刷新页面
        'switch_tab',           # 切换标签页
        'new_tab',              # 新建标签页
        'close_tab',            # 关闭标签页
    ])
    
    # 模型训练配置
    TRAIN_BATCH_SIZE: int = 32
    TRAIN_EPOCHS: int = 100
    LEARNING_RATE: float = 0.001
    VALIDATION_SPLIT: float = 0.2
    
    # 数据集配置
    DATASET_SIZE: int = 10000  # 目标数据集大小
    AUGMENTATION_FACTOR: int = 5  # 数据增强倍数
    
    # GUI配置
    UI_REFRESH_RATE: int = 30  # FPS
    THEME: str = "dark"  # dark/light
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 性能配置
    MAX_MEMORY_USAGE: int = 4096  # MB
    MAX_CPU_USAGE: float = 0.8  # 80%
    
    def __post_init__(self):
        """初始化后创建必要目录"""
        for dir_path in [self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR, self.WORKFLOWS_DIR]:
            dir_path.mkdir(exist_ok=True, parents=True)
        # 检查敏感配置
        if not self.DEEPSEEK_API_KEY:
            logging.warning("DEEPSEEK_API_KEY 未设置 — 与 DeepSeek 相关的功能将无法使用。请通过环境变量 DEEPSEEK_API_KEY 提供。")
    
    @classmethod
    def from_yaml(cls, yaml_path: str = "config.yaml") -> 'Config':
        """从YAML文件加载配置"""
        yaml_path = Path(yaml_path)
        if yaml_path.exists():
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            
            # 将YAML配置转换为Config对象
            config_dict = {}
            for key, value in yaml_config.items():
                if key.upper() in cls.__dataclass_fields__:
                    config_dict[key.upper()] = value
            
            return cls(**config_dict)
        return cls()
    
    def to_yaml(self, yaml_path: str = "config.yaml"):
        """保存配置到YAML文件"""
        yaml_path = Path(yaml_path)
        config_dict = {
            key.lower(): getattr(self, key) 
            for key in self.__dataclass_fields__.keys()
            if not key.startswith('_')
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

# 创建全局配置实例
config = Config.from_yaml()