# AI智能桌面操作员

AI智能桌面操作员是一个基于深度学习和大语言模型的智能桌面自动化系统，能够理解用户的自然语言指令并执行相应的桌面操作。

## 功能特性

- 智能任务理解：使用 DeepSeek API 进行自然语言理解和任务分解
- 屏幕分析：使用增强 CNN 模型分析屏幕内容和识别 UI 元素
- 自动化操作：支持鼠标点击、键盘输入、拖拽、滚轮等自动化操作
- 工作流支持：提供复杂任务的自动化工作流（如文档写作）
- 图形界面：提供基于 PyQt6 的交互界面

## 项目结构

```
ai_operator/
├── core/           # 核心配置和基础组件
├── agents/         # AI智能体
── models/         # 机器学习模型
├── automation/     # 自动化控制器
├── ui/             # 用户界面
├── data/           # 数据处理
└── main.py         # 主入口文件
```

## 安装和运行

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 配置 API 密钥（推荐）：

推荐通过环境变量设置 `DEEPSEEK_API_KEY`，避免在代码或仓库中明文保存密钥。例如（Windows PowerShell）：

```powershell
setx DEEPSEEK_API_KEY "your_api_key_here"
```

或者在当前会话中：

```powershell
$env:DEEPSEEK_API_KEY = "your_api_key_here"
```

3. 运行图形界面：

```bash
python -m ai_operator.main --mode gui
```

4. 运行其他模式与训练示例：

```bash
# 运行简单测试
python -m ai_operator.main --mode test

# 运行写作工作流
python -m ai_operator.main --mode workflow --topic "人工智能发展趋势" --requirements "{\"length\": \"1000字\", \"tone\": \"专业\"}"

# 训练示例（生成合成数据并短训练）
python scripts/train_example.py --epochs 1 --batch-size 8

# 在有 GPU 时启用 AMP
python scripts/train_example.py --epochs 5 --batch-size 16 --use-amp
```

训练模式（通过 main）现在支持以下参数：

- `--dataset` 数据集路径
- `--model-path` 模型保存路径或 checkpoint 保存目录
- `--save-dir` 训练输出保存目录（覆盖 model-path 用法）
- `--resume` 从 checkpoint 恢复训练
- `--epochs` 训练轮数
- `--batch-size` 批次大小
- `--use-amp` 在有 GPU 时启用混合精度训练

## 使用说明

1. 启动图形界面后，系统会自动捕获屏幕内容
2. 在任务输入框中输入自然语言指令
3. 点击“执行任务”按钮，AI 将分析并执行相应操作
4. 通过右侧面板可以调整模型参数和查看操作历史

## 注意事项

- 请确保在安全和受控的环境中使用自动化功能
- 系统需要访问屏幕内容和控制鼠标键盘的权限（在某些系统需额外授权）
- 请通过环境变量或秘密管理来保护 API 密钥
- 训练依赖较多（包含 PyTorch/torchvision、TensorFlow（可选）等），请在具有合适资源的平台上运行训练
# AI智能桌面操作员

AI智能桌面操作员是一个基于深度学习和大语言模型的智能桌面自动化系统，能够理解用户的自然语言指令并执行相应的桌面操作。

## 功能特性

- **智能任务理解**：使用DeepSeek API进行自然语言理解和任务分解
- **屏幕分析**：使用增强CNN模型分析屏幕内容和识别UI元素
- **自动化操作**：支持鼠标点击、键盘输入、拖拽等桌面操作
- **工作流支持**：提供复杂任务的自动化工作流（如文档写作）
- **图形界面**：提供直观的PyQt6图形用户界面

## 项目结构

```
ai_operator/
├── core/           # 核心配置和基础组件
│   └── config.py   # 项目配置
├── agents/         # AI智能体
│   └── deepseek_agent.py  # DeepSeek集成
├── models/         # 机器学习模型
│   ├── cnn_model.py       # 增强CNN模型
│   └── training_pipeline.py  # 训练流水线
├── automation/     # 自动化控制器
│   ├── mouse_controller.py    # 鼠标控制器
│   └── keyboard_controller.py # 键盘控制器
├── ui/             # 用户界面
│   └── main_window.py  # 主界面
├── data/           # 数据处理
│   └── data_processor.py  # 数据处理器
├── workflows/      # 自动化工作流
│   └── writing_workflow.py  # 写作工作流
└── main.py         # 主入口文件
```

## 安装和运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置API密钥：
在`ai_operator/core/config.py`中设置DeepSeek API密钥

3. 运行图形界面：
```bash
python -m ai_operator.main --mode gui
```

4. 运行其他模式：
```bash
# 运行简单测试
python -m ai_operator.main --mode test

# 运行写作工作流
python -m ai_operator.main --mode workflow --topic "人工智能发展趋势" --requirements "{\"length\": \"1000字\", \"tone\": \"专业\"}"
```

## 使用说明

1. 启动图形界面后，系统会自动捕获屏幕内容
2. 在任务输入框中输入您的自然语言指令
3. 点击"执行任务"按钮，AI将分析指令并执行相应操作
4. 通过右侧面板可以调整模型参数和查看操作历史

## 注意事项

- 请确保在安全的环境中使用自动化功能
- 系统需要访问屏幕内容和控制鼠标键盘的权限
- API调用需要有效的DeepSeek密钥