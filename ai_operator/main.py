"""
AI智能桌面操作员 - 主入口文件
启动应用程序和协调各个模块
"""
import sys
import argparse
from typing import Optional
import logging

from .core.config import Config
from .ui.main_window import MainWindow
from .agents.deepseek_agent import DeepSeekAgent
from .models.cnn_model import EnhancedCNN
from .models.inference import ScreenPredictor
from .models.training_pipeline import TrainingPipeline
from .workflows.writing_workflow import WritingWorkflow
from .data.data_processor import DataProcessor
from .automation.mouse_controller import AdvancedMouseController
from .automation.keyboard_controller import KeyboardController


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ai_operator.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def run_gui_app():
    """运行GUI应用程序"""
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def run_training_pipeline(dataset_path: str, model_save_path: str):
    """运行训练流水线"""
    config = Config()
    pipeline = TrainingPipeline(config)

    # 使用配置和命令行传入的保存路径/参数
    num_actions = len(config.AVAILABLE_ACTIONS) if hasattr(config, 'AVAILABLE_ACTIONS') else 14
    pipeline.setup_model("EnhancedCNN", num_actions=num_actions, lr=config.LEARNING_RATE, use_amp=True)

    # 准备数据集
    print(f"准备数据集: {dataset_path}")
    train_data, val_data = pipeline.prepare_dataset(dataset_path)

    # 训练模型（兼容新的接口）
    print("开始训练模型...")
    history = pipeline.train_model(
        train_data,
        val_data,
        epochs=50,
        batch_size=32,
        save_dir=model_save_path,
        resume_from=None,
        use_amp=True
    )

    print("训练完成！")


def run_writing_workflow(topic: str, requirements: dict):
    """运行写作工作流"""
    config = Config()
    workflow = WritingWorkflow(config)
    
    print(f"开始写作任务: {topic}")
    result = workflow.execute_writing_task(topic, requirements)
    
    if result['success']:
        print("写作任务完成！")
        print(f"最终文档长度: {len(result['final_document'])} 字符")
    else:
        print(f"写作任务失败: {result['error']}")
    
    return result


def run_simple_test():
    """自检：环境、DeepSeek 可用性、屏幕分析模型、数据处理器。"""
    config = Config()

    print("=== AI智能桌面操作员 - 自检 ===")
    print(f"[1/4] 配置加载            : OK（项目根目录 {config.PROJECT_ROOT.parent}）")

    # DeepSeek：缺少密钥不再是致命错误，这里如实报告
    agent = DeepSeekAgent(config)
    if agent.is_available():
        print("[2/4] DeepSeek 智能体     : 已配置密钥")
        test_result = agent.process_task("你好，请简单介绍一下自己。")
        keys = ", ".join(sorted(test_result.keys())[:4]) if isinstance(test_result, dict) else "非字典"
        print(f"      试调用返回字段      : {keys}")
    else:
        print("[2/4] DeepSeek 智能体     : 未配置密钥 —— 任务理解不可用")
        print("      设置方式            : 环境变量 DEEPSEEK_API_KEY，或项目根目录 .env（KEY=VALUE）")

    # 屏幕分析模型：以前这里只是 `EnhancedCNN(34)` 随机初始化后打印一句"初始化成功"，
    # 打印成功并不代表可用。现在改为真正加载已训练模型。
    try:
        predictor = ScreenPredictor.load()
        info = predictor.info
        print(f"[3/4] 屏幕分析模型        : {info.name}"
              f"（{info.num_actions} 类，{info.input_size}px 输入，"
              f"{'空间坐标头' if info.spatial_coord else '全局池化坐标头'}）")
        acc = info.metrics.get("best_acc")
        coord = info.metrics.get("best_coord_mae_px")
        if isinstance(acc, (int, float)) or isinstance(coord, (int, float)):
            detail = []
            if isinstance(acc, (int, float)):
                detail.append(f"验证准确率 {acc:.1%}")
            if isinstance(coord, (int, float)):
                detail.append(f"坐标误差 {coord:.0f}px")
            print(f"      训练指标            : {', '.join(detail)}")
    except FileNotFoundError as e:
        print("[3/4] 屏幕分析模型        : 未找到可用模型")
        for line in str(e).splitlines()[1:]:
            print(f"      {line}")

    # 数据处理器
    processor = DataProcessor(config)
    print(f"[4/4] 数据处理器          : {'OK' if processor is not None else '失败'}")

    print("\n自检完成。")


def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description='AI智能桌面操作员')
    parser.add_argument('--mode', type=str, default='gui', 
                       choices=['gui', 'train', 'workflow', 'test'],
                       help='运行模式: gui(图形界面), train(训练), workflow(工作流), test(测试)')
    parser.add_argument('--dataset', type=str, help='数据集路径（训练模式）')
    parser.add_argument('--model-path', type=str, help='模型保存路径（训练模式）或checkpoint保存目录')
    parser.add_argument('--resume', type=str, help='从checkpoint恢复（训练模式）')
    parser.add_argument('--save-dir', type=str, help='训练输出保存目录（训练模式）')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数（训练模式）')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小（训练模式）')
    parser.add_argument('--use-amp', action='store_true', help='在有GPU时启用混合精度训练')
    parser.add_argument('--topic', type=str, help='写作主题（工作流模式）')
    parser.add_argument('--requirements', type=str, help='任务要求JSON字符串（工作流模式）')
    
    args = parser.parse_args()
    
    logger.info(f"启动AI智能桌面操作员，模式: {args.mode}")
    
    if args.mode == 'gui':
        logger.info("启动GUI应用程序")
        run_gui_app()
        
    elif args.mode == 'train':
        if not args.dataset or not args.model_path:
            print("训练模式需要指定 --dataset 和 --model-path 参数")
            sys.exit(1)
        
        logger.info(f"启动训练流水线，数据集: {args.dataset}")
        # 将命令行参数传递给训练流水线
        pipeline_config = {
            'dataset': args.dataset,
            'save_dir': args.save_dir or args.model_path,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'resume': args.resume,
            'use_amp': args.use_amp
        }

        # 初始化并运行训练
        config = Config()
        tp = TrainingPipeline(config)
        num_actions = len(config.AVAILABLE_ACTIONS) if hasattr(config, 'AVAILABLE_ACTIONS') else 14
        # 先准备数据，根据标签维度再创建模型以保证输出维度对齐
        train_data, val_data = tp.prepare_dataset(pipeline_config['dataset'], validation_split=config.VALIDATION_SPLIT)
        sample_y = train_data[1]
        inferred_actions = sample_y.shape[1] if hasattr(sample_y, 'shape') else num_actions

        tp.setup_model('EnhancedCNN', num_actions=inferred_actions, lr=config.LEARNING_RATE, use_amp=args.use_amp)

        tp.train_model(
            train_data,
            val_data,
            epochs=pipeline_config['epochs'],
            batch_size=pipeline_config['batch_size'],
            save_dir=pipeline_config['save_dir'],
            resume_from=pipeline_config['resume'],
            use_amp=pipeline_config['use_amp']
        )
        
    elif args.mode == 'workflow':
        if not args.topic:
            print("工作流模式需要指定 --topic 参数")
            sys.exit(1)
        
        # 解析任务要求
        import json
        requirements = {}
        if args.requirements:
            try:
                requirements = json.loads(args.requirements)
            except json.JSONDecodeError:
                print("任务要求不是有效的JSON格式")
                sys.exit(1)
        
        logger.info(f"启动写作工作流，主题: {args.topic}")
        run_writing_workflow(args.topic, requirements)
        
    elif args.mode == 'test':
        logger.info("运行简单测试")
        run_simple_test()
        
    else:
        print(f"未知模式: {args.mode}")
        parser.print_help()
        sys.exit(1)


# 直接运行GUI的便捷函数
def start_gui():
    """直接启动GUI界面的便捷函数"""
    run_gui_app()


# 用于从命令行直接运行的入口点
if __name__ == "__main__":
    main()