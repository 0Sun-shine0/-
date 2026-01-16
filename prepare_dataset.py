"""
AI桌面操作员 - 数据集准备和训练启动脚本
"""
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import cv2
import torch

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def create_dataset_structure():
    """创建数据集目录结构"""
    print("创建数据集目录结构...")
    
    dataset_dir = Path("ai_operator_dataset")
    splits = ["train", "val", "test"]
    subdirs = ["images", "annotations"]
    
    for split in splits:
        for subdir in subdirs:
            (dataset_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"数据集目录结构创建完成: {dataset_dir}")
    return dataset_dir

def generate_basic_dataset(num_samples=1000):
    """生成基础数据集"""
    print(f"开始生成基础数据集 ({num_samples} 个样本)...")
    
    from ai_operator.data_generation.dataset_creator import ProfessionalDatasetCreator
    
    creator = ProfessionalDatasetCreator("ai_operator_dataset")
    creator.generate_synthetic_dataset(num_samples)
    
    print("基础数据集生成完成!")

def validate_dataset():
    """验证数据集"""
    print("验证数据集...")
    
    from ai_operator.utils.dataset_validator import DatasetValidator
    
    validator = DatasetValidator("ai_operator_dataset")
    result = validator.validate_all()
    
    print(f"验证完成! 发现 {len(result['issues'])} 个问题")
    
    if result['issues']:
        print("主要问题:")
        for issue in result['issues'][:10]:  # 只显示前10个问题
            print(f"  - {issue}")
    
    # 生成报告
    validator.generate_report("dataset_validation_report.html")
    
    return result

def split_dataset():
    """划分数据集"""
    print("划分数据集为训练集、验证集和测试集...")
    
    dataset_dir = Path("ai_operator_dataset")
    samples_dir = dataset_dir / "samples"
    
    if not samples_dir.exists():
        print("没有找到样本目录，跳过分割")
        return
    
    # 获取所有样本
    all_samples = [d for d in samples_dir.iterdir() if d.is_dir()]
    total_samples = len(all_samples)
    
    if total_samples == 0:
        print("没有找到样本，跳过分割")
        return
    
    # 计算分割点
    train_end = int(total_samples * 0.7)  # 70% 训练
    val_end = int(total_samples * 0.85)   # 15% 验证 (总共85%)
    
    # 重新排列样本以确保随机性
    import random
    random.seed(42)  # 确保结果可重现
    samples_list = list(all_samples)
    random.shuffle(samples_list)
    
    # 分割样本
    train_samples = samples_list[:train_end]
    val_samples = samples_list[train_end:val_end]
    test_samples = samples_list[val_end:]
    
    print(f"数据集分割:")
    print(f"  训练集: {len(train_samples)} 样本")
    print(f"  验证集: {len(val_samples)} 样本")
    print(f"  测试集: {len(test_samples)} 样本")
    
    # 移动文件
    splits_map = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }
    
    for split_name, samples in splits_map.items():
        split_dir = dataset_dir / split_name
        for sample_dir in samples:
            # 复制样本到对应分割目录
            import shutil
            # 遍历样本目录中的所有内容
            for item in sample_dir.iterdir():
                if item.name.endswith(('.png', '.jpg', '.jpeg')):
                    # 复制图像到对应分割的images目录
                    target_img_dir = split_dir / "images" / sample_dir.name
                    target_img_dir.mkdir(exist_ok=True)
                    shutil.copy2(item, target_img_dir / item.name)
                elif item.name.endswith('.json'):
                    # 复制标注到对应分割的annotations目录
                    shutil.copy2(item, split_dir / "annotations" / item.name)
    
    print("数据集分割完成!")

def train_model():
    """开始训练模型"""
    print("开始训练模型...")
    
    try:
        from ai_operator.models.training_pipeline import TrainingPipeline
        from ai_operator.core.config import Config
        import numpy as np
        import json
        
        config = Config()
        pipeline = TrainingPipeline(config)
        
        print("设置模型...")
        pipeline.setup_model("EnhancedCNN")
        
        print("准备数据集...")
        
        # 检查是否有已分割的数据集
        train_dir = Path("ai_operator_dataset/train/images")
        val_dir = Path("ai_operator_dataset/val/images")
        test_dir = Path("ai_operator_dataset/test/images")
        
        if not train_dir.exists():
            print("未找到训练数据，跳过训练")
            return
        
        # 读取训练图像和标注
        train_images = []
        train_annotations = []
        
        # 遍历train/images下的所有子目录
        for sub_dir in train_dir.iterdir():
            if sub_dir.is_dir():
                for img_file in sub_dir.glob("*.png"):
                    # 对于每个图像，查找对应的标注
                    # 根据子目录名构建标注文件名
                    annotation_file = Path("ai_operator_dataset/train/annotations") / f"{sub_dir.name}.json"
                    if annotation_file.exists():
                        # 加载图像数据
                        from PIL import Image
                        img = Image.open(img_file)
                        # 统一调整图像尺寸为模型输入尺寸 (例如 224x224)
                        img = img.resize((224, 224))
                        img_array = np.array(img)
                        # 转换为模型期望的格式
                        img_array = img_array.astype(np.float32) / 255.0
                        # 调整维度顺序从 (H, W, C) 到 (C, H, W) 以适应PyTorch
                        img_array = np.transpose(img_array, (2, 0, 1))
                        train_images.append(img_array)
                        
                        # 加载标注数据
                        with open(annotation_file, "r", encoding="utf-8") as f:
                            annotation_data = json.load(f)
                        train_annotations.append(annotation_data)
        
        if len(train_images) == 0:
            print("没有找到训练数据，跳过训练")
            return
        
        # 确保所有图像具有相同的尺寸
        if train_images:
            # 获取第一个图像的尺寸
            first_img_shape = train_images[0].shape
            # 确保所有图像都具有相同的尺寸
            processed_train_images = []
            for img in train_images:
                if img.shape != first_img_shape:
                    # 调整图像尺寸以匹配
                    target_size = (first_img_shape[1], first_img_shape[0])  # (width, height)
                    if len(img.shape) == 3 and len(first_img_shape) == 3:
                        # 彩色图像
                        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                    elif len(img.shape) == 2 and len(first_img_shape) == 3:
                        # 灰度图转彩色图
                        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                        img_resized = np.stack([img_resized, img_resized, img_resized], axis=-1)
                    elif len(img.shape) == 3 and len(first_img_shape) == 2:
                        # 彩色图转灰度图
                        img_resized = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        img_resized = cv2.resize(img_resized, (first_img_shape[1], first_img_shape[0]), interpolation=cv2.INTER_AREA)
                    else:
                        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                    processed_train_images.append(img_resized)
                else:
                    processed_train_images.append(img)
            X_train = np.array(processed_train_images)
        else:
            X_train = np.array([])
        
        # 处理验证集
        val_images = []
        val_annotations = []
        
        if val_dir.exists():
            # 遍历val/images下的所有子目录
            for sub_dir in val_dir.iterdir():
                if sub_dir.is_dir():
                    for img_file in sub_dir.glob("*.png"):
                        annotation_file = Path("ai_operator_dataset/val/annotations") / f"{sub_dir.name}.json"
                        if annotation_file.exists():
                            from PIL import Image
                            img = Image.open(img_file)
                            # 统一调整图像尺寸为模型输入尺寸 (例如 224x224)
                            img = img.resize((224, 224))
                            img_array = np.array(img)
                            img_array = img_array.astype(np.float32) / 255.0
                            # 调整维度顺序从 (H, W, C) 到 (C, H, W) 以适应PyTorch
                            img_array = np.transpose(img_array, (2, 0, 1))
                            val_images.append(img_array)
                            
                            with open(annotation_file, "r", encoding="utf-8") as f:
                                annotation_data = json.load(f)
                            val_annotations.append(annotation_data)
        
        # 处理验证集图像尺寸
        if val_images:
            # 获取第一个图像的尺寸
            if len(val_images) > 0:
                first_val_shape = val_images[0].shape
                # 确保所有图像都具有相同的尺寸
                processed_val_images = []
                for img in val_images:
                    if img.shape != first_val_shape:
                        # 调整图像尺寸以匹配
                        target_size = (first_val_shape[1], first_val_shape[0])  # (width, height)
                        if len(img.shape) == 3 and len(first_val_shape) == 3:
                            # 彩色图像
                            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                        elif len(img.shape) == 2 and len(first_val_shape) == 3:
                            # 灰度图转彩色图
                            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                            img_resized = np.stack([img_resized, img_resized, img_resized], axis=-1)
                        elif len(img.shape) == 3 and len(first_val_shape) == 2:
                            # 彩色图转灰度图
                            img_resized = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                            img_resized = cv2.resize(img_resized, (first_val_shape[1], first_val_shape[0]), interpolation=cv2.INTER_AREA)
                        else:
                            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                        processed_val_images.append(img_resized)
                    else:
                        processed_val_images.append(img)
                X_val = np.array(processed_val_images)
            else:
                X_val = np.array([])
        else:
            X_val = np.array([])
        
        # 创建动作输出向量
        from ai_operator.data.data_processor import DataProcessor
        data_processor = DataProcessor(config)
        
        y_train = []
        for annotation in train_annotations:
            action_vector = data_processor._action_to_vector(annotation)
            y_train.append(action_vector)
        y_train = np.array(y_train)
        
        y_val = []
        if val_annotations:
            for annotation in val_annotations:
                action_vector = data_processor._action_to_vector(annotation)
                y_val.append(action_vector)
            y_val = np.array(y_val)
        
        print(f"训练数据形状: X_train={X_train.shape}, y_train={y_train.shape}")
        if X_val.shape[0] > 0:
            print(f"验证数据形状: X_val={X_val.shape}, y_val={y_val.shape}")
        
        # 准备数据元组
        train_data = (X_train, y_train)
        val_data = (X_val, y_val) if X_val.shape[0] > 0 else (X_train[:len(X_train)//5], y_train[:len(y_train)//5])  # 使用训练集的20%作为验证集
        
        # 开始训练
        print("开始训练...")
        history = pipeline.train_model(train_data, val_data, epochs=10, batch_size=16, verbose=1)
        
        print("模型训练完成!")
        
        # 保存模型
        model_save_path = "ai_operator_dataset/models"
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(pipeline.model.state_dict(), f"{model_save_path}/ai_operator_model.pth")
        print(f"模型已保存到: {model_save_path}/ai_operator_model.pth")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("=" * 60)
    print("AI桌面操作员 - 数据集准备和训练")
    print("=" * 60)
    
    print("\n步骤 1: 创建数据集目录结构")
    dataset_dir = create_dataset_structure()
    
    print("\n步骤 2: 生成基础数据集")
    generate_basic_dataset(500)  # 生成500个样本作为演示
    
    print("\n步骤 3: 验证数据集")
    validation_result = validate_dataset()
    
    print("\n步骤 4: 分割数据集")
    split_dataset()
    
    print("\n步骤 5: 准备训练")
    print("数据集准备完成! 现在可以开始训练模型了")
    
    print("\n可选步骤 - 开始训练:")
    print("  1. 使用生成的数据集训练模型")
    print("  2. 调整训练参数")
    print("  3. 监控训练过程")
    
    choice = input("\n是否开始模型训练? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("\n步骤 6: 开始训练")
        train_model()
    
    print("\n" + "=" * 60)
    print("数据集准备和训练流程完成!")
    print(f"数据集位置: {dataset_dir.absolute()}")
    print("验证报告: dataset_validation_report.html")
    print("=" * 60)

if __name__ == "__main__":
    main()