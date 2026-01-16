"""示例训练脚本：生成合成数据并运行一次短训练以验证训练流水线"""
import os
import sys
import json
import numpy as np

# 将项目根路径加入 sys.path 以便直接运行脚本时可以导入包
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai_operator.core.config import Config
from ai_operator.models.training_pipeline import TrainingPipeline


def generate_synthetic_dataset(path: str, num_samples: int = 20, image_size=(64, 64)):
    samples = []
    for i in range(num_samples):
        # 生成随机图像 (H,W,C)
        img = (np.random.rand(image_size[0], image_size[1], 3) * 255).astype(np.uint8).tolist()

        # 随机动作
        coords = [int(np.random.randint(0, max(image_size))), int(np.random.randint(0, max(image_size)))]
        action_type = 'left_click' if np.random.rand() > 0.5 else 'right_click'

        sample = {
            'screen_input': img,
            'action_output': {
                'coordinates': coords,
                'action_type': action_type
            }
        }
        samples.append(sample)

    dataset = {
        'metadata': {
            'created_by': 'train_example',
            'sample_count': len(samples)
        },
        'samples': samples
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False)

    print(f"合成数据集已保存: {path} (samples={len(samples)})")


def main():
    import argparse

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, 'data')
    ds_path = os.path.join(data_dir, 'synthetic_dataset_small.json')

    parser = argparse.ArgumentParser(description='示例训练脚本')
    parser.add_argument('--dataset', type=str, default=ds_path)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--save-dir', type=str, default=os.path.join(root, 'models', 'example_run'))
    parser.add_argument('--use-amp', action='store_true', help='当可用时启用混合精度')
    parser.add_argument('--samples', type=int, default=40, help='合成样本数量')
    args = parser.parse_args()

    generate_synthetic_dataset(args.dataset, num_samples=args.samples, image_size=(64, 64))

    cfg = Config()
    tp = TrainingPipeline(cfg)

    # 先准备数据
    train_data, val_data = tp.prepare_dataset(args.dataset, validation_split=0.2)

    # 根据标签向量维度设置模型输出
    sample_y = train_data[1]
    num_actions = sample_y.shape[1] if hasattr(sample_y, 'shape') else len(cfg.AVAILABLE_ACTIONS)
    # 自动启用 AMP 仅当 GPU 可用并且用户要求
    enable_amp = args.use_amp and (tp.device.type == 'cuda')
    tp.setup_model('EnhancedCNN', num_actions=num_actions, lr=cfg.LEARNING_RATE, use_amp=enable_amp)

    # 运行训练
    history = tp.train_model(train_data, val_data, epochs=args.epochs, batch_size=args.batch_size, save_dir=args.save_dir, use_amp=enable_amp)

    print('训练历史:', history)


if __name__ == '__main__':
    main()
