#!/usr/bin/env python3
"""
导出训练好的 PyTorch 模型为 ONNX 格式的脚本
用法示例:
  python scripts/export_onnx.py --checkpoint models/trained_example/best_model.pth --onnx-path models/trained_example/best_model.onnx
"""
import argparse
import os
import torch

from ai_operator.models.cnn_model import EnhancedCNN


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        # 返回一个有序元组，便于 ONNX 导出
        return (out['actions'], out['applications'], out['bounding_boxes'], out['mouse_positions'], out['segmentation'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='models/trained_example/best_model.pth')
    parser.add_argument('--onnx-path', type=str, default='models/trained_example/best_model.onnx')
    parser.add_argument('--num-actions', type=int, default=14)
    parser.add_argument('--screen-h', type=int, default=224)
    parser.add_argument('--screen-w', type=int, default=224)
    parser.add_argument('--opset', type=int, default=11)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'checkpoint not found: {args.checkpoint}')

    # 构建模型并加载权重
    model = EnhancedCNN(args.num_actions, screen_size=(args.screen_h, args.screen_w))
    state = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)

    model.eval()

    wrapper = ExportWrapper(model)

    dummy = torch.randn(1, 3, args.screen_h, args.screen_w)

    os.makedirs(os.path.dirname(args.onnx_path) or '.', exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        args.onnx_path,
        input_names=['input_image'],
        output_names=['actions', 'applications', 'bounding_boxes', 'mouse_positions', 'segmentation'],
        opset_version=args.opset,
        do_constant_folding=True
    )

    print(f'导出完成 -> {args.onnx_path}')


if __name__ == '__main__':
    main()
