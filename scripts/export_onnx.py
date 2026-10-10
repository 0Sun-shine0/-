#!/usr/bin/env python3
"""
导出训练好的 PyTorch 模型为 ONNX 格式的脚本
用法示例:
  python scripts/export_onnx.py --checkpoint models/trained_example/best_model.pth --onnx-path models/trained_example/best_model.onnx
"""
import argparse
import os
import re
import sys
import torch

# ensure project root is on sys.path so scripts can be run from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# torch.onnx 的成功提示里带 emoji，Windows 控制台默认 GBK 会直接抛
# UnicodeEncodeError，把导出流程打断；统一改成 UTF-8 输出。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

from ai_operator.models.cnn_model import EnhancedCNN, spatial_coord_from_state


def infer_num_actions(state_dict) -> int:
    """从 state_dict 推断动作头的输出维度。

    ``action_head`` 是一个 ``nn.Sequential``，其中第 0 层是 [512, 2048]，
    最后一层才是输出层（形状 [num_actions, 256]）。必须按层号取**最后**一层，
    否则会把中间层的 512 当成类别数（旧实现就是这个 bug）。
    """
    best_index, num_actions = -1, None
    for key, tensor in state_dict.items():
        match = re.match(r"^action_head\.(\d+)\.weight$", key)
        if match and tensor.ndim == 2:
            index = int(match.group(1))
            if index > best_index:
                best_index, num_actions = index, int(tensor.shape[0])
    return num_actions if num_actions is not None else 14


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
    parser.add_argument('--num-actions', type=int, default=None, help='如果不指定，将尝试从 checkpoint 推断')
    parser.add_argument('--screen-h', type=int, default=224)
    parser.add_argument('--screen-w', type=int, default=224)
    parser.add_argument('--opset', type=int, default=11)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'checkpoint not found: {args.checkpoint}')

    # 加载 checkpoint 并推断 num_actions（如果未提供）
    state = torch.load(args.checkpoint, map_location='cpu')
    state_dict = state['model_state'] if isinstance(state, dict) and 'model_state' in state else state

    if args.num_actions is None:
        num_actions = infer_num_actions(state_dict)
    else:
        num_actions = args.num_actions

    # 构建模型并加载权重
    # 坐标头的类型必须与训练时一致，否则 coord_head.* 会加载失败或坐标被多套一次 sigmoid
    spatial = spatial_coord_from_state(state_dict)
    model = EnhancedCNN(num_actions, screen_size=(args.screen_h, args.screen_w),
                        pretrained=False, spatial_coord=spatial)
    model.load_state_dict(state_dict)
    print(f"坐标头: {'空间 soft-argmax' if spatial else '全局池化 + 线性'}")

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
