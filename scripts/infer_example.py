#!/usr/bin/env python3
"""
示例推理脚本：优先使用 ONNX Runtime（如果可用），否则回退到 PyTorch
用法示例：
  python scripts/infer_example.py --use-onnx --onnx-path models/trained_example/best_model.onnx
  python scripts/infer_example.py --checkpoint models/trained_example/best_model.pth
"""
import argparse
import os
import numpy as np
import torch

try:
    import onnxruntime as ort
except Exception:
    ort = None

from ai_operator.models.cnn_model import EnhancedCNN


def infer_pytorch(checkpoint, num_actions=14, screen_h=224, screen_w=224):
    model = EnhancedCNN(num_actions, screen_size=(screen_h, screen_w))
    state = torch.load(checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, screen_h, screen_w)
    with torch.no_grad():
        out = model(dummy)

    actions = torch.softmax(out['actions'], dim=1)
    topk = torch.topk(actions, 3)
    print('PyTorch 推理：top-3 actions ->')
    for idx, prob in zip(topk.indices[0], topk.values[0]):
        print(f'  action {idx.item()}  prob={prob.item():.4f}')


def infer_onnx(onnx_path, screen_h=224, screen_w=224):
    if ort is None:
        raise RuntimeError('onnxruntime 未安装，无法执行 ONNX 推理')

    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, screen_h, screen_w).astype(np.float32)
    outputs = sess.run(None, {input_name: dummy})

    actions = outputs[0]
    topk_idx = np.argsort(actions[0])[-3:][::-1]
    print('ONNX 推理：top-3 actions ->')
    for idx in topk_idx:
        print(f'  action {int(idx)}  score={float(actions[0, idx]):.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-onnx', action='store_true')
    parser.add_argument('--onnx-path', type=str, default='models/trained_example/best_model.onnx')
    parser.add_argument('--checkpoint', type=str, default='models/trained_example/best_model.pth')
    parser.add_argument('--num-actions', type=int, default=14)
    parser.add_argument('--screen-h', type=int, default=224)
    parser.add_argument('--screen-w', type=int, default=224)
    args = parser.parse_args()

    if args.use_onnx:
        if not os.path.exists(args.onnx_path):
            print('ONNX 模型不存在，尝试先用 PyTorch 加载 checkpoint')
            if os.path.exists(args.checkpoint):
                infer_pytorch(args.checkpoint, args.num_actions, args.screen_h, args.screen_w)
            else:
                raise FileNotFoundError('既没有 ONNX 模型也没有 checkpoint')
        else:
            infer_onnx(args.onnx_path, args.screen_h, args.screen_w)
    else:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f'checkpoint 未找到: {args.checkpoint}')
        infer_pytorch(args.checkpoint, args.num_actions, args.screen_h, args.screen_w)


if __name__ == '__main__':
    main()
