#!/usr/bin/env python3
"""
打包发布脚本：将推理脚本、导出脚本和可选 ONNX 模型打包为 zip
用法示例:
  python scripts/create_release.py --include-model
"""
import argparse
import os
import zipfile
from datetime import datetime


def collect_files(include_model: bool):
    base = os.getcwd()
    files = [
        'scripts/infer_example.py',
        'scripts/export_onnx.py',
        'README.md',
    ]
    model_path = 'models/trained_example/best_model.onnx'
    if include_model and os.path.exists(model_path):
        files.append(model_path)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-model', action='store_true', help='如果存在则包含 ONNX 模型')
    parser.add_argument('--out-dir', type=str, default='dist')
    args = parser.parse_args()

    files = collect_files(args.include_model)
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_name = os.path.join(args.out_dir, f'release_{ts}.zip')

    with zipfile.ZipFile(out_name, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if os.path.exists(f):
                z.write(f, arcname=os.path.basename(f))

    print(f'发布包已生成: {out_name}')


if __name__ == '__main__':
    main()
