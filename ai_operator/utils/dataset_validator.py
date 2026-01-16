"""
数据集验证和质量检查工具
"""
import json
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt

class DatasetValidator:
    """数据集验证器"""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.issues = []
        self.stats = {}
    
    def validate_all(self) -> Dict[str, Any]:
        """全面验证数据集"""
        print(f"验证数据集: {self.dataset_dir}")
        
        checks = [
            self.check_directory_structure,
            self.check_image_files,
            self.check_annotation_files,
            self.check_class_balance,
            self.check_image_quality,
            self.check_annotation_quality,
        ]
        
        for check_func in checks:
            print(f"\n执行检查: {check_func.__name__}")
            check_func()
        
        return {
            "issues": self.issues,
            "statistics": self.stats,
            "passed": len(self.issues) == 0
        }
    
    def check_directory_structure(self):
        """检查目录结构"""
        required_dirs = ["train", "val", "test"]
        
        for dir_name in required_dirs:
            dir_path = self.dataset_dir / dir_name
            if not dir_path.exists():
                self.issues.append(f"缺少目录: {dir_name}")
            else:
                # 检查子目录
                subdirs = ["images", "annotations"]
                for subdir in subdirs:
                    subdir_path = dir_path / subdir
                    if not subdir_path.exists():
                        self.issues.append(f"缺少子目录: {dir_name}/{subdir}")
    
    def check_image_files(self):
        """检查图像文件"""
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        image_stats = {}
        
        for split in ["train", "val", "test"]:
            split_dir = self.dataset_dir / split / "images"
            if not split_dir.exists():
                continue
            
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(split_dir.rglob(f"*{ext}")))
            
            image_stats[split] = {
                "count": len(image_files),
                "extensions": {},
                "sizes": []
            }
            
            # 检查每个图像
            for img_file in image_files:
                try:
                    with Image.open(img_file) as img:
                        width, height = img.size
                        image_stats[split]["sizes"].append((width, height))
                        
                        # 检查图像模式
                        if img.mode != 'RGB':
                            self.issues.append(f"非RGB图像: {img_file}")
                        
                        # 检查图像质量
                        img_array = np.array(img)
                        if np.min(img_array) == np.max(img_array):
                            self.issues.append(f"可能损坏的图像: {img_file}")
                
                except Exception as e:
                    self.issues.append(f"无法读取图像 {img_file}: {e}")
            
            # 统计尺寸分布
            if image_stats[split]["sizes"]:
                sizes = np.array(image_stats[split]["sizes"])
                avg_size = sizes.mean(axis=0).astype(int)
                std_size = sizes.std(axis=0).astype(int)
                
                image_stats[split]["avg_size"] = avg_size.tolist()
                image_stats[split]["std_size"] = std_size.tolist()
        
        self.stats["images"] = image_stats
    
    def check_annotation_files(self):
        """检查标注文件"""
        annotation_stats = {}
        
        for split in ["train", "val", "test"]:
            ann_dir = self.dataset_dir / split / "annotations"
            if not ann_dir.exists():
                continue
            
            json_files = list(ann_dir.rglob("*.json"))
            annotation_stats[split] = {
                "count": len(json_files),
                "valid_count": 0,
                "format_errors": 0,
                "missing_fields": 0
            }
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        annotation = json.load(f)
                    
                    # 检查必要字段
                    required_fields = ["image_path", "operation", "elements"]
                    missing = [field for field in required_fields if field not in annotation]
                    
                    if missing:
                        annotation_stats[split]["missing_fields"] += 1
                        self.issues.append(f"标注缺少字段 {missing}: {json_file}")
                    else:
                        annotation_stats[split]["valid_count"] += 1
                        
                        # 检查操作类型
                        operation = annotation["operation"]
                        if "action" not in operation:
                            self.issues.append(f"操作缺少action字段: {json_file}")
                        
                        # 检查元素边界
                        for element in annotation.get("elements", []):
                            bounds = element.get("bounds", [])
                            if len(bounds) != 4:
                                self.issues.append(f"元素边界格式错误: {json_file}")
                
                except json.JSONDecodeError as e:
                    annotation_stats[split]["format_errors"] += 1
                    self.issues.append(f"JSON格式错误 {json_file}: {e}")
                except Exception as e:
                    self.issues.append(f"读取标注错误 {json_file}: {e}")
        
        self.stats["annotations"] = annotation_stats
    
    def check_class_balance(self):
        """检查类别平衡"""
        action_counts = {}
        
        for split in ["train", "val", "test"]:
            ann_dir = self.dataset_dir / split / "annotations"
            if not ann_dir.exists():
                continue
            
            json_files = list(ann_dir.rglob("*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        annotation = json.load(f)
                    
                    action = annotation.get("operation", {}).get("action", "unknown")
                    action_counts[action] = action_counts.get(action, 0) + 1
                
                except:
                    continue
        
        # 分析类别分布
        if action_counts:
            total_samples = sum(action_counts.values())
            avg_samples = total_samples / len(action_counts)
            
            imbalance_issues = []
            for action, count in action_counts.items():
                percentage = count / total_samples * 100
                if percentage < 1.0:  # 少于1%
                    imbalance_issues.append(f"类别 '{action}' 样本过少: {count} ({percentage:.1f}%)")
                elif percentage > 20.0:  # 多于20%
                    imbalance_issues.append(f"类别 '{action}' 样本过多: {count} ({percentage:.1f}%)")
            
            self.issues.extend(imbalance_issues)
            self.stats["class_distribution"] = {
                "total_classes": len(action_counts),
                "total_samples": total_samples,
                "avg_samples_per_class": avg_samples,
                "min_samples": min(action_counts.values()),
                "max_samples": max(action_counts.values()),
                "distribution": action_counts
            }
    
    def check_image_quality(self):
        """检查图像质量"""
        image_dir = self.dataset_dir / "train" / "images"
        if not image_dir.exists():
            return
        
        low_quality_count = 0
        for img_file in image_dir.rglob("*"):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                try:
                    with Image.open(img_file) as img:
                        # 检查图像分辨率
                        width, height = img.size
                        if width < 640 or height < 480:
                            self.issues.append(f"图像分辨率过低 {width}x{height}: {img_file}")
                        
                        # 检查是否为纯色图像
                        img_array = np.array(img)
                        if len(np.unique(img_array)) < 10:  # 唯一像素值太少
                            self.issues.append(f"图像可能为纯色或质量过低: {img_file}")
                
                except Exception as e:
                    self.issues.append(f"图像质量检查失败 {img_file}: {e}")
    
    def check_annotation_quality(self):
        """检查标注质量"""
        ann_dir = self.dataset_dir / "train" / "annotations"
        if not ann_dir.exists():
            return
        
        for json_file in ann_dir.rglob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    annotation = json.load(f)
                
                # 检查边界框是否在图像范围内
                elements = annotation.get("elements", [])
                for element in elements:
                    bounds = element.get("bounds", [])
                    if len(bounds) == 4:
                        x1, y1, x2, y2 = bounds
                        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                            self.issues.append(f"边界框坐标为负数: {json_file}")
                        if x1 > x2 or y1 > y2:
                            self.issues.append(f"边界框坐标错误 (x1 > x2 或 y1 > y2): {json_file}")
            
            except Exception as e:
                self.issues.append(f"标注质量检查失败 {json_file}: {e}")
    
    def generate_report(self, output_file: str = "dataset_validation_report.html"):
        """生成验证报告"""
        from datetime import datetime
        
        report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>数据集验证报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .issue {{ color: #d00; }}
                .warning {{ color: #f60; }}
                .success {{ color: #0a0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>数据集验证报告</h1>
                <p>数据集路径: {self.dataset_dir}</p>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # 问题汇总
        if self.issues:
            report += f"""
            <div class="section">
                <h2 class="issue">发现的问题 ({len(self.issues)} 个)</h2>
                <ul>
            """
            for issue in self.issues:
                report += f"<li>{issue}</li>"
            report += "</ul></div>"
        else:
            report += '<div class="section"><h2 class="success">✓ 未发现问题</h2></div>'
        
        # 统计信息
        if self.stats:
            report += '<div class="section"><h2>数据集统计</h2>'
            
            for section, data in self.stats.items():
                report += f"<h3>{section}</h3>"
                report += "<pre>" + json.dumps(data, indent=2, ensure_ascii=False) + "</pre>"
            
            report += "</div>"
        
        report += """
        </body>
        </html>
        """
        
        output_path = Path(output_file)
        output_path.write_text(report, encoding='utf-8')
        print(f"报告已生成: {output_path.absolute()}")