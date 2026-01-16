"""
训练流水线（PyTorch 优化版）

改进点：
- 统一使用 PyTorch（移除对 TensorFlow/Keras 的直接依赖）
- 自动检测 GPU 并使用混合精度（AMP）可选
- 使用 DataLoader 进行批次迭代
- 支持断点续训与周期性保存 checkpoint
- 保存训练历史到 JSON，保存最佳模型 state_dict
"""
import os
import json
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ..core.config import Config
from ..models.cnn_model import EnhancedCNN
from ..data.data_processor import DataProcessor


class TrainingPipeline:
    """训练流水线类（PyTorch）"""

    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.model: Optional[torch.nn.Module] = None
        self.optimizer = None
        self.criterion = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = None  # 用于 AMP
        self.history: Dict[str, List[float]] = {}

    def setup_model(self, model_type: str = "EnhancedCNN", num_actions: int = 14, lr: float = 1e-3, use_amp: bool = True):
        """设置模型与优化器

        Args:
            model_type: 模型名称
            num_actions: 输出动作维度
            lr: 学习率
            use_amp: 是否启用混合精度（仅在 CUDA 可用时）
        """
        if model_type != "EnhancedCNN":
            raise ValueError(f"不支持的模型类型: {model_type}")

        self.model = EnhancedCNN(num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        # AMP 混合精度
        if use_amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        print(f"模型设置完成: {model_type} -> device: {self.device}, amp: {self.scaler is not None}")

    def prepare_dataset(self, dataset_path: str, validation_split: float = 0.2) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """加载并准备数据集，返回 (X_train,y_train),(X_val,y_val)
        假设数据处理器返回的样本格式与之前兼容：sample['screen_input'] 和 sample['action_output']
        """
        dataset = self.data_processor.load_training_dataset(dataset_path)
        samples = dataset.get('samples', [])

        if not samples:
            raise ValueError("数据集中没有样本")

        images = []
        actions = []

        for sample in samples:
            img_arr = np.array(sample['screen_input'])
            # 如果屏幕图像有批次维度，取第一个
            if img_arr.ndim == 4:
                img = img_arr[0]
            else:
                img = img_arr

            images.append(img)

            action_vec = self._action_to_vector(sample.get('action_output', {}))
            actions.append(action_vec)

        X = np.array(images, dtype=np.float32)
        y = np.array(actions, dtype=np.float32)

        # 切分
        idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:idx], X[idx:]
        y_train, y_val = y[:idx], y[idx:]

        print(f"准备数据集: {len(X)} 样本 -> 训练: {len(X_train)}, 验证: {len(X_val)}")
        return (X_train, y_train), (X_val, y_val)

    def _action_to_vector(self, action_data: Dict[str, Any]) -> np.ndarray:
        """将动作数据转换为向量（简单且健壮）"""
        coords = action_data.get('coordinates', [0, 0])
        action_type = action_data.get('action_type', 'click')

        action_types = ['left_click', 'left_double_click', 'right_click', 'right_double_click', 'drag', 'keypress', 'type_text', 'move']
        type_encoding = [1.0 if action_type == t else 0.0 for t in action_types]

        # 使用配置的 SCREEN_RESIZE 做归一化基准
        screen_size = getattr(self.config, 'SCREEN_RESIZE', (640, 480))
        max_coord = max(screen_size) if screen_size else 1.0
        norm_coords = [coords[0] / max_coord, coords[1] / max_coord]

        vec = np.array(norm_coords + type_encoding, dtype=np.float32)
        return vec

    def _make_dataloader(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
        # 转换为 torch tensor，注意通道顺序可能需要调整到 (N,C,H,W)
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)

        # 如果图像没有通道维，尝试添加
        if X_t.dim() == 3:
            # 假设 (H,W,C) -> 转置为 (C,H,W)
            X_t = X_t.permute(0, 3, 1, 2)
        elif X_t.dim() == 4 and X_t.shape[1] in (1, 3):
            # 可能已经是 (N,C,H,W)
            pass
        else:
            # 尝试将最后一维视为通道
            try:
                X_t = X_t.permute(0, 3, 1, 2)
            except Exception:
                pass

        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_model(self,
                    train_data: Tuple[np.ndarray, np.ndarray],
                    val_data: Tuple[np.ndarray, np.ndarray],
                    epochs: int = 50,
                    batch_size: int = 32,
                    save_dir: Optional[str] = None,
                    resume_from: Optional[str] = None,
                    use_amp: bool = True,
                    verbose: int = 1) -> Dict[str, List[float]]:
        """训练模型（支持 checkpoint/resume 和 AMP）"""
        if self.model is None:
            raise ValueError("模型未设置，请先调用 setup_model")

        X_train, y_train = train_data
        X_val, y_val = val_data

        train_loader = self._make_dataloader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self._make_dataloader(X_val, y_val, batch_size, shuffle=False)

        start_epoch = 0
        best_val_loss = float('inf')

        # 恢复 checkpoint
        if resume_from and os.path.exists(resume_from):
            ckpt = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])
            self.optimizer.load_state_dict(ckpt['optim_state'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_loss = ckpt.get('best_val_loss', best_val_loss)
            if self.scaler and 'scaler_state' in ckpt:
                self.scaler.load_state_dict(ckpt['scaler_state'])
            print(f"已从 checkpoint 恢复: {resume_from}, 从 epoch {start_epoch} 开始")

        # 训练循环
        self.history = {'loss': [], 'val_loss': []}

        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            self.model.train()
            running_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()

                if use_amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        preds = outputs['actions']
                        loss = self.criterion(preds, batch_y)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_x)
                    preds = outputs['actions']
                    loss = self.criterion(preds, batch_y)
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * batch_x.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            # 验证
            val_loss = self._validate_model(val_loader)

            self.history['loss'].append(epoch_loss)
            self.history['val_loss'].append(val_loss)

            # 保存 checkpoint
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                ckpt_path = os.path.join(save_dir, f'ckpt_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optim_state': self.optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'scaler_state': self.scaler.state_dict() if self.scaler is not None else None
                }, ckpt_path)

                # 保存最佳模型副本
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(save_dir, 'best_model.pth')
                    torch.save(self.model.state_dict(), best_path)

            if verbose:
                print(f'Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - time: {time.time()-epoch_start:.1f}s')

        # 最终保存训练历史
        if save_dir:
            hist_path = os.path.join(save_dir, 'training_history.json')
            with open(hist_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)

        return self.history

    def _validate_model(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_x)
                preds = outputs['actions']
                loss = self.criterion(preds, batch_y)
                total += loss.item() * batch_x.size(0)

        return total / len(val_loader.dataset)

    def evaluate_model(self, test_data: Tuple[np.ndarray, np.ndarray], batch_size: int = 32) -> Dict[str, float]:
        """在测试集上评估模型并返回损失"""
        X_test, y_test = test_data
        loader = self._make_dataloader(X_test, y_test, batch_size, shuffle=False)
        val_loss = self._validate_model(loader)
        print(f"评估结果 - loss: {val_loss:.4f}")
        return {'loss': val_loss}

    def fine_tune_model(self, new_data: Tuple[np.ndarray, np.ndarray], epochs: int = 10, lr: float = 1e-4, batch_size: int = 32, save_dir: Optional[str] = None) -> Dict[str, List[float]]:
        """使用较低学习率继续训练（微调）"""
        if self.model is None:
            raise ValueError("模型未设置")

        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        history = self.train_model(new_data, new_data, epochs=epochs, batch_size=batch_size, save_dir=save_dir, use_amp=(self.scaler is not None))
        return history

    def cross_validate(self, dataset_path: str, k_folds: int = 5, epochs: int = 30, batch_size: int = 16, save_dir: Optional[str] = None) -> List[Dict[str, float]]:
        dataset = self.data_processor.load_training_dataset(dataset_path)
        samples = dataset.get('samples', [])

        if len(samples) < k_folds:
            raise ValueError("样本数量少于折数")

        np.random.shuffle(samples)
        fold_size = len(samples) // k_folds
        results = []

        for fold in range(k_folds):
            start = fold * fold_size
            end = start + fold_size if fold < k_folds - 1 else len(samples)
            val_samples = samples[start:end]
            train_samples = samples[:start] + samples[end:]

            def to_temp_path(prefix, s):
                p = f"{prefix}_fold_{fold}.json"
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump({'metadata': dataset.get('metadata', {}), 'samples': s}, f, ensure_ascii=False)
                return p

            train_path = to_temp_path('temp_train', train_samples)
            val_path = to_temp_path('temp_val', val_samples)

            self.setup_model('EnhancedCNN')
            train_data = self.prepare_dataset(train_path, validation_split=0.0)[0]
            val_data = self.prepare_dataset(val_path, validation_split=0.0)[0]

            fold_save = None
            if save_dir:
                fold_save = os.path.join(save_dir, f'fold_{fold}')

            self.train_model(train_data, val_data, epochs=epochs, batch_size=batch_size, save_dir=fold_save)
            res = self.evaluate_model(val_data)
            results.append(res)

            os.remove(train_path)
            os.remove(val_path)

        avg = {'loss': float(np.mean([r['loss'] for r in results]))}
        print(f"交叉验证平均损失: {avg['loss']:.4f}")
        return results
