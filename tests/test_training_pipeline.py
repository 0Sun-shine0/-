import os
import sys
import tempfile
import json
import numpy as np
import unittest

# ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai_operator.core.config import Config
from ai_operator.models.training_pipeline import TrainingPipeline


def create_small_dataset(path, n=16, img_size=(32, 32)):
    samples = []
    for i in range(n):
        img = (np.random.rand(img_size[0], img_size[1], 3) * 255).astype(np.uint8).tolist()
        coords = [int(np.random.randint(0, max(img_size))), int(np.random.randint(0, max(img_size)))]
        action_type = 'left_click' if np.random.rand() > 0.5 else 'right_click'
        samples.append({'screen_input': img, 'action_output': {'coordinates': coords, 'action_type': action_type}})

    dataset = {'metadata': {'sample_count': n}, 'samples': samples}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False)


class TrainingPipelineSmokeTest(unittest.TestCase):

    def test_train_one_epoch(self):
        tmpdir = tempfile.mkdtemp()
        ds_path = os.path.join(tmpdir, 'ds.json')
        create_small_dataset(ds_path, n=24, img_size=(32, 32))

        cfg = Config()
        tp = TrainingPipeline(cfg)

        train_data, val_data = tp.prepare_dataset(ds_path, validation_split=0.2)
        sample_y = train_data[1]
        num_actions = sample_y.shape[1]
        tp.setup_model('EnhancedCNN', num_actions=num_actions, lr=cfg.LEARNING_RATE, use_amp=False)

        hist = tp.train_model(train_data, val_data, epochs=1, batch_size=4, save_dir=tmpdir, use_amp=False)

        self.assertIn('loss', hist)
        self.assertIn('val_loss', hist)
        self.assertEqual(len(hist['loss']), 1)


if __name__ == '__main__':
    unittest.main()
