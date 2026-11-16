import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

from data_config import load_config, window_starts
from dataset_multi_oct import Multi_OCT_Dataset, map_label, read_mask, read_samples
from evaluate import evaluate_samples
from from_seg_create_dataset import assign_splits, prepare_dataset
from metrics import confusion_matrix, segmentation_scores
from model_utils import build_model, load_checkpoint
from Preprocess import optimal_oct_preprocess
from trainer import trainer_multi_oct


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.mapping = {0: 0, 40: 1}
        self.datasets = {'fixture': {'base_dir': str(self.root), 'list_dir': str(self.root / 'list'),
                                     'disease': 'example'}}
        (self.root / 'list').mkdir()
        for split in ('train', 'val', 'test'):
            for folder in ('img', 'mask'):
                (self.root / split / folder).mkdir(parents=True)
            image = np.arange(21 * 29, dtype=np.uint8).reshape(21, 29)
            label = np.where(image > 128, 40, 0).astype(np.uint8)
            Image.fromarray(image).save(self.root / split / 'img' / 'case.jpg')
            Image.fromarray(label).save(self.root / split / 'mask' / 'case.png')
            (self.root / 'list' / f'{split}.txt').write_text('case.jpg\n', encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, split='val', **kwargs):
        return Multi_OCT_Dataset(self.datasets, split, crop_size=(16, 16), stride=(8, 8),
                                 label_map=self.mapping, **kwargs)

    def test_windows_cover_edges_and_small_images(self):
        for length in (5, 16, 21, 29, 40):
            covered = np.zeros(max(length, 16), dtype=bool)
            for start in window_starts(length, 16, 8):
                covered[start:start + 16] = True
            self.assertTrue(covered[:length].all())
        with self.assertRaises(ValueError):
            window_starts(30, 16, 17)

    def test_jpeg_png_pair_and_no_preprocess(self):
        dataset = self.dataset(preprocess=False)
        sample = dataset[0]
        self.assertEqual(sample['image'].shape, (1, 16, 16))
        self.assertEqual(sample['label'].dtype, torch.int64)
        self.assertGreater(sample['image'].max(), 1)
        self.assertEqual(set(sample['label'].unique().tolist()), {0, 1})
        self.assertEqual(sample['disease'], 'example')

    def test_small_images_pad_in_class_space(self):
        self.datasets['fixture']['label_map'] = {0: 1, 40: 0}
        dataset = Multi_OCT_Dataset(self.datasets, 'val', crop_size=(32, 32), stride=(16, 16),
                                    label_map=self.mapping, preprocess=False)
        self.assertEqual(dataset[0]['label'][31, 31].item(), 0)

    def test_validation_does_not_augment(self):
        def fail(sample):
            raise AssertionError('validation must not augment')
        dataset = self.dataset(transform=fail)
        self.assertTrue(torch.equal(dataset[0]['image'], dataset[0]['image']))

    def test_unknown_label_and_rgb_mask_rejected(self):
        with self.assertRaisesRegex(ValueError, '255'):
            map_label(np.array([[255]], dtype=np.uint8), self.mapping)
        path = self.root / 'rgb.png'
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(path)
        with self.assertRaises(ValueError):
            read_mask(path)

    def test_palette_mask_keeps_indices(self):
        path = self.root / 'palette.png'
        mask = Image.fromarray(np.array([[0, 40]], dtype=np.uint8)).convert('P')
        mask.save(path)
        self.assertEqual(read_mask(path).tolist(), [[0, 40]])

    def test_missing_list_and_mask_reported(self):
        (self.root / 'list' / 'val.txt').unlink()
        with self.assertRaises(FileNotFoundError):
            self.dataset()
        (self.root / 'test' / 'mask' / 'case.png').unlink()
        with self.assertRaises(FileNotFoundError):
            read_samples(self.datasets, 'test', self.mapping)

    def test_list_requires_relative_paths_and_pair_shape(self):
        (self.root / 'list' / 'test.txt').write_text('../case.jpg\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            read_samples(self.datasets, 'test', self.mapping)
        Image.fromarray(np.zeros((3, 3), dtype=np.uint8)).save(self.root / 'val/mask/case.png')
        with self.assertRaisesRegex(ValueError, 'dimensions differ'):
            self.dataset()[0]

    def test_preprocessing_matches_effective_original_operations(self):
        image = np.random.default_rng(42).integers(0, 256, (31, 35), dtype=np.uint8)
        filtered = cv2.bilateralFilter(image, 5, 25, 25)
        blurred = cv2.GaussianBlur(filtered, (5, 5), 1.0)
        sharpened = cv2.addWeighted(filtered, 1.3, blurred, -0.3, 0)
        lo, hi = np.percentile(sharpened, (1, 99))
        expected = ((np.clip(sharpened, lo, hi) - lo) / (hi - lo)).astype(np.float32)
        np.testing.assert_array_equal(optimal_oct_preprocess(image), expected)
        self.assertTrue(np.isfinite(optimal_oct_preprocess(np.zeros((8, 8), dtype=np.uint8))).all())

    def test_config_resolves_paths_and_rejects_seventh_class(self):
        path = self.root / 'config.json'
        config = {'num_classes': 2, 'label_map': {'0': 0, '40': 1},
                  'datasets': {'fixture': {'base_dir': '.', 'list_dir': 'list'}}}
        path.write_text(json.dumps(config), encoding='utf-8')
        result = load_config(path)
        self.assertEqual(result['datasets']['fixture']['base_dir'], str(self.root.resolve()))
        config['label_map']['255'] = 2
        path.write_text(json.dumps(config), encoding='utf-8')
        with self.assertRaises(ValueError):
            load_config(path)

    def test_metrics_count_false_positives_and_absent_classes(self):
        matrix = confusion_matrix(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]), 3)
        rows = segmentation_scores(matrix, [1, 2])
        self.assertAlmostEqual(rows[0]['dice'], 0.5)
        self.assertAlmostEqual(rows[0]['iou'], 1 / 3)
        self.assertTrue(np.isnan(rows[1]['dice']))
        false_positive = confusion_matrix(np.array([1]), np.array([0]), 2)
        self.assertEqual(segmentation_scores(false_positive, [1])[0]['dice'], 0)

    def test_group_split_and_existing_split(self):
        rows = [{'patient_id': f'p{i // 3}', 'case_name': str(i)} for i in range(60)]
        split = assign_splits(rows, seed=42)
        for patient in {row['patient_id'] for row in split}:
            self.assertEqual(len({r['split'] for r in split if r['patient_id'] == patient}), 1)
        with self.assertRaises(ValueError):
            assign_splits([{'patient_id': 'p', 'split': 'train'}, {'patient_id': 'p', 'split': 'test'}])
        with self.assertRaises(ValueError):
            assign_splits([{'case_name': 'a'}])

    def test_prepare_lossless_pair_and_metadata(self):
        csv_path = self.root / 'source.csv'
        csv_path.write_text('case_name,image,mask,split,patient_id,disease\n'
                            'renamed.png,train/img/case.jpg,train/mask/case.png,train,p1,example\n', encoding='utf-8')
        out = self.root / 'prepared'
        prepare_dataset(csv_path, self.root, out)
        np.testing.assert_array_equal(read_mask(out / 'train/mask/renamed.png'),
                                       read_mask(self.root / 'train/mask/case.png'))
        self.assertEqual((out / 'list/train.txt').read_text().strip(), 'renamed.png')
        self.assertIn('example', (out / 'metadata.csv').read_text())

    def test_whole_image_evaluation_and_training_loop(self):
        model = torch.nn.Conv2d(1, 2, 1)
        config = {'num_classes': 2, 'label_map': self.mapping, 'retinal_layer': [1],
                  'datasets': self.datasets, 'preprocess': True, 'img_size': [16, 16], 'stride': [16, 16]}
        samples = read_samples(self.datasets, 'val', self.mapping)
        summary, per_image = evaluate_samples(model, samples, config, [16, 16], torch.device('cpu'), 2)
        self.assertEqual({row['group_type'] for row in summary}, {'overall', 'dataset', 'disease'})
        self.assertEqual(len(per_image), 1)
        args = SimpleNamespace(config_data=config, seed=42, batch_size=2, num_workers=0,
                               device=torch.device('cpu'), base_lr=0.001, max_epochs=1, sw_batch_size=2)
        out = self.root / 'training'
        trainer_multi_oct(args, model, out)
        self.assertTrue((out / 'best_model.pth').is_file())
        loaded = torch.nn.Conv2d(1, 2, 1)
        load_checkpoint(loaded, out / 'best_model.pth', torch.device('cpu'))
        for a, b in zip(model.parameters(), loaded.parameters()):
            self.assertTrue(torch.equal(a, b))


class ModelTests(unittest.TestCase):
    def test_current_r50_supports_non_square_input(self):
        # Check the actual network forward pass with the retained non-square input support.
        torch.set_num_threads(2)
        config = {'num_classes': 6, 'vit_name': 'R50-ViT-B_16', 'n_skip': 3}
        model = build_model(config, [32, 48], torch.device('cpu')).eval()
        with torch.no_grad():
            result = model(torch.zeros(1, 1, 32, 48))
        self.assertEqual(tuple(result.shape), (1, 6, 32, 48))


if __name__ == '__main__':
    unittest.main()
