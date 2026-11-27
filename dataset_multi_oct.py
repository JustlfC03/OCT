# The OCT configuration uses five retinal layers plus background; excluded mask regions map to background.
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from data_config import window_starts
from Preprocess import load_image


def random_rot_flip(image, label):
    k = np.random.choice([0, 1, 2, 3])
    image, label = np.rot90(image, k), np.rot90(label, k)
    axis = np.random.randint(0, 2)
    return np.flip(image, axis=axis).copy(), np.flip(label, axis=axis).copy()


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    return (ndimage.rotate(image, angle, order=0, reshape=False),
            ndimage.rotate(label, angle, order=0, reshape=False))


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        h, w = image.shape
        if (h, w) != tuple(self.output_size):
            image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w), order=3)
            label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)
        return to_tensor(image, label)


def to_tensor(image, label):
    return {'image': torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0),
            'label': torch.from_numpy(np.ascontiguousarray(label, dtype=np.int64))}


def read_mask(mask_path):
    try:
        with Image.open(mask_path) as mask:
            label = np.array(mask)
    except OSError as exc:
        raise OSError(f"Cannot read mask: {mask_path}") from exc
    # Preserve palette indices; do not convert RGB masks to grayscale and alter class values.
    if label.ndim != 2:
        raise ValueError(f"Masks must be single-channel grayscale or palette-indexed images: {mask_path}")
    return label


def map_label(label, mapping, mask_path=''):
    unknown = set(np.unique(label).tolist()) - set(mapping)
    if unknown:
        raise ValueError(f"Mask contains unconfigured pixel values {sorted(unknown)}: {mask_path}")
    result = np.zeros(label.shape, dtype=np.int64)
    for pixel, class_index in mapping.items():
        result[label == pixel] = class_index
    return result


def read_samples(dataset_config, split, default_mapping):
    samples = []
    for ds_name, config in dataset_config.items():
        base_dir = Path(config['base_dir'])
        list_path = Path(config['list_dir']) / (split + '.txt')
        # Fail on missing lists to avoid silently omitting a dataset.
        lines = list_path.read_text(encoding='utf-8-sig').splitlines()
        metadata = {}
        if config.get('metadata'):
            with open(config['metadata'], encoding='utf-8-sig', newline='') as stream:
                metadata = {row['case_name']: row for row in csv.DictReader(stream)}
        seen = set()
        for line in lines:
            name = line.strip().replace('\\', '/')
            if not name:
                continue
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError(f'Sample lists must use paths relative to the image directory: {name}')
            if name in seen:
                raise ValueError(f"Duplicate sample in list {name}: {list_path}")
            seen.add(name)
            image_path = base_dir / split / config.get('image_dir', 'img') / name
            mask_name = str(Path(name).with_suffix(config.get('mask_suffix', '.png')))
            mask_path = base_dir / split / config.get('mask_dir', 'mask') / mask_name
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"Image/mask pair not found: {image_path} | {mask_path}")
            info = metadata.get(name, {})
            samples.append({'slice_name': name, 'image_path': str(image_path), 'mask_path': str(mask_path),
                            'dataset': ds_name, 'disease': info.get('disease') or config.get('disease', 'unknown'),
                            'mapping': config.get('label_map', default_mapping)})
    if not samples:
        raise ValueError(f'{split} split is empty')
    return samples


class Multi_OCT_Dataset(Dataset):
    def __init__(self, dataset_config, split, crop_size=(192, 192), stride=(96, 96),
                 args=None, transform=None, label_map=None, preprocess=None, preprocess_config=None):
        self.transform = transform
        self.split = split
        self.crop_size = tuple(crop_size)
        self.stride = tuple(stride)
        self.preprocess = getattr(args, 'preprocess', True) if preprocess is None else preprocess
        self.preprocess_config = preprocess_config
        if label_map is None:
            raise ValueError('Pass label_map explicitly to avoid mixing label definitions across datasets')
        self.sample_list = read_samples(dataset_config, split, label_map)
        self.windows = []
        for sample_index, sample in enumerate(self.sample_list):
            with Image.open(sample['image_path']) as image:
                w, h = image.size
            for y1 in window_starts(h, crop_size[0], stride[0]):
                for x1 in window_starts(w, crop_size[1], stride[1]):
                    self.windows.append((sample_index, y1, x1))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sample_index, y1, x1 = self.windows[idx]
        info = self.sample_list[sample_index]
        image = load_image(info['image_path'], self.preprocess, self.preprocess_config)
        label = read_mask(info['mask_path'])
        if image.shape != label.shape:
            raise ValueError(f"Image and mask dimensions differ: {info['slice_name']}")
        label = map_label(label, info['mapping'], info['mask_path'])
        h, w = image.shape
        th, tw = self.crop_size
        if h < th or w < tw:
            padding = ((0, max(0, th - h)), (0, max(0, tw - w)))
            image = np.pad(image, padding, mode='constant')
            # Pad with background class indices to avoid mapping GOALS pixel value 0 to foreground.
            label = np.pad(label, padding, mode='constant')
        image, label = image[y1:y1 + th, x1:x1 + tw], label[y1:y1 + th, x1:x1 + tw]
        if self.transform and self.split == 'train':
            sample = self.transform({'image': image, 'label': label})
        else:
            sample = to_tensor(image, label)
        sample.update(case_name=info['slice_name'], dataset=info['dataset'], disease=info['disease'])
        return sample
