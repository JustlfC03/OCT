import argparse
import csv
from pathlib import Path

import numpy as np

from data_config import load_config
from dataset_multi_oct import map_label, read_mask, read_samples
from inference import predict_image
from metrics import confusion_matrix, segmentation_scores
from model_utils import build_model, load_checkpoint, select_device
from Preprocess import load_image


def evaluate_samples(model, samples, config, img_size, device, sw_batch_size=4):
    model.eval()
    matrices = {}
    counts = {}
    per_image = []
    n = config['num_classes']
    for info in samples:
        image = load_image(info['image_path'], config.get('preprocess', True), config.get('preprocessing'))
        label = map_label(read_mask(info['mask_path']), info['mapping'], info['mask_path'])
        prediction = predict_image(model, image, img_size, device, sw_batch_size)
        matrix = confusion_matrix(prediction, label, n)
        for group_type, group in [('overall', 'all'), ('dataset', info['dataset']),
                                  ('disease', info['disease'])]:
            key = (group_type, group)
            matrices.setdefault(key, np.zeros((n, n), dtype=np.int64))
            matrices[key] += matrix
            counts[key] = counts.get(key, 0) + 1
        for row in segmentation_scores(matrix, config['retinal_layer']):
            per_image.append(dict(case_name=info['slice_name'], dataset=info['dataset'],
                                  disease=info['disease'], **row))
    summary = []
    for (group_type, group), matrix in matrices.items():
        for row in segmentation_scores(matrix, config['retinal_layer']):
            summary.append(dict(group_type=group_type, group=group, images=counts[(group_type, group)], **row))
    return summary, per_image


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Evaluate whole-image sliding-window Dice / IoU, grouped by dataset and disease')
    parser.add_argument('--config', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--output_dir', default='outputs/evaluation')
    parser.add_argument('--device')
    parser.add_argument('--sw_batch_size', type=int, default=4)
    args = parser.parse_args()
    config = load_config(args.config)
    size = config.get('img_size', [192, 192])
    device = select_device(args.device)
    model = build_model(config, size, device)
    load_checkpoint(model, args.model_path, device)
    samples = read_samples(config['datasets'], args.split, config['label_map'])
    summary, per_image = evaluate_samples(model, samples, config, size, device, args.sw_batch_size)
    write_csv(Path(args.output_dir) / 'summary.csv', summary)
    write_csv(Path(args.output_dir) / 'per_image.csv', per_image)
    print(f'Evaluated {len(samples)} images; results saved to {args.output_dir}')


if __name__ == '__main__':
    main()
