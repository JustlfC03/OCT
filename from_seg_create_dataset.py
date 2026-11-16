"""Organize image/mask pairs from a CSV with headers; preserve existing splits or split by group."""
import argparse
import csv
import random
from pathlib import Path

from PIL import Image


def assign_splits(rows, seed=42, ratios=(0.8, 0.1, 0.1), group_column='patient_id'):
    if all(row.get('split') in {'train', 'val', 'test'} for row in rows):
        # Check for group leakage when anonymous patient or volume identifiers are available.
        groups = {}
        for row in rows:
            group = row.get(group_column)
            if group:
                if group in groups and groups[group] != row['split']:
                    raise ValueError(f'The same group appears in multiple splits: {group}')
                groups[group] = row['split']
        return rows
    if any(row.get('split') for row in rows):
        raise ValueError('split must be train/val/test for every row, or empty for every row')
    if any(not row.get(group_column) for row in rows):
        raise ValueError(f'Random splitting requires {group_column} to keep each patient or volume within one split')
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError('ratios must contain three nonnegative values summing to 1')
    groups = sorted({row[group_column] for row in rows})
    random.Random(seed).shuffle(groups)
    n_train = int(len(groups) * ratios[0])
    n_val = int(len(groups) * ratios[1])
    assignments = dict.fromkeys(groups[:n_train], 'train')
    assignments.update(dict.fromkeys(groups[n_train:n_train + n_val], 'val'))
    assignments.update(dict.fromkeys(groups[n_train + n_val:], 'test'))
    for row in rows:
        row['split'] = assignments[row[group_column]]
    return rows


def center_crop_image(img, reduce_w, reduce_h):
    width, height = img.size
    if reduce_w < 0 or reduce_h < 0 or reduce_w >= width or reduce_h >= height:
        raise ValueError('Total crop width and height must be nonnegative and smaller than the image dimensions')
    left, top = reduce_w // 2, reduce_h // 2
    return img.crop((left, top, width - (reduce_w - left), height - (reduce_h - top)))


def prepare_dataset(csv_path, input_dir, output_dir, seed=42, ratios=(0.8, 0.1, 0.1),
                    group_column='patient_id', crop_w=0, crop_h=0):
    root, output = Path(input_dir), Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('The output directory is not empty; use a new directory')
    with open(csv_path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'case_name', 'image', 'mask'} <= set(reader.fieldnames or []):
            raise ValueError('CSV must contain case_name,image,mask columns; add explicit headers to legacy four-column files')
        rows = list(reader)
    if not rows:
        raise ValueError('CSV contains no samples')
    rows = assign_splits(rows, seed, ratios, group_column)
    seen = set()
    # Check pairing and names before writing any data.
    for row in rows:
        name = Path(row['case_name']).with_suffix('.png').as_posix()
        if Path(name).is_absolute() or '..' in Path(name).parts or name in seen:
            raise ValueError(f'case_name is duplicated or is not a safe relative path: {name}')
        seen.add(name)
        row['case_name'] = name
        for key in ('image', 'mask'):
            if not (root / row[key]).is_file():
                raise FileNotFoundError(root / row[key])
    for split in ('train', 'val', 'test'):
        for folder in ('img', 'mask'):
            (output / split / folder).mkdir(parents=True, exist_ok=True)
    (output / 'list').mkdir(exist_ok=True)
    metadata = []
    for row in rows:
        try:
            with Image.open(root / row['image']) as image, Image.open(root / row['mask']) as mask:
                if image.size != mask.size:
                    raise ValueError(f"Image and mask dimensions differ: {row['case_name']}")
                if mask.mode not in ('L', 'P'):
                    raise ValueError(f"Masks must be 8-bit grayscale or palette-indexed images: {row['case_name']}")
                # Preserve image colors and mask indices; re-saving a decoded JPEG does not recover lossless source data.
                for img, folder in ((image, 'img'), (mask, 'mask')):
                    target = output / row['split'] / folder / row['case_name']
                    target.parent.mkdir(parents=True, exist_ok=True)
                    center_crop_image(img, crop_w, crop_h).save(target, format='PNG')
        except OSError as exc:
            raise OSError(f"Failed to prepare sample: {row['case_name']}") from exc
        metadata.append(row)
    for split in ('train', 'val', 'test'):
        names = [row['case_name'] for row in metadata if row['split'] == split]
        (output / 'list' / f'{split}.txt').write_text(''.join(n + '\n' for n in names), encoding='utf-8')
    with (output / 'metadata.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Organize img/mask/list directories and preserve provenance metadata')
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--group_column', default='patient_id')
    parser.add_argument('--ratios', type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--crop_w', type=int, default=0, help='Total width removed from the left and right sides')
    parser.add_argument('--crop_h', type=int, default=0, help='Total height removed from the top and bottom sides')
    args = parser.parse_args()
    rows = prepare_dataset(**vars(args))
    print(f'Prepared {len(rows)} sample pairs in {args.output_dir}')


if __name__ == '__main__':
    main()
