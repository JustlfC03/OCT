import argparse
from pathlib import Path


def generate_lists(base_dir, list_dir):
    base_dir, list_dir = Path(base_dir), Path(list_dir)
    list_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        image_dir, mask_dir = base_dir / split / 'img', base_dir / split / 'mask'
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        names = []
        for image in sorted(image_dir.rglob('*')):
            if image.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}:
                continue
            name = image.relative_to(image_dir)
            if not (mask_dir / name.with_suffix('.png')).is_file():
                raise FileNotFoundError(mask_dir / name.with_suffix('.png'))
            names.append(name.as_posix())
        target = list_dir / f'{split}.txt'
        if target.exists():
            raise FileExistsError(f'List already exists; choose a new directory: {target}')
        target.write_text(''.join(n + '\n' for n in names), encoding='utf-8')
        print(f'{split}: {len(names)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate lists from existing image/mask splits without repartitioning')
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--list_dir', required=True)
    args = parser.parse_args()
    generate_lists(args.base_dir, args.list_dir)
