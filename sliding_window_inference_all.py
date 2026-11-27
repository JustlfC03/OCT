import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from data_config import load_config
from inference import predict_image
from model_utils import build_model, load_checkpoint, select_device
from Preprocess import load_image


def predict_sliding_window(args):
    config = load_config(args.config)
    device = select_device(args.device)
    size = config.get('img_size', [192, 192])
    model = build_model(config, size, device)
    load_checkpoint(model, args.model_path, device)
    model.eval()
    source = Path(args.input_dir).resolve()
    output = Path(args.output_dir).resolve()
    if source == output or source in output.parents:
        raise ValueError('output_dir must be outside input_dir to avoid processing generated outputs as inputs')
    suffixes = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    files = sorted(p for p in source.rglob('*') if p.suffix.lower() in suffixes)
    if not files:
        raise FileNotFoundError(f'No images found: {source}')
    destinations = set()
    for path in files:
        target = output / path.relative_to(source).with_suffix('.png')
        if target in destinations:
            raise ValueError(f'Images in the same directory share a stem but have different extensions: {path}')
        destinations.add(target)
        if target.exists():
            raise FileExistsError(f'Output already exists; choose a new output directory: {target}')
    inverse = np.zeros(config['num_classes'], dtype=np.uint8)
    for class_id in range(config['num_classes']):
        pixels = [pixel for pixel, mapped_class in config['label_map'].items() if mapped_class == class_id]
        # Background aliases are saved as 0; foreground grayscale values remain unique.
        if class_id == 0 and 0 in pixels:
            inverse[class_id] = 0
        elif len(pixels) == 1:
            inverse[class_id] = pixels[0]
        elif not args.index_mask:
            raise ValueError('Multiple grayscale values map to one class; use --index_mask to avoid ambiguous reverse encoding')
    for path in files:
        image = load_image(path, config.get('preprocess', True), config.get('preprocessing'))
        prediction = predict_image(model, image, size, device, args.sw_batch_size)
        target = output / path.relative_to(source).with_suffix('.png')
        target.parent.mkdir(parents=True, exist_ok=True)
        mask = prediction if args.index_mask else inverse[prediction]
        Image.fromarray(mask).save(target)
        print(target)


def main():
    parser = argparse.ArgumentParser(description='Run recursive batch sliding-window inference and save lossless PNG masks')
    parser.add_argument('--config', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device')
    parser.add_argument('--sw_batch_size', type=int, default=4)
    parser.add_argument('--index_mask', action='store_true', help='Save class indices instead of configured grayscale values')
    predict_sliding_window(parser.parse_args())


if __name__ == '__main__':
    main()
