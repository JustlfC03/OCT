"""OCT preprocessing with the original effective filter order and parameters."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def read_grayscale(image_path):
    # Use imdecode for Unicode paths on Windows while retaining OpenCV grayscale decoding.
    try:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except OSError as exc:
        raise OSError(f"Cannot read image: {image_path}") from exc
    if image is None:
        raise ValueError(f"Cannot decode image: {image_path}")
    return image


def optimal_oct_preprocess(image_path_or_array, config=None):
    config = config or {}
    if isinstance(image_path_or_array, (str, Path)):
        img = read_grayscale(image_path_or_array)
    else:
        img = np.asarray(image_path_or_array)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8 or img.ndim != 2:
        raise ValueError("Preprocessing requires a uint8 grayscale or BGR image")

    # The original code overwrote medianBlur and NLM results; only bilateral filtering affected the output.
    filtered = cv2.bilateralFilter(img, config.get('bilateral_d', 5),
                                   config.get('sigma_color', 25), config.get('sigma_space', 25))
    kernel = tuple(config.get('gaussian_kernel', [5, 5]))
    blurred = cv2.GaussianBlur(filtered, kernel, config.get('gaussian_sigma', 1.0))
    amount = config.get('sharpen_amount', 0.3)
    sharpened = cv2.addWeighted(filtered, 1.0 + amount, blurred, -amount, 0)
    plow, phigh = np.percentile(sharpened, config.get('percentiles', [1, 99]))
    if phigh <= plow:
        return np.zeros_like(sharpened, dtype=np.float32)
    # Clip and linearly scale to [0, 1].
    return ((np.clip(sharpened, plow, phigh) - plow) / (phigh - plow)).astype(np.float32)


def load_image(image_path, preprocess=True, config=None):
    if preprocess:
        return optimal_oct_preprocess(image_path, config)
    # Preserve the original sliding-window loader's [0, 255] scale when preprocessing is disabled.
    return read_grayscale(image_path).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Preprocess a single OCT image and save it as a lossless PNG')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.suffix.lower() != '.png':
        parser.error('--output must have a .png extension')
    result = np.rint(optimal_oct_preprocess(args.input) * 255).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode('.png', result)[1].tofile(output)


if __name__ == '__main__':
    main()
