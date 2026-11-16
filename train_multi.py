import argparse
import random

import numpy as np
import torch

from data_config import load_config
from model_utils import build_model, load_checkpoint, select_device
from trainer import trainer_multi_oct


def main():
    parser = argparse.ArgumentParser(description='Train OCT TransUNet using a configuration file')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output_dir', default='outputs/train')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--sw_batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', help='For example: cpu, cuda, or cuda:0')
    initial = parser.add_mutually_exclusive_group(required=True)
    initial.add_argument('--pretrained', help='Pretrained ViT weights in NPZ format')
    initial.add_argument('--init_checkpoint', help='Initialize from existing OCT model weights with a new optimizer')
    initial.add_argument('--from_scratch', action='store_true', help='Explicitly use random initialization')
    args = parser.parse_args()
    if args.max_epochs <= 0 or args.batch_size <= 0:
        parser.error('max_epochs and batch_size must be positive')
    args.config_data = load_config(args.config)
    args.config_data.setdefault('img_size', [192, 192])
    args.config_data.setdefault('stride', [96, 96])
    args.device = select_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_model(args.config_data, args.config_data['img_size'], args.device)
    if args.init_checkpoint:
        load_checkpoint(model, args.init_checkpoint, args.device)
    elif args.pretrained:
        with np.load(args.pretrained, allow_pickle=False) as weights:
            model.load_from(weights)
    trainer_multi_oct(args, model, args.output_dir)


if __name__ == '__main__':
    main()
