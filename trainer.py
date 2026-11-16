import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_multi_oct import Multi_OCT_Dataset, RandomGenerator, read_samples
from evaluate import evaluate_samples
from metrics import mean_score
from utils import DiceLoss


def seed_worker(worker_id):
    # A top-level function supports multiprocessing DataLoader workers on Windows.
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def trainer_multi_oct(args, model, snapshot_path):
    config = args.config_data
    output = Path(snapshot_path)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metrics.csv').exists():
        raise FileExistsError('The output directory contains training records; choose a new --output_dir')
    db_train = Multi_OCT_Dataset(config['datasets'], 'train', crop_size=config['img_size'],
                                stride=config['stride'], label_map=config['label_map'],
                                preprocess=config.get('preprocess', True),
                                preprocess_config=config.get('preprocessing'),
                                transform=RandomGenerator(config['img_size']))
    val_samples = read_samples(config['datasets'], 'val', config['label_map'])
    generator = torch.Generator().manual_seed(args.seed)
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=args.device.type == 'cuda',
                             worker_init_fn=seed_worker, generator=generator)
    weights = config.get('class_weights', [0.75] + [1.0] * (config['num_classes'] - 1))
    if len(weights) != config['num_classes']:
        raise ValueError('class_weights length must equal num_classes')
    ce_loss = CrossEntropyLoss(weight=torch.tensor(weights, device=args.device))
    dice_loss = DiceLoss(config['num_classes'])
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    run_args = {k: str(v) if isinstance(v, (Path, torch.device)) else v
                for k, v in vars(args).items() if k != 'config_data'}
    (output / 'run_args.json').write_text(json.dumps(run_args, ensure_ascii=False, indent=2), encoding='utf-8')
    best_performance = -1.0
    print(f'Training: {len(db_train.sample_list)} images / {len(db_train)} windows; validation: {len(val_samples)} full images')
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'val_dice', 'val_iou'])
        writer.writeheader()
        for epoch_num in range(args.max_epochs):
            model.train()
            epoch_loss = 0.0
            for sampled_batch in tqdm(trainloader, desc=f'Epoch {epoch_num + 1}/{args.max_epochs}'):
                image_batch = sampled_batch['image'].to(args.device)
                label_batch = sampled_batch['label'].to(args.device)
                outputs = model(image_batch)
                loss = 0.5 * ce_loss(outputs, label_batch) + 0.5 * dice_loss(outputs, label_batch, softmax=True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * image_batch.shape[0]
            summary, _ = evaluate_samples(model, val_samples, config, config['img_size'], args.device,
                                          args.sw_batch_size)
            rows = [row for row in summary if row['group_type'] == 'overall']
            val_dice, val_iou = mean_score(rows, 'dice'), mean_score(rows, 'iou')
            record = dict(epoch=epoch_num + 1, train_loss=epoch_loss / len(db_train),
                          val_dice=val_dice, val_iou=val_iou)
            writer.writerow(record)
            stream.flush()
            print(record)
            checkpoint = {'model': model.state_dict(), 'epoch': epoch_num + 1, 'config': config}
            torch.save(checkpoint, output / 'last_model.pth')
            if val_dice > best_performance:
                best_performance = val_dice
                torch.save(checkpoint, output / 'best_model.pth')
    return 'Training Finished!'
