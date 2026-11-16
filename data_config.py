import json
from pathlib import Path


def load_config(config_path):
    path = Path(config_path).resolve()
    try:
        config = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read configuration: {path}") from exc
    mapping = {int(k): int(v) for k, v in config['label_map'].items()}
    num_classes = config['num_classes']
    if set(mapping.values()) != set(range(num_classes)):
        raise ValueError('label_map must cover 0 through num_classes - 1; check the label definitions')
    if any(k < 0 or k > 255 for k in mapping):
        raise ValueError('The data loader supports 8-bit indexed or grayscale masks')
    config['label_map'] = mapping
    config.setdefault('retinal_layer', list(range(1, num_classes)))
    if not config['retinal_layer'] or any(c not in range(num_classes) for c in config['retinal_layer']):
        raise ValueError('retinal_layer classes must be within the model output range')
    for dataset in config['datasets'].values():
        for key in ('base_dir', 'list_dir', 'metadata'):
            if dataset.get(key):
                value = Path(dataset[key])
                dataset[key] = str((path.parent / value).resolve()) if not value.is_absolute() else str(value)
        if 'label_map' in dataset:
            dataset['label_map'] = {int(k): int(v) for k, v in dataset['label_map'].items()}
            if any(c not in range(num_classes) for c in dataset['label_map'].values()):
                raise ValueError('Dataset label_map exceeds the model output range')
    return config


def window_starts(length, size, stride):
    if size <= 0 or stride <= 0 or stride > size:
        raise ValueError('Window sizes and strides must be positive, and strides must not exceed window sizes')
    end = max(0, length - size)
    starts = list(range(0, end + 1, stride))
    if starts[-1] != end:
        starts.append(end)
    return starts
