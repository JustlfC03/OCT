import copy

import torch


def build_model(config, img_size, device):
    from networks.vit_seg_modeling import VisionTransformer as ViT_seg
    from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
    # Deep-copy the configuration to avoid mutating global config/skip_channels across model builds.
    name = config.get('vit_name', 'R50-ViT-B_16')
    model_config = copy.deepcopy(CONFIGS_ViT_seg[name])
    if len(img_size) != 2 or any(s <= 0 or s % 16 for s in img_size):
        raise ValueError('The decoder requires positive img_size dimensions divisible by 16')
    model_config.n_classes = config['num_classes']
    model_config.n_skip = config.get('n_skip', 3) if name.startswith('R50') else 0
    if name.startswith('R50'):
        model_config.patches.grid = (img_size[0] // 16, img_size[1] // 16)
    elif tuple(model_config.patches.size) != (16, 16) or img_size[0] != img_size[1]:
        raise ValueError('The non-R50 entry point supports only square inputs with 16x16 patches')
    return ViT_seg(model_config, img_size=tuple(img_size), num_classes=config['num_classes']).to(device)


def load_checkpoint(model, path, device):
    # Load weights only; support legacy state_dict files and checkpoints containing configuration.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get('model', checkpoint)
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    model.load_state_dict(state)
    return checkpoint


def select_device(name=None):
    return torch.device(name or ('cuda' if torch.cuda.is_available() else 'cpu'))
