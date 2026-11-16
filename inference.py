import numpy as np
import torch
from monai.inferers import sliding_window_inference


def predict_image(model, image, img_size, device, sw_batch_size=4):
    # Keep full images on CPU, send windows to the model, and blend logits with Gaussian weights.
    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None]
    with torch.no_grad():
        output = sliding_window_inference(tensor, roi_size=tuple(img_size),
                                          sw_batch_size=sw_batch_size, predictor=model,
                                          overlap=0.5, mode='gaussian', sw_device=device,
                                          device=torch.device('cpu'))
    return output.argmax(dim=1)[0].numpy().astype(np.uint8)
