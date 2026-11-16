import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    # Preserve the original Dice loss: squared denominator, background included, equal class weights.
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = torch.stack([target == i for i in range(self.n_classes)], dim=1).float()
        if inputs.shape != target.shape:
            raise ValueError(f'Prediction {inputs.shape} and target {target.shape} dimensions differ')
        weight = [1] * self.n_classes if weight is None else weight
        loss = 0.0
        for i in range(self.n_classes):
            score, truth = inputs[:, i], target[:, i]
            dice = (2 * torch.sum(score * truth) + 1e-5) / (torch.sum(score * score) + torch.sum(truth * truth) + 1e-5)
            loss = loss + (1 - dice) * weight[i]
        return loss / self.n_classes
