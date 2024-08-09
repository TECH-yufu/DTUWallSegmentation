import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

class Criterion(nn.Module):

    def __init__(self, lambda_=1, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.lambda_ = lambda_

    def dice(self, input, target):
        input_formatted = input.argmax(dim=1) # input.max(dim=1)[1]
        input_formatted = input_formatted.to(dtype=torch.long)
        input_formatted = F.one_hot(input_formatted, 2).permute(0, 3, 1, 2).float()
        target_formatted = target.to(dtype=torch.long) if len(target.shape) == 3 else target.to(dtype=torch.long).squeeze(1)
        target_formatted = F.one_hot(target_formatted, 2).permute(0, 3, 1, 2).float()
        dice = dice_loss(input_formatted, target_formatted, multiclass=True)
        return dice

    def forward(self, input, target):
        bs, c, h, w = input.shape
        # ce = self.ce(input.view(bs*h*w, c), target.view(bs*h*w))
        ce = self.ce(input, target.squeeze(1).long())
        dice = self.dice(input, target)
        # dice = 0
        return ce+self.lambda_*dice
