import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["dice_loss", "bce_loss", "dice_bce_loss", "SegmentationLoss"]


def _masked_mean(value, fov):
    if fov is None:
        return value.mean()

    fov = fov.to(value.dtype)
    denom = fov.sum().clamp_min(1.0)

    return (value * fov).sum() / denom


def dice_loss(logits, target, fov=None, eps=1.0):
    prob = torch.sigmoid(logits)

    if fov is not None:
        fov = fov.to(prob.dtype)
        prob = prob * fov
        target = target * fov

    dims = (1, 2, 3)
    numerator = 2.0 * (prob * target).sum(dims) + eps
    denominator = prob.sum(dims) + target.sum(dims) + eps

    return (1.0 - numerator / denominator).mean()


def bce_loss(logits, target, fov=None):
    per_pixel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    return _masked_mean(per_pixel, fov)


def dice_bce_loss(logits, target, fov=None, dice_weight=1.0, bce_weight=1.0, eps=1.0):
    return dice_weight * dice_loss(logits, target, fov, eps) + bce_weight * bce_loss(logits, target, fov)


class SegmentationLoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0, eps=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.eps = eps

    def forward(self, logits, target, fov=None):
        return dice_bce_loss(logits, target, fov, self.dice_weight, self.bce_weight, self.eps)
