import torch
import torch.nn as nn

__all__ = ["configure_tent", "entropy_loss"]


def configure_tent(model):
    for param in model.parameters():
        param.requires_grad_(False)

    trainable = []

    for module in model.modules():

        if isinstance(module, nn.GroupNorm):

            if module.weight is not None:
                module.weight.requires_grad_(True)
                trainable.append(module.weight)

            if module.bias is not None:
                module.bias.requires_grad_(True)
                trainable.append(module.bias)

    return trainable


def entropy_loss(logits, fov, eps=1e-6):
    prob = torch.sigmoid(logits)
    entropy = -(prob * torch.log(prob + eps) + (1.0 - prob) * torch.log(1.0 - prob + eps))

    return (entropy * fov).sum() / fov.sum().clamp_min(1.0)
