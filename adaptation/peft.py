import math

import torch
import torch.nn as nn

__all__ = ["LoRAConv2d", "configure_peft", "trainable_parameters", "anchor_loss"]


class LoRAConv2d(nn.Module):
    def __init__(self, base_conv, r=8, alpha=16):
        super().__init__()
        self.base = base_conv

        for p in self.base.parameters():
            p.requires_grad_(False)

        self.scaling = alpha / r
        self.lora_A = nn.Conv2d(base_conv.in_channels, r, kernel_size=base_conv.kernel_size,
                                stride=base_conv.stride, padding=base_conv.padding, bias=False)
        self.lora_B = nn.Conv2d(r, base_conv.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(base_conv.weight.device)
        self.lora_B.to(base_conv.weight.device)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))

    def delta_w_fro_sq(self):
        b = self.lora_B.weight[:, :, 0, 0]
        dw = self.scaling * torch.einsum("or,rikl->oikl", b, self.lora_A.weight)

        return (dw ** 2).sum()


def _encoder_doubleconvs(unet):
    return [unet.inc, unet.down1.conv, unet.down2.conv, unet.down3.conv, unet.down4.conv]


def _decoder_doubleconvs(unet):
    return [unet.up1.conv, unet.up2.conv, unet.up3.conv, unet.up4.conv]


def configure_peft(unet, r=8, alpha=16, include_decoder=False, use_lora=True, train_head=False):
    for p in unet.parameters():
        p.requires_grad_(False)

    lora_modules = []
    gn_anchors = []
    blocks = _encoder_doubleconvs(unet) + (_decoder_doubleconvs(unet) if include_decoder else [])

    for dc in blocks:
        block = dc.block

        if use_lora:

            for i in (0, 3):
                lora = LoRAConv2d(block[i], r=r, alpha=alpha)
                block[i] = lora
                lora_modules.append(lora)

        for i in (1, 4):
            gn = block[i]
            gn.weight.requires_grad_(True)
            gn.bias.requires_grad_(True)
            gn_anchors.append((gn, gn.weight.detach().clone(), gn.bias.detach().clone()))

    if train_head:
        unet.head.weight.requires_grad_(True)

    return {"lora": lora_modules, "gn": gn_anchors}


def trainable_parameters(unet):
    return [p for p in unet.parameters() if p.requires_grad]


def anchor_loss(peft_state, lambda_gn):
    total = 0.0

    for lora in peft_state["lora"]:
        total = total + lora.delta_w_fro_sq()

    for gn, gamma_o, beta_o in peft_state["gn"]:
        total = total + lambda_gn * (((gn.weight - gamma_o) ** 2).sum() + ((gn.bias - beta_o) ** 2).sum())

    return total
