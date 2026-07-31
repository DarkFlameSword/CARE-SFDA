from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from dataset.retinal_vessel_dataset import SourceDataset
from loss.segmentation_loss import dice_bce_loss

__all__ = ["SourceOnlyTrainer"]


def _infinite(loader):
    while True:

        for batch in loader:
            yield batch


def _loader_kwargs(batch_size, num_workers, device, prefetch_factor):
    kwargs = {"batch_size": batch_size, "shuffle": True, "drop_last": True,
              "num_workers": num_workers, "pin_memory": device.type == "cuda"}

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor

    return kwargs


class SourceOnlyTrainer:
    def __init__(self, model, optimizer, source_domain="FIVES", patch_size=512,
                 batch_size=4, source_loss=dice_bce_loss, device="cpu", use_amp=False,
                 amp_dtype=torch.float16, num_workers=0, prefetch_factor=2, grad_clip=0.0,
                 augment=True, photometric=False, train_ids=None, fdh=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.source_loss = source_loss
        self.device = torch.device(device)
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.use_scaler = use_amp and self.device.type == "cuda" and amp_dtype == torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)
        self.grad_clip = grad_clip
        self._clip_params = list(self.model.parameters())
        dataset = SourceDataset(source_domain, split="train", patch_size=patch_size, augment=augment,
                                sample_ids=train_ids, photometric=photometric, **dict(fdh or {}))
        loader = DataLoader(dataset, **_loader_kwargs(batch_size, num_workers, self.device, prefetch_factor))
        self._iter = _infinite(loader)
        self._step = 0

    def _autocast(self):
        if self.use_amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

        return nullcontext()

    def state_dict(self):
        return {
            "step": self._step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state):
        self._step = state["step"]
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])

    def step(self):
        self.model.train()
        self._step += 1
        batch = next(self._iter)
        image = batch["image"].to(self.device, non_blocking=True)
        label = batch["label"].to(self.device, non_blocking=True)
        fov = batch["fov"].to(self.device, non_blocking=True)

        with self._autocast():
            logits = self.model(image)
            loss = self.source_loss(logits, label, fov)

        self.optimizer.zero_grad(set_to_none=True)

        if self.use_scaler:
            self.scaler.scale(loss).backward()

            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self._clip_params, self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            loss.backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self._clip_params, self.grad_clip)

            self.optimizer.step()

        return {"step": self._step, "source_loss": float(loss.detach())}
