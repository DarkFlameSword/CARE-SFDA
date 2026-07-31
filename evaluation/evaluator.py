from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch

from dataset.retinal_vessel_dataset import EvalDataset
from dataset.tiling import score_native
from dataset.tiling import sliding_window_predict

__all__ = ["Evaluator", "make_predict_fn"]


def make_predict_fn(model, device, use_amp=False):
    device = torch.device(device)

    def predict(tile):
        tensor = torch.from_numpy(np.ascontiguousarray(tile, dtype=np.float32))[None, None].to(device)
        autocast = (torch.autocast(device_type="cuda", dtype=torch.float16)
                    if use_amp and device.type == "cuda" else nullcontext())

        with torch.no_grad():

            with autocast:
                prob = torch.sigmoid(model(tensor))

        return prob[0, 0].float().cpu().numpy()

    return predict


class Evaluator:
    def __init__(self, source_domain, target_domains, device="cpu", tile=512,
                 overlap=0.25, threshold=0.5, max_images=None, use_amp=False, datasets=None):
        self.source_domain = source_domain
        self.target_domains = list(target_domains)
        self.device = device
        self.tile = tile
        self.overlap = overlap
        self.threshold = threshold
        self.max_images = max_images
        self.use_amp = use_amp

        if datasets is not None:
            self.datasets = datasets

        else:
            self.datasets = {source_domain: EvalDataset(source_domain, split="test")}

            for d in self.target_domains:
                self.datasets[d] = EvalDataset(d, split="train")

    def _evaluate_dataset(self, dataset, predict):
        accum = defaultdict(list)
        count = len(dataset) if self.max_images is None else min(len(dataset), self.max_images)

        if count == 0:
            raise ValueError("Evaluator: no images to score for a domain (check --max-eval-images).")

        for i in range(count):
            item = dataset[i]
            prob = sliding_window_predict(predict, item["image"][0].numpy(), tile=self.tile, overlap=self.overlap)
            metrics = score_native(prob, item["native_shape"], item["label_native"],
                                   item["fov_native"], threshold=self.threshold)

            for key, value in metrics.items():
                accum[key].append(value)

        return {key: float(np.mean(values)) for key, values in accum.items()}

    def evaluate(self, model):
        was_training = model.training
        model.eval()
        predict = make_predict_fn(model, self.device, self.use_amp)
        results = {name: self._evaluate_dataset(dataset, predict) for name, dataset in self.datasets.items()}

        if was_training:
            model.train()

        return results
