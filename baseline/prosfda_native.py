import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from skimage.transform import resize as sk_resize
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_CARE_ROOT = os.path.dirname(_HERE)
_PROSFDA_ROOT = os.path.join(_CARE_ROOT, "model", "ProSFDA")
sys.path.insert(0, _CARE_ROOT)
sys.path.insert(0, _PROSFDA_ROOT)

from prosfda.loss_functions.bn_loss import bn_loss
from prosfda.models.unet import UNet
from prosfda.models.unet_fas import UNet_FAS
from prosfda.models.unet_pls import UNet_PLS
from prosfda.utils.fourier import FDA_source_to_target_np
from prosfda.utils.mix_prompt import mix_data_prompt

from dataset.domain_registry import DOMAINS
from dataset.domain_registry import SOURCE_DOMAIN
from dataset.domain_registry import TARGET_DOMAINS
from dataset.domain_registry import index_domain
from dataset.harmonization import working_channel
from dataset.retinal_vessel_dataset import _load_label
from dataset.retinal_vessel_dataset import _load_rgb
from dataset.retinal_vessel_dataset import _resolve_fov
from dataset.splits import domain_split
from util.evaluate_metrics import evaluate as care_evaluate

SOURCE_VAL_FRACTION = 0.15
TARGET_EVAL_FRACTION = 0.30
INPUT_SIZE = 512
NUM_CLASSES = 1
ALPHA_SA = 0.01
GAMMA_FAS = 0.1
BN_LAYERS = 5
FDA_BETA = 0.15
PL_THRESHOLD = 0.5


def _native_rgb(sample, invert):
    rgb = _load_rgb(sample.image_path)
    fov = _resolve_fov(sample, working_channel(rgb))
    rgb = rgb.astype(np.float32)

    if invert:
        rgb = (255.0 - rgb) * fov[..., None].astype(np.float32)

    return rgb, fov


def _zscore_chw(img_chw):
    out = np.empty_like(img_chw, dtype=np.float32)

    for c in range(img_chw.shape[0]):
        channel = img_chw[c]
        std = channel.std()
        out[c] = (channel - channel.mean()) / (std if std > 1e-6 else 1.0)

    return out


def _fda_batch(raw_bchw, beta, rng):
    donor = raw_bchw[rng.permutation(raw_bchw.shape[0])]
    this_beta = round(float(rng.random()) * beta, 2)
    fda = FDA_source_to_target_np(raw_bchw, donor, L=this_beta)

    return fda.astype(np.float32)


def modality_groups(targets):
    groups = {}

    for domain in targets:
        groups.setdefault(DOMAINS[domain].modality, []).append(domain)

    return groups


def summarize(per_domain, groups):
    group_value = {m: float(np.mean([per_domain[d] for d in members])) for m, members in groups.items()}
    values = list(group_value.values())

    return {"mean": float(np.mean(list(per_domain.values()))),
            "worst": float(min(values)), "spread": float(max(values) - min(values)),
            "group_dice": group_value}


def _poly_lr(optimizer, lr0, step, total, power=0.9):
    lr = lr0 * (1.0 - step / max(total, 1)) ** power

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


def _infinite(loader):
    while True:

        for batch in loader:

            yield batch


class ProSFDADataset(Dataset):
    def __init__(self, name, sample_ids, invert, size=INPUT_SIZE, label_source="gt",
                 pseudo=None, augment=True, seed=0):
        allowed = set(sample_ids)
        require_label = label_source == "gt"
        self.samples = [s for s in index_domain(name, split="train", require_label=require_label)
                        if s.sample_id in allowed]
        self.invert = invert
        self.size = size
        self.label_source = label_source
        self.pseudo = pseudo or {}
        self.augment = augment
        self.seed = seed
        self._draw = 0

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):
        rng = np.random.default_rng(np.random.SeedSequence(entropy=self.seed, spawn_key=(idx, self._draw)))
        self._draw += 1
        sample = self.samples[idx]
        rgb, fov = _native_rgb(sample, self.invert)
        size = (self.size, self.size)
        rgb = sk_resize(rgb, size, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)

        if self.label_source == "gt":
            label = _load_label(sample.label_path).astype(np.float32)
            label = (sk_resize(label, size, order=0, preserve_range=True, anti_aliasing=False) > 0.5).astype(np.float32)

        elif self.label_source == "pseudo":
            label = self.pseudo[sample.sample_id].astype(np.float32)

        else:
            label = np.zeros(size, np.float32)

        if self.augment:

            if rng.random() < 0.5:
                rgb = rgb[:, ::-1]
                label = label[:, ::-1]

            k = int(rng.integers(0, 4))

            if k:
                rgb = np.rot90(rgb, k)
                label = np.rot90(label, k)

        raw = np.ascontiguousarray(rgb.transpose(2, 0, 1))

        return {"raw": torch.from_numpy(raw),
                "label": torch.from_numpy(np.ascontiguousarray(label)[None]),
                "sample_id": sample.sample_id}


def _zscore_batch(raw_bchw):
    arr = raw_bchw.cpu().numpy() if torch.is_tensor(raw_bchw) else raw_bchw

    return torch.from_numpy(np.stack([_zscore_chw(img) for img in arr]))


def pretrain_source(source_ids, iters, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = UNet(resnet="resnet34", num_classes=NUM_CLASSES, pretrained=False).to(device)
    net.train()
    dataset = ProSFDADataset(SOURCE_DOMAIN, source_ids, invert=DOMAINS[SOURCE_DOMAIN].invert,
                             label_source="gt", augment=True, seed=seed)
    loader = _infinite(DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True))
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.99, nesterov=True)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    for step in range(iters):
        batch = next(loader)
        image = _zscore_batch(batch["raw"]).to(device)
        label = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = net(image)
            loss = criterion(logits, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        _poly_lr(optimizer, 1e-2, step, iters)

        if (step + 1) % 1000 == 0:
            print(f"[prosfda-pretrain] iter {step + 1}/{iters} bce {float(loss):.4f}", flush=True)

    net.close()

    return {"model_state_dict": net.state_dict()}


def run_pls(name, source_ckpt_path, iters, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    source_params = torch.load(source_ckpt_path, map_location=device)
    model = UNet_PLS(pretrained_path=source_ckpt_path, patch_size=(INPUT_SIZE, INPUT_SIZE),
                     resnet="resnet34", num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.eval()
    adapt_ids = domain_split(name, eval_fraction=TARGET_EVAL_FRACTION, seed=seed)[0]
    dataset = ProSFDADataset(name, adapt_ids, invert=DOMAINS[name].invert,
                             label_source="none", augment=False, seed=seed)
    loader = _infinite(DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True))
    optimizer = torch.optim.SGD([model.data_prompt], lr=1e-2, momentum=0.99, nesterov=True)

    for step in range(iters):
        batch = next(loader)
        image = _zscore_batch(batch["raw"]).to(device)
        optimizer.zero_grad(set_to_none=True)
        _, bn_f = model(image, training=True)
        loss = bn_loss(model, source_params, bn_f, alpha=ALPHA_SA, i=BN_LAYERS)
        loss.backward()
        optimizer.step()
        _poly_lr(optimizer, 1e-2, step, iters)

        if (step + 1) % 200 == 0:
            print(f"[prosfda-pls:{name}] iter {step + 1}/{iters} L_SA {float(loss):.4f}", flush=True)

    return model, model.data_prompt.detach().clone()


def generate_pseudo_labels(pls_model, name, device, seed):
    pls_model.eval()
    adapt_ids = domain_split(name, eval_fraction=TARGET_EVAL_FRACTION, seed=seed)[0]
    invert = DOMAINS[name].invert
    id_to_sample = {s.sample_id: s for s in index_domain(name, split="train", require_label=False)}
    pseudo = {}
    fractions = []

    for sid in adapt_ids:
        sample = id_to_sample[sid]
        rgb, _ = _native_rgb(sample, invert)
        rgb = sk_resize(rgb, (INPUT_SIZE, INPUT_SIZE), order=1, preserve_range=True,
                        anti_aliasing=True).astype(np.float32)
        image = _zscore_batch(torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]).to(device)

        with torch.no_grad():
            logits = pls_model(image)

        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        mask = (prob > PL_THRESHOLD).astype(np.float32)
        pseudo[sid] = mask
        fractions.append(float(mask.mean()))

    return pseudo, float(np.mean(fractions)) if fractions else 0.0


def run_fas(name, source_ckpt_path, prompt, pseudo, iters, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    fda_rng = np.random.default_rng(seed)
    source_params = torch.load(source_ckpt_path, map_location=device)
    model = UNet_FAS(resnet="resnet34", num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(source_params["model_state_dict"])
    model.train()
    prompt = prompt.to(device)
    adapt_ids = domain_split(name, eval_fraction=TARGET_EVAL_FRACTION, seed=seed)[0]
    dataset = ProSFDADataset(name, adapt_ids, invert=DOMAINS[name].invert,
                             label_source="pseudo", pseudo=pseudo, augment=True, seed=seed)
    loader = _infinite(DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, nesterov=True)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    for step in range(iters):
        batch = next(loader)
        raw = batch["raw"].cpu().numpy()
        label = batch["label"].to(device)
        data = _zscore_batch(torch.from_numpy(raw)).to(device)
        fda = _zscore_batch(torch.from_numpy(_fda_batch(raw, FDA_BETA, fda_rng))).to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            out_data, gf_data = model(mix_data_prompt(data, prompt), gfeat=True)
            out_fda, gf_fda = model(mix_data_prompt(fda, prompt), gfeat=True)
            compact = nn.functional.l1_loss(gf_fda, gf_data.detach()) + \
                nn.functional.l1_loss(gf_data, gf_fda.detach())
            seg = criterion(out_data[:, 0:1], label) + criterion(out_fda[:, 0:1], label)
            loss = seg + GAMMA_FAS * compact

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        _poly_lr(optimizer, 1e-3, step, iters)

        if (step + 1) % 200 == 0:
            print(f"[prosfda-fas:{name}] iter {step + 1}/{iters} seg {float(seg):.3f} "
                  f"align {float(compact):.3f}", flush=True)

    return model


def evaluate_target(model, prompt, name, device, seed):
    model.eval()
    prompt = prompt.to(device)
    eval_ids = domain_split(name, eval_fraction=TARGET_EVAL_FRACTION, seed=seed)[1]
    invert = DOMAINS[name].invert
    id_to_sample = {s.sample_id: s for s in index_domain(name, split="train", require_label=True)}
    per_image = []

    for sid in eval_ids:
        sample = id_to_sample[sid]
        rgb, fov = _native_rgb(sample, invert)
        label_native = _load_label(sample.label_path)
        native_shape = fov.shape
        rgb_eval = sk_resize(rgb, (INPUT_SIZE, INPUT_SIZE), order=1, preserve_range=True,
                             anti_aliasing=True).astype(np.float32)
        image = _zscore_batch(torch.from_numpy(np.ascontiguousarray(rgb_eval.transpose(2, 0, 1)))[None]).to(device)

        with torch.no_grad():

            with autocast():
                logits = model(mix_data_prompt(image, prompt))

        prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
        prob_native = sk_resize(prob, native_shape, order=1, preserve_range=True, anti_aliasing=False)
        per_image.append(care_evaluate(prob_native, label_native, roi=fov, threshold=0.5))
        torch.cuda.empty_cache()

    keys = ["dice", "cldice", "iou", "sensitivity", "specificity"]

    return {k: float(np.mean([m[k] for m in per_image])) for k in keys}


def main():
    parser = argparse.ArgumentParser(description="B-6 ProSFDA native-design baseline.")
    parser.add_argument("--targets", type=str, default=",".join(TARGET_DOMAINS))
    parser.add_argument("--pretrain-iters", type=int, default=6000)
    parser.add_argument("--pls-iters", type=int, default=1000)
    parser.add_argument("--fas-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    device = torch.device(args.device)
    run_dir = Path("result/baselines_7targets") / f"prosfda_native_{args.date}"
    (run_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
    source_ids = domain_split(SOURCE_DOMAIN, eval_fraction=SOURCE_VAL_FRACTION, seed=args.seed)[0]

    config = {
        "experiment": "B-6 ProSFDA (Pattern Recognition 2024) native-design reproduction",
        "method": "prompt-learning SFDA: PLS (frozen source + additive visual prompt via BN statistic "
                  "alignment L_SA, alpha=0.01) -> prompted-source pseudo-labels -> FAS (full fine-tune, "
                  "self-training on pseudo-labels over prompted clean + Fourier-styled views + L1 feature "
                  "alignment L_AL, gamma=0.1) (verbatim model/ProSFDA/prosfda)",
        "regime": "native: RGB, resize whole to 512x512, per-channel z-score; tailored ResNet-34 BN "
                  "U-Net trained FROM SCRATCH on the weak source (paper claims no ImageNet init); "
                  "single-logit vessel head (num_classes=1)",
        "controlled_only": {"source": f"DRIVEN train split seed {args.seed} ({len(source_ids)} imgs)",
                            "splits": f"domain_split(t, {TARGET_EVAL_FRACTION}, {args.seed}) identical to CARE",
                            "evaluation": "CARE util.evaluate_metrics on native GT/FOV (dice/cldice/iou)",
                            "label_free": "fixed PLS+FAS budgets, final checkpoint, NO target-label selection",
                            "shared_physical_necessity": "JRCFA polarity flip + re-zero FOV only"},
        "hyperparameters": {"alpha_SA": ALPHA_SA, "gamma_FAS": GAMMA_FAS, "bn_layers": BN_LAYERS,
                            "fda_beta": FDA_BETA, "pl_threshold": PL_THRESHOLD,
                            "source_lr": "SGD 1e-2 mom0.99 nesterov, poly power0.9",
                            "pls_lr": "SGD 1e-2 mom0.99 nesterov, poly", "fas_lr": "SGD 1e-3 mom0.99 nesterov, poly"},
        "protocol": f"pretrain {args.pretrain_iters} / PLS {args.pls_iters} / FAS {args.fas_iters} iters, "
                    f"batch 2, AMP",
        "seed": args.seed,
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    source_ckpt = run_dir / "checkpoint" / "source.pt"

    if source_ckpt.exists():
        print(f"[prosfda] loading existing source checkpoint {source_ckpt}", flush=True)

    else:
        print(f"[prosfda] pretraining source on DRIVEN ({len(source_ids)} imgs, {args.pretrain_iters} iters)", flush=True)
        source_state = pretrain_source(source_ids, args.pretrain_iters, device, args.seed)
        torch.save(source_state, source_ckpt)

    metric_path = run_dir / "metric.json"
    results = json.load(open(metric_path, encoding="utf-8"))["per_domain"] if metric_path.exists() else {}
    pl_fractions = json.load(open(metric_path, encoding="utf-8")).get("pseudo_label_vessel_fraction", {}) \
        if metric_path.exists() else {}

    for name in targets:

        if name in results:
            print(f"[prosfda] {name} already done, skipping", flush=True)
            continue

        print(f"[prosfda] PLS {name} ({args.pls_iters} iters, invert={DOMAINS[name].invert})", flush=True)
        pls_model, prompt = run_pls(name, str(source_ckpt), args.pls_iters, device, args.seed)
        pseudo, vessel_fraction = generate_pseudo_labels(pls_model, name, device, args.seed)
        print(f"[prosfda] {name} pseudo-label vessel fraction {vessel_fraction:.4f}", flush=True)
        del pls_model
        torch.cuda.empty_cache()

        print(f"[prosfda] FAS {name} ({args.fas_iters} iters)", flush=True)
        model = run_fas(name, str(source_ckpt), prompt, pseudo, args.fas_iters, device, args.seed)
        results[name] = evaluate_target(model, prompt, name, device, args.seed)
        pl_fractions[name] = vessel_fraction
        print(f"[prosfda] {name}: dice {results[name]['dice']:.4f} cldice {results[name]['cldice']:.4f} "
              f"iou {results[name]['iou']:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "prompt": prompt.cpu()},
                   run_dir / "checkpoint" / f"{name}_adapted.pt")
        del model
        torch.cuda.empty_cache()

        groups = modality_groups(list(results.keys()))
        payload = {"config": config, "per_domain": results,
                   "pseudo_label_vessel_fraction": pl_fractions,
                   "dice_summary": summarize({d: results[d]["dice"] for d in results}, groups),
                   "cldice_summary": summarize({d: results[d]["cldice"] for d in results}, groups),
                   "iou_summary": summarize({d: results[d]["iou"] for d in results}, groups)}

        with open(metric_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"[prosfda] done. artifacts under {run_dir}", flush=True)


if __name__ == "__main__":
    main()
