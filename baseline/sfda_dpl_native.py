import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from skimage.transform import resize as sk_resize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_CARE_ROOT = os.path.dirname(_HERE)
_DPL_ROOT = os.path.join(_CARE_ROOT, "model", "SFDA-DPL")
sys.path.insert(0, _CARE_ROOT)
sys.path.insert(0, _DPL_ROOT)

import networks.backbone.mobilenet as _dpl_mobilenet
_dpl_mobilenet.MobileNetV2._load_pretrained_model = lambda self: None

from networks.deeplabv3 import DeepLab
from networks.deeplabv3_eval import DeepLab as DeepLabEval

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
MC_PASSES = 10
ETA = 0.05
PL_THRESHOLD = 0.75
GEN_BATCH = 4
SOURCE_LR = 1e-3
ADAPT_LR = 2e-3


def _native_rgb(sample, invert):
    rgb = _load_rgb(sample.image_path)
    fov = _resolve_fov(sample, working_channel(rgb))
    rgb = rgb.astype(np.float32)

    if invert:
        rgb = (255.0 - rgb) * fov[..., None].astype(np.float32)

    return rgb, fov


def _normalize_tf(rgb_chw):
    return rgb_chw / 127.5 - 1.0


def _infinite(loader):
    while True:

        for batch in loader:

            yield batch


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


def build_model(device):
    return DeepLab(num_classes=NUM_CLASSES, backbone="mobilenet", output_stride=16,
                   sync_bn=False, freeze_bn=False).to(device)


class DPLDataset(Dataset):
    def __init__(self, name, sample_ids, invert, mode="source", pseudo=None, seed=0):
        allowed = set(sample_ids)
        require_label = mode == "source"
        self.samples = [s for s in index_domain(name, split="train", require_label=require_label)
                        if s.sample_id in allowed]
        self.invert = invert
        self.mode = mode
        self.pseudo = pseudo or {}
        self.seed = seed
        self._draw = 0

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):
        rng = np.random.default_rng(np.random.SeedSequence(entropy=self.seed, spawn_key=(idx, self._draw)))
        self._draw += 1
        sample = self.samples[idx]
        rgb, fov = _native_rgb(sample, self.invert)
        size = (INPUT_SIZE, INPUT_SIZE)
        rgb = sk_resize(rgb, size, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)

        if self.mode == "source":
            label = _load_label(sample.label_path).astype(np.float32)
            label = (sk_resize(label, size, order=0, preserve_range=True, anti_aliasing=False) > 0.5).astype(np.float32)

        else:
            label = np.zeros(size, np.float32)

        if self.mode == "source":

            if rng.random() < 0.5:
                rgb, label = rgb[:, ::-1], label[:, ::-1]

            k = int(rng.integers(0, 4))

            if k:
                rgb, label = np.rot90(rgb, k), np.rot90(label, k)

        elif self.mode == "adapt":

            if rng.random() < 0.5:
                rgb = rgb + rng.normal(0, 8.0, rgb.shape).astype(np.float32)

            if rng.random() < 0.5:
                gamma = float(rng.uniform(0.7, 1.5))
                rgb = np.clip(255.0 * (np.clip(rgb, 0, 255) / 255.0) ** gamma, 0, 255).astype(np.float32)

        image = torch.from_numpy(_normalize_tf(np.ascontiguousarray(rgb.transpose(2, 0, 1))))
        out = {"image": image, "sample_id": sample.sample_id}

        if self.mode == "source":
            out["label"] = torch.from_numpy(np.ascontiguousarray(label)[None])

        elif self.mode == "adapt":
            out["pseudo"] = torch.from_numpy(np.ascontiguousarray(self.pseudo[sample.sample_id]))

        return out


def pretrain_source(source_ids, iters, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(device)
    model.train()
    dataset = DPLDataset(SOURCE_DOMAIN, source_ids, invert=DOMAINS[SOURCE_DOMAIN].invert, mode="source", seed=seed)
    loader = _infinite(DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=SOURCE_LR, betas=(0.9, 0.99))
    criterion = nn.BCEWithLogitsLoss()

    for step in range(iters):
        batch = next(loader)
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        logits = model(image)[0]
        loss = criterion(logits, label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % 1000 == 0:
            print(f"[dpl-pretrain] iter {step + 1}/{iters} bce {float(loss):.4f}", flush=True)

    return model.state_dict()


def generate_pseudo_labels(model, name, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.train()
    adapt_ids = domain_split(name, eval_fraction=TARGET_EVAL_FRACTION, seed=seed)[0]
    dataset = DPLDataset(name, adapt_ids, invert=DOMAINS[name].invert, mode="gen", seed=seed)
    loader = DataLoader(dataset, batch_size=GEN_BATCH, shuffle=False, num_workers=0)
    store = {}
    fractions = []

    with torch.no_grad():

        for batch in loader:
            image = batch["image"].to(device)
            names = batch["sample_id"]

            if image.shape[0] == 1:
                image = image.repeat(2, 1, 1, 1)

            b = image.shape[0]
            logit_list = []
            feat_sum = None

            for _ in range(MC_PASSES):
                logit, _, feat = model(image)
                logit_list.append(logit)
                feat_sum = feat if feat_sum is None else feat_sum + feat

            logits = torch.stack(logit_list, dim=0)
            preds1 = torch.sigmoid(logits)
            preds_half = torch.sigmoid(logits / 2.0)
            std_map = torch.std(preds_half, dim=0)
            prediction = torch.mean(preds1, dim=0)
            pseudo_label = (prediction > PL_THRESHOLD).float()
            feature = feat_sum / MC_PASSES

            fh, fw = feature.shape[2:]
            target_obj = F.interpolate(pseudo_label, size=(fh, fw), mode="nearest")
            target_bck = 1.0 - target_obj
            pred_small = F.interpolate(prediction, size=(fh, fw), mode="bilinear", align_corners=True)
            std_small = F.interpolate(std_map, size=(fh, fw), mode="bilinear", align_corners=True)
            mask_conf = (std_small < ETA).float()

            feature_obj = feature * target_obj * mask_conf
            feature_bck = feature * target_bck * mask_conf
            centroid_obj = torch.sum(feature_obj * pred_small, dim=[0, 2, 3], keepdim=True)
            centroid_bck = torch.sum(feature_bck * (1.0 - pred_small), dim=[0, 2, 3], keepdim=True)
            cnt_obj = torch.sum(mask_conf * target_obj * pred_small, dim=[0, 2, 3], keepdim=True)
            cnt_bck = torch.sum(mask_conf * target_bck * (1.0 - pred_small), dim=[0, 2, 3], keepdim=True)
            centroid_obj = centroid_obj / (cnt_obj + 1e-7)
            centroid_bck = centroid_bck / (cnt_bck + 1e-7)
            distance_obj = torch.sum((feature - centroid_obj) ** 2, dim=1, keepdim=True)
            distance_bck = torch.sum((feature - centroid_bck) ** 2, dim=1, keepdim=True)
            proto_pseudo = (distance_obj < distance_bck).float()
            proto_pseudo = F.interpolate(proto_pseudo, size=(INPUT_SIZE, INPUT_SIZE), mode="nearest")

            for i in range(len(names)):
                store[names[i]] = {"pseudo": pseudo_label[i].cpu().numpy(),
                                   "std": std_map[i].cpu().numpy(),
                                   "proto": proto_pseudo[i].cpu().numpy()}
                fractions.append(float(pseudo_label[i].mean()))

    return store, float(np.mean(fractions)) if fractions else 0.0


def adapt_target(name, source_state, pseudo_store, iters, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(device)
    model.load_state_dict(source_state)
    model.train()
    pseudo = {sid: v["pseudo"] for sid, v in pseudo_store.items()}
    std = {sid: v["std"] for sid, v in pseudo_store.items()}
    proto = {sid: v["proto"] for sid, v in pseudo_store.items()}
    adapt_ids = domain_split(name, eval_fraction=TARGET_EVAL_FRACTION, seed=seed)[0]
    dataset = DPLDataset(name, adapt_ids, invert=DOMAINS[name].invert, mode="adapt", pseudo=pseudo, seed=seed)
    loader = _infinite(DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=ADAPT_LR, betas=(0.9, 0.99))
    bce = nn.BCELoss(reduction="none")

    for step in range(iters):
        batch = next(loader)
        image = batch["image"].to(device)
        names = batch["sample_id"]
        pl = batch["pseudo"].to(device)
        std_b = torch.from_numpy(np.stack([std[n] for n in names])).to(device)
        proto_b = torch.from_numpy(np.stack([proto[n] for n in names])).to(device)
        mask = ((std_b < ETA).float()) * ((pl == proto_b).float())
        prediction = torch.sigmoid(model(image)[0])
        loss_pixel = bce(prediction, pl)
        loss = torch.sum(mask * loss_pixel) / (torch.sum(mask) + 1e-7)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % 200 == 0:
            print(f"[dpl-adapt:{name}] iter {step + 1}/{iters} loss {float(loss):.4f} "
                  f"kept {float(mask.mean()):.3f}", flush=True)

    return model


def evaluate_target(model, name, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    eval_model = DeepLabEval(num_classes=NUM_CLASSES, backbone="mobilenet", output_stride=16,
                             sync_bn=False, freeze_bn=False).to(device)
    src_state = model.state_dict()
    dst_state = eval_model.state_dict()
    eval_model.load_state_dict({k: v for k, v in src_state.items() if k in dst_state})
    eval_model.train()
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
        image = torch.from_numpy(_normalize_tf(np.ascontiguousarray(rgb_eval.transpose(2, 0, 1)))[None]).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(eval_model(image)[0])[0, 0].cpu().numpy()

        prob_native = sk_resize(prob, native_shape, order=1, preserve_range=True, anti_aliasing=False)
        per_image.append(care_evaluate(prob_native, label_native, roi=fov, threshold=0.5))
        torch.cuda.empty_cache()

    keys = ["dice", "cldice", "iou", "sensitivity", "specificity"]

    return {k: float(np.mean([m[k] for m in per_image])) for k in keys}


def main():
    parser = argparse.ArgumentParser(description="B-8 SFDA-DPL native-design baseline.")
    parser.add_argument("--targets", type=str, default=",".join(TARGET_DOMAINS))
    parser.add_argument("--pretrain-iters", type=int, default=6000)
    parser.add_argument("--adapt-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    device = torch.device(args.device)
    run_dir = Path("result/baselines_7targets") / f"sfda_dpl_native_{args.date}"
    (run_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
    source_ids = domain_split(SOURCE_DOMAIN, eval_fraction=SOURCE_VAL_FRACTION, seed=args.seed)[0]

    config = {
        "experiment": "B-8 SFDA-DPL (Denoised Pseudo-Labeling, MICCAI 2021) native-design reproduction",
        "method": "source-free denoised pseudo-labeling: MC-dropout (K=10) pixel-level uncertainty (std<eta) "
                  "+ prototype (class-centroid) class-level denoising -> masked BCE self-training "
                  "(verbatim model/SFDA-DPL DeepLabv3+ architecture)",
        "regime": "native: RGB -> polarity flip (JRCFA) + re-zero FOV -> resize whole to 512x512 -> "
                  "Normalize_tf [-1,1]; MobileNetV2 DeepLabv3+ trained FROM SCRATCH (dead ImageNet URL, "
                  "monkeypatched, CBMT parity); single-logit vessel head; train-mode BN inference",
        "twin_of_B4_CBMT": "same DeepLabv3+ mobilenet + Normalize_tf + dead-URL-from-scratch + 512^2; only "
                           "the adaptation differs (DPL denoised-PL vs CBMT class-balanced mean-teacher)",
        "controlled_only": {"source": f"DRIVEN train split seed {args.seed} ({len(source_ids)} imgs)",
                            "splits": f"domain_split(t, {TARGET_EVAL_FRACTION}, {args.seed}) identical to CARE",
                            "evaluation": "CARE util.evaluate_metrics on native GT/FOV (dice/cldice/iou), thr 0.5",
                            "label_free": "fixed adapt budget, final checkpoint, NO target-label selection",
                            "shared_physical_necessity": "JRCFA polarity flip + re-zero FOV only"},
        "hyperparameters": {"mc_passes": MC_PASSES, "eta": ETA, "pl_threshold": PL_THRESHOLD,
                            "source_lr": SOURCE_LR, "adapt_lr": ADAPT_LR, "gen_batch": GEN_BATCH,
                            "adapt_batch": 2},
        "caveat": "DPL paper (Sec.4) targets MINOR (same-modality) shift and anticipates failure under "
                  "SEVERE cross-modality shift (source under-segments -> scarce positive pseudo-labels); "
                  "our benchmark is severe cross-modality + weak source -> expected weak (honest).",
        "protocol": f"pretrain {args.pretrain_iters} / adapt {args.adapt_iters} iters, batch 2 (adapt) / {GEN_BATCH} (gen)",
        "seed": args.seed,
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    source_ckpt = run_dir / "checkpoint" / "source.pt"

    if source_ckpt.exists():
        print(f"[dpl] loading existing source checkpoint {source_ckpt}", flush=True)
        source_state = torch.load(source_ckpt, map_location=device)

    else:
        print(f"[dpl] pretraining source on DRIVEN ({len(source_ids)} imgs, {args.pretrain_iters} iters)", flush=True)
        source_state = pretrain_source(source_ids, args.pretrain_iters, device, args.seed)
        torch.save(source_state, source_ckpt)

    metric_path = run_dir / "metric.json"
    results = json.load(open(metric_path, encoding="utf-8"))["per_domain"] if metric_path.exists() else {}
    pl_fractions = json.load(open(metric_path, encoding="utf-8")).get("pseudo_label_vessel_fraction", {}) \
        if metric_path.exists() else {}

    for name in targets:

        if name in results:
            print(f"[dpl] {name} already done, skipping", flush=True)
            continue

        print(f"[dpl] generating denoised pseudo-labels for {name} (invert={DOMAINS[name].invert})", flush=True)
        gen_model = build_model(device)
        gen_model.load_state_dict(source_state)
        pseudo_store, vessel_fraction = generate_pseudo_labels(gen_model, name, device, args.seed)
        print(f"[dpl] {name} pseudo-label vessel fraction {vessel_fraction:.4f}", flush=True)
        del gen_model
        torch.cuda.empty_cache()

        print(f"[dpl] adapting {name} ({args.adapt_iters} iters)", flush=True)
        model = adapt_target(name, source_state, pseudo_store, args.adapt_iters, device, args.seed)
        results[name] = evaluate_target(model, name, device, args.seed)
        pl_fractions[name] = vessel_fraction
        print(f"[dpl] {name}: dice {results[name]['dice']:.4f} cldice {results[name]['cldice']:.4f} "
              f"iou {results[name]['iou']:.4f}", flush=True)
        torch.save(model.state_dict(), run_dir / "checkpoint" / f"{name}_adapted.pt")
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

    print(f"[dpl] done. artifacts under {run_dir}", flush=True)


if __name__ == "__main__":
    main()
