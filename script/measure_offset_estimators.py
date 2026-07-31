import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.domain_registry import DOMAINS
from dataset.domain_registry import SOURCE_DOMAIN
from dataset.domain_registry import TARGET_DOMAINS
from dataset.domain_registry import index_domain
from dataset.harmonization import _zscore_fov
from dataset.photometric import photometric_augment
from dataset.retinal_vessel_dataset import EvalDataset
from dataset.retinal_vessel_dataset import _load_label
from dataset.retinal_vessel_dataset import _prepare_image_side
from dataset.scale import compute_domain_scale
from dataset.splits import domain_split
from dataset.tiling import sliding_window_predict
from dataset.tiling import upsample_to_native
from evaluation.evaluator import make_predict_fn
from util.evaluate_metrics import dice_coefficient

from measure_offset_oracle import DEFAULT_ANCHOR
from measure_offset_oracle import DEFAULT_REGRET
from measure_offset_oracle import GRID
from measure_offset_oracle import load_care_model

EVAL_FRACTION = 0.30
K_VIEWS = 2
N_EST = 30
BAND = (0.03, 0.45)


def source_vessel_prevalence():
    from dataset.retinal_vessel_dataset import _resolve_fov
    from dataset.harmonization import working_channel
    from dataset.retinal_vessel_dataset import _load_rgb

    train_ids = set(domain_split(SOURCE_DOMAIN, eval_fraction=0.15, seed=0)[0])
    fracs = []

    for s in index_domain(SOURCE_DOMAIN, split="train", require_label=True):

        if s.sample_id not in train_ids:
            continue

        green = working_channel(_load_rgb(s.image_path))
        fov = _resolve_fov(s, green).astype(bool)
        label = _load_label(s.label_path)
        fracs.append(float(label[fov].mean()))

    return float(np.mean(fracs))


def _views_for_image(sample, spec, scale, invert, rng):
    raw, fov = _prepare_image_side(sample, spec, scale, invert=invert, norm="raw")
    fov_f = fov.astype(np.float32)
    views = []

    for _ in range(K_VIEWS):
        aug = photometric_augment(np.clip(raw, 0.0, 255.0).astype(np.uint8), fov, rng, strength=1.0)
        views.append((_zscore_fov(aug.astype(np.float32), fov) * fov_f).astype(np.float32))

    return views, fov


def estimate_thresholds(model, name, device, tile, overlap, use_amp, pi_s):
    spec = DOMAINS[name]
    invert = spec.invert
    scale = compute_domain_scale(name, 1024.0)
    adapt_ids = set(domain_split(name, eval_fraction=EVAL_FRACTION, seed=0)[0])
    samples = [s for s in index_domain(name, split="train", require_label=False) if s.sample_id in adapt_ids][:N_EST]
    predict = make_predict_fn(model, device, use_amp)
    stability = np.zeros(len(GRID))
    kappa = np.zeros(len(GRID))
    fgfrac = np.zeros(len(GRID))
    n = 0

    for idx, sample in enumerate(samples):
        rng = np.random.default_rng(np.random.SeedSequence(entropy=0, spawn_key=(idx,)))
        views, fov = _views_for_image(sample, spec, scale, invert, rng)
        fov_b = fov.astype(bool)
        total = float(fov_b.sum())
        probs = [sliding_window_predict(predict, v, tile=tile, overlap=overlap) for v in views]

        for j, t in enumerate(GRID):
            a = (probs[0] > t) & fov_b
            b = (probs[1] > t) & fov_b
            na = float(a.sum())
            nb = float(b.sum())
            both = float((a & b).sum())
            denom = na + nb
            stability[j] += (2.0 * both / denom) if denom > 0 else 1.0
            fgfrac[j] += 0.5 * (na + nb) / max(1.0, total)
            p_o = (total - na - nb + 2.0 * both) / max(1.0, total)
            pa = na / max(1.0, total)
            pb = nb / max(1.0, total)
            p_e = pa * pb + (1.0 - pa) * (1.0 - pb)
            kappa[j] += ((p_o - p_e) / (1.0 - p_e)) if (1.0 - p_e) > 1e-8 else 0.0

        n += 1

    stability /= max(1, n)
    kappa /= max(1, n)
    fgfrac /= max(1, n)
    band = (fgfrac >= BAND[0]) & (fgfrac <= BAND[1])
    banded = np.where(band, stability, -np.inf)
    tau_stability = float(GRID[int(np.argmax(banded))]) if band.any() else 0.5
    tau_kappa = float(GRID[int(np.argmax(kappa))])
    tau_prior = float(GRID[int(np.argmin(np.abs(fgfrac - pi_s)))])

    return tau_stability, tau_kappa, tau_prior, stability.tolist(), kappa.tolist(), fgfrac.tolist(), n


def score_eval(model, name, device, tile, overlap, use_amp, taus):
    eval_ids = domain_split(name, eval_fraction=EVAL_FRACTION, seed=0)[1]
    dataset = EvalDataset(name, split="train", sample_ids=eval_ids)
    predict = make_predict_fn(model, device, use_amp)
    per_image = {k: [] for k in taus}

    for i in range(len(dataset)):
        item = dataset[i]
        prob_scaled = sliding_window_predict(predict, item["image"][0].numpy(), tile=tile, overlap=overlap)
        prob_native = upsample_to_native(prob_scaled, item["native_shape"])

        for label, t in taus.items():
            per_image[label].append(dice_coefficient(prob_native, item["label_native"],
                                                      roi=item["fov_native"], threshold=float(t)))

    return {k: float(np.mean(v)) for k, v in per_image.items()}


def main():
    parser = argparse.ArgumentParser(description="Label-free offset estimators vs oracle (FCA-v2).")
    parser.add_argument("--regret", type=str, default=DEFAULT_REGRET)
    parser.add_argument("--oracle-json", type=str,
                        default="result/mechanism_fca_offset_7targets/oracle_sweep_2026-07-20/metric.json")
    parser.add_argument("--targets", type=str, default=",".join(TARGET_DOMAINS))
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    device = torch.device(args.device)
    run_dir = Path("result/mechanism_fca_offset_7targets") / f"estimators_{args.date}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model, seed = load_care_model(args.regret, device)
    oracle = json.load(open(args.oracle_json, encoding="utf-8"))["results"]["care_regret"]
    pi_s = source_vessel_prevalence()
    print(f"[fca-v2] source vessel prevalence pi_S = {pi_s:.4f}", flush=True)

    config = {
        "experiment": "FCA-v2 label-free per-domain offset estimators vs oracle",
        "estimators": {"prior_match": "tau s.t. target fg-fraction == pi_S (baseline)",
                       "style_stability": "argmax_tau E pairwise Dice over K photometric views, banded"},
        "model": args.regret,
        "seed": seed,
        "pi_S_source_vessel_prevalence": pi_s,
        "k_views": K_VIEWS,
        "n_est_images": N_EST,
        "plausibility_band": list(BAND),
        "estimation_data": "unlabeled adapt pool (never scored); scoring on held-out eval set",
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    results = {}
    agg = {k: [] for k in ["fixed", "prior", "stability", "kappa", "oracle"]}

    for name in targets:
        tau_stab, tau_kappa, tau_prior, stab_curve, kappa_curve, fg_curve, n_est = estimate_thresholds(
            model, name, device, args.tile, args.overlap, args.amp, pi_s)
        tau_oracle = float(oracle[name]["oracle_threshold"])
        taus = {"fixed": 0.5, "prior": tau_prior, "stability": tau_stab, "kappa": tau_kappa, "oracle": tau_oracle}
        dice = score_eval(model, name, device, args.tile, args.overlap, args.amp, taus)
        results[name] = {"tau": taus, "eval_dice": dice, "stability_curve": stab_curve,
                         "kappa_curve": kappa_curve, "fgfrac_curve": fg_curve, "n_est": n_est}

        for k in agg:
            agg[k].append(dice[k])

        print(f"  {name:10s} tau[prior {tau_prior:.2f} stab {tau_stab:.2f} kappa {tau_kappa:.2f} "
              f"oracle {tau_oracle:.2f}]  Dice[0.5 {dice['fixed']:.4f} prior {dice['prior']:.4f} "
              f"stab {dice['stability']:.4f} kappa {dice['kappa']:.4f} oracle {dice['oracle']:.4f}]  "
              f"kappa-gain {dice['kappa']-dice['fixed']:+.4f}", flush=True)

    mean7 = {k: float(np.mean(v)) for k, v in agg.items()}
    results["_mean7"] = mean7

    def frac(label):

        return ((mean7[label] - mean7["fixed"]) / (mean7["oracle"] - mean7["fixed"])
                if mean7["oracle"] > mean7["fixed"] else float("nan"))

    print(f"\n  MEAN7 Dice[0.5 {mean7['fixed']:.4f} prior {mean7['prior']:.4f} stab {mean7['stability']:.4f} "
          f"kappa {mean7['kappa']:.4f} oracle {mean7['oracle']:.4f}]", flush=True)
    print(f"  oracle recovered: prior {100*frac('prior'):.0f}% | stab {100*frac('stability'):.0f}% | "
          f"kappa {100*frac('kappa'):.0f}%", flush=True)

    with open(run_dir / "metric.json", "w", encoding="utf-8") as handle:
        json.dump({"config": config, "results": results, "mean7": mean7,
                   "oracle_recovered": {k: frac(k) for k in ["prior", "stability", "kappa"]}},
                  handle, indent=2, ensure_ascii=False)

    print(f"\n[fca-v2] done. artifacts under {run_dir}", flush=True)


if __name__ == "__main__":
    main()
