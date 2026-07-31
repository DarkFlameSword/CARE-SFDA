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

from adaptation.offset import estimate_domain_offset
from adaptation.offset import source_vessel_prevalence
from dataset.domain_registry import DOMAINS
from dataset.domain_registry import TARGET_DOMAINS
from dataset.retinal_vessel_dataset import EvalDataset
from dataset.splits import domain_split
from dataset.tiling import sliding_window_predict
from dataset.tiling import upsample_to_native
from evaluation.evaluator import make_predict_fn
from util.evaluate_metrics import evaluate as care_evaluate

from measure_offset_oracle import DEFAULT_REGRET
from measure_offset_oracle import load_care_model

GROUPS = {"color_fundus": ["FIVES", "CHASEDB1", "STARE_AH"], "slo": ["IOSTAR"],
          "fluorescein_angiography": ["JRCFA"], "autofluorescence": ["JRCFAF"], "infrared_reflectance": ["JRCIR"]}
METRICS = ["dice", "cldice", "iou"]


def eval_at_thresholds(model, name, device, tile, overlap, use_amp, taus, seed=0):
    eval_ids = domain_split(name, eval_fraction=0.30, seed=seed)[1]
    dataset = EvalDataset(name, split="train", sample_ids=eval_ids)
    predict = make_predict_fn(model, device, use_amp)
    accum = {lab: {m: [] for m in METRICS} for lab in taus}

    for i in range(len(dataset)):
        item = dataset[i]
        prob = sliding_window_predict(predict, item["image"][0].numpy(), tile=tile, overlap=overlap)
        prob_native = upsample_to_native(prob, item["native_shape"])

        for lab, t in taus.items():
            m = care_evaluate(prob_native, item["label_native"], roi=item["fov_native"], threshold=float(t))

            for k in METRICS:
                accum[lab][k].append(m[k])

    return {lab: {k: float(np.mean(v)) for k, v in mm.items()} for lab, mm in accum.items()}


def summarize(per_domain, targets, key):
    present = set(per_domain)
    group = {g: float(np.mean([per_domain[d][key] for d in mem if d in present]))
             for g, mem in GROUPS.items() if any(d in present for d in mem)}
    vals = list(group.values())

    return {"mean": float(np.mean([per_domain[d][key] for d in targets])),
            "worst": float(min(vals)), "spread": float(max(vals) - min(vals))}


def main():
    parser = argparse.ArgumentParser(description="FCA-v2 offset evaluation (full metric suite).")
    parser.add_argument("--regret", type=str, default=DEFAULT_REGRET)
    parser.add_argument("--targets", type=str, default=",".join(TARGET_DOMAINS))
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--n-est", type=int, default=40)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    device = torch.device(args.device)
    run_dir = Path("result/mechanism_fca_offset_7targets") / f"fca_v2_eval_{args.date}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model, seed = load_care_model(args.regret, device)
    pi_s = source_vessel_prevalence(seed=0)
    print(f"[fca-v2-eval] pi_S = {pi_s:.4f}", flush=True)

    config = {
        "experiment": "FCA-v2 production eval: one-sided prior-anchored offset, Dice/clDice/IoU",
        "method": "tau_d = max(0.5, prior-match to pi_S) on unlabeled adapt pool; score eval at 0.5 vs tau_d",
        "model": args.regret,
        "seed": seed,
        "splits": "adaptation pool and held-out eval ids at the checkpoint's own seed; pi_S stays "
                  "the seed-0 source constant declared in the implementation details",
        "pi_S": pi_s,
        "n_est": args.n_est,
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    per_domain_fixed = {}
    per_domain_offset = {}
    taus_used = {}

    for name in targets:
        tau_d, tau_prior = estimate_domain_offset(model, name, device, pi_s, tile=args.tile,
                                                  overlap=args.overlap, use_amp=args.amp,
                                                  n_est=args.n_est, seed=seed)
        taus_used[name] = {"tau_d": tau_d, "tau_prior_raw": tau_prior}
        scores = eval_at_thresholds(model, name, device, args.tile, args.overlap, args.amp,
                                    {"fixed": 0.5, "offset": tau_d}, seed=seed)
        per_domain_fixed[name] = scores["fixed"]
        per_domain_offset[name] = scores["offset"]
        f = scores["fixed"]
        o = scores["offset"]
        print(f"  {name:10s} tau_d {tau_d:.2f}  Dice {f['dice']:.4f}->{o['dice']:.4f} ({o['dice']-f['dice']:+.4f})  "
              f"clDice {f['cldice']:.4f}->{o['cldice']:.4f} ({o['cldice']-f['cldice']:+.4f})  "
              f"IoU {f['iou']:.4f}->{o['iou']:.4f} ({o['iou']-f['iou']:+.4f})", flush=True)

        payload = {"config": config, "tau": taus_used,
                   "per_domain_fixed": per_domain_fixed, "per_domain_offset": per_domain_offset}

        with open(run_dir / "metric.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    print("\n  === mean7 / worst-mod / spread ===", flush=True)

    for key in METRICS:
        sf = summarize(per_domain_fixed, targets, key)
        so = summarize(per_domain_offset, targets, key)
        print(f"  {key:7s} fixed[mean {sf['mean']:.4f} worst {sf['worst']:.4f} spread {sf['spread']:.4f}] -> "
              f"offset[mean {so['mean']:.4f} worst {so['worst']:.4f} spread {so['spread']:.4f}]  "
              f"dmean {so['mean']-sf['mean']:+.4f}", flush=True)

    payload["summary"] = {key: {"fixed": summarize(per_domain_fixed, targets, key),
                                "offset": summarize(per_domain_offset, targets, key)} for key in METRICS}

    with open(run_dir / "metric.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"\n[fca-v2-eval] done. artifacts under {run_dir}", flush=True)


if __name__ == "__main__":
    main()
