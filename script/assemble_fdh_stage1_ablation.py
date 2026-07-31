import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ADAPT_THEME = "result/multi_target_adapt_7targets"
SOURCE_THEME = "result/stage_a_source_driven_to_7targets"
ORDER = ["FIVES", "CHASEDB1", "STARE_AH", "IOSTAR", "JRCFA", "JRCFAF", "JRCIR"]
GROUPS = {"color_fundus": ["FIVES", "CHASEDB1", "STARE_AH"], "slo": ["IOSTAR"],
          "fluorescein_angiography": ["JRCFA"], "autofluorescence": ["JRCFAF"],
          "infrared_reflectance": ["JRCIR"]}
ARMS = [
    ("control", "Deployed control (all axes on)",
     "nativeF2_tau10_uniform_noalign_2026-07-25", "robust_2026-07-09"),
    ("fdhfull", "Re-trained reference (all axes on)",
     "FDH1_fdhfull_tau10_uniform_*", "robust_fdhfull_*"),
    ("nocaliber", "No caliber resampling",
     "FDH1_nocaliber_tau10_uniform_*", "robust_nocaliber_*"),
    ("nozscore", "No FOV z-score",
     "FDH1_nozscore_tau10_uniform_*", "robust_nozscore_*"),
    ("luminance", "Luminance working channel",
     "FDH1_luminance_tau10_uniform_*", "robust_luminance_*"),
    ("nopolarity", "No polarity sign",
     "FDH1_nopolarity_tau10_uniform_*", None),
]


def newest(theme, pattern):
    hits = sorted(glob.glob(str(Path(theme) / pattern)))

    if not hits:
        raise FileNotFoundError(f"No run under {theme} matching {pattern}")

    return Path(hits[-1])


def summarize(per_domain):
    group = [float(np.mean([per_domain[d] for d in members])) for members in GROUPS.values()]

    return {"mean": float(np.mean([per_domain[d] for d in ORDER])),
            "worst": float(min(group)), "spread": float(max(group) - min(group))}


def main():
    parser = argparse.ArgumentParser(description="Assemble the stage-1 per-axis ablation.")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))

    args = parser.parse_args()
    run_dir = Path("result/ablation_7targets") / f"fdh_stage1_{args.date}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    sources = {}

    for key, label, adapt_glob, source_glob in ARMS:
        adapt_dir = newest(ADAPT_THEME, adapt_glob)
        final = json.loads((adapt_dir / "metric.json").read_text(encoding="utf-8"))["trajectory"][-1]
        rows[key] = {
            "label": label,
            "adapt_run": adapt_dir.name,
            "dice": {d: float(final["dice"][d]) for d in ORDER},
            "cldice": {d: float(final["cldice"][d]) for d in ORDER},
            "summary": {"dice": summarize(final["dice"]), "cldice": summarize(final["cldice"])},
        }

        if source_glob is not None:
            source_dir = newest(SOURCE_THEME, source_glob)
            payload = json.loads((source_dir / "metric.json").read_text(encoding="utf-8"))
            sources[key] = {"run": source_dir.name, "best_val_dice": float(payload["best"]["val_dice"]),
                            "best_iter": int(payload["best"]["iter"])}

        else:
            sources[key] = {"run": sources["control"]["run"], "best_val_dice": sources["control"]["best_val_dice"],
                            "best_iter": sources["control"]["best_iter"],
                            "note": "axis does not touch source training; the deployed anchor is reused"}

    config = {
        "experiment": "Per-axis ablation of harmonization stage 1 (deterministic alignment)",
        "design": "each axis is removed from the front-end shared by source pre-training and "
                  "adaptation, so the arm re-trains the source and then adapts on top of it; the "
                  "polarity axis leaves source training untouched and reuses the deployed anchor",
        "adaptation_base": "sampling temperature 1.0, 2000 iterations, seed 0, uniform fairness, "
                           "variance gate pinned on, threshold 0.5, full held-out evaluation",
        "arms": {key: {"label": label, "adapt_run": rows[key]["adapt_run"], "source_run": sources[key]["run"]}
                 for key, label, _, _ in ARMS},
        "seed_band_mean_dice": 0.008,
        "launcher": "script/run_fdh_stage1_suite.py",
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    lines = ["# Stage-1 per-axis ablation (adapted models)", "",
             "| arm | " + " | ".join(ORDER) + " | Dice | clDice | Worst | Spread | source-val |",
             "|" + "---|" * (len(ORDER) + 5)]

    for key, label, _, _ in ARMS:
        row = rows[key]
        cells = " | ".join(f"{row['dice'][d]:.3f}" for d in ORDER)
        summary = row["summary"]
        lines.append(f"| {label} | {cells} | {summary['dice']['mean']:.4f} | "
                     f"{summary['cldice']['mean']:.4f} | {summary['dice']['worst']:.4f} | "
                     f"{summary['dice']['spread']:.4f} | {sources[key]['best_val_dice']:.4f} |")

    reference = rows["fdhfull"]["summary"]["dice"]
    lines += ["", "Deltas against the re-trained reference (mean Dice / worst-modality / spread):", ""]

    for key, label, _, _ in ARMS:

        if key in ("control", "fdhfull"):
            continue

        summary = rows[key]["summary"]["dice"]
        lines.append(f"- {label}: {summary['mean'] - reference['mean']:+.4f} / "
                     f"{summary['worst'] - reference['worst']:+.4f} / "
                     f"{summary['spread'] - reference['spread']:+.4f}")

    (run_dir / "fdh_stage1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(run_dir / "metric.json", "w", encoding="utf-8") as handle:
        json.dump({"config": config, "rows": rows, "sources": sources}, handle, indent=2, ensure_ascii=False)

    print("\n".join(lines))
    print(f"\n[fdh-stage1] artifacts under {run_dir}")


if __name__ == "__main__":
    main()
