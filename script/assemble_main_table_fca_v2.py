import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GROUPS = {"color_fundus": ["FIVES", "CHASEDB1", "STARE_AH"], "slo": ["IOSTAR"],
          "fluorescein_angiography": ["JRCFA"], "autofluorescence": ["JRCFAF"], "infrared_reflectance": ["JRCIR"]}
ORDER = ["FIVES", "CHASEDB1", "STARE_AH", "IOSTAR", "JRCFA", "JRCFAF", "JRCIR"]
METRICS = ["dice", "cldice", "iou"]


def summarize(per_domain, key):
    group = {g: float(np.mean([per_domain[d][key] for d in mem])) for g, mem in GROUPS.items()}
    vals = list(group.values())

    return {"mean": float(np.mean([per_domain[d][key] for d in ORDER])),
            "worst": float(min(vals)), "spread": float(max(vals) - min(vals))}


def main():
    parser = argparse.ArgumentParser(description="Assemble FCA-v2 main table.")
    parser.add_argument("--m1", type=str, default="result/main_results_7targets/main_table_2026-07-20/main_table.json")
    parser.add_argument("--source-only", type=str,
                        default="result/baselines_7targets/vanilla_unet_source_only_2026-07-25/metric.json",
                        help="B-1 lower bound: vanilla U-Net, no adaptation and no harmonization.")
    parser.add_argument("--upl", type=str, default="result/baselines_7targets/upl_sfda_native_2026-07-20/metric.json")
    parser.add_argument("--cbmt", type=str, default="result/baselines_7targets/sfda_cbmt_native_2026-07-21/metric.json")
    parser.add_argument("--prosfda", type=str, default="result/baselines_7targets/prosfda_native_2026-07-21/metric.json")
    parser.add_argument("--bateson", type=str, default="result/baselines_7targets/sfda_bateson_native_2026-07-21/metric.json")
    parser.add_argument("--dpl", type=str, default="result/baselines_7targets/sfda_dpl_native_2026-07-22/metric.json")
    parser.add_argument("--fcav2", type=str,
                        default="result/mechanism_fca_offset_7targets/fca_v2_eval_2026-07-26-suite-regret/metric.json")
    parser.add_argument("--care-key", type=str, default="per_domain_offset",
                        choices=["per_domain_offset", "per_domain_fixed"],
                        help="Full method (offset applied, ablation's last row) or the offset-free variant.")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))

    args = parser.parse_args()
    m1 = json.load(open(args.m1, encoding="utf-8"))["rows"]
    source_only = json.load(open(args.source_only, encoding="utf-8"))["per_domain"]
    upl = json.load(open(args.upl, encoding="utf-8"))["per_domain"]
    cbmt = json.load(open(args.cbmt, encoding="utf-8"))["per_domain"]
    prosfda = json.load(open(args.prosfda, encoding="utf-8"))["per_domain"]
    bateson = json.load(open(args.bateson, encoding="utf-8"))["per_domain"]
    dpl = json.load(open(args.dpl, encoding="utf-8"))["per_domain"]
    fcav2 = json.load(open(args.fcav2, encoding="utf-8"))

    rows = {
        "Source-only": {d: {k: source_only[d][k] for k in METRICS} for d in ORDER},
        "TENT": m1["TENT"]["per_domain"],
        "AIF-SFDA": m1["AIF-SFDA"]["per_domain"],
        "SFDA-CBMT": {d: {k: cbmt[d][k] for k in METRICS} for d in ORDER},
        "UPL-SFDA": {d: {k: upl[d][k] for k in METRICS} for d in ORDER},
        "ProSFDA": {d: {k: prosfda[d][k] for k in METRICS} for d in ORDER},
        "Bateson-SFDA": {d: {k: bateson[d][k] for k in METRICS} for d in ORDER},
        "SFDA-DPL": {d: {k: dpl[d][k] for k in METRICS} for d in ORDER},
        "CARE-SFDA (ours)": fcav2[args.care_key],
    }
    summaries = {name: {k: summarize(pd, k) for k in METRICS} for name, pd in rows.items()}

    run_dir = Path("result/main_results_7targets") / f"main_table_{args.date}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "M-1/M-3 main table; CARE row = the full method (harmonization + regret-equalized "
                      "fairness + class-anchored decision offset)",
        "sources": {"baselines_and_TENT_AIF": args.m1, "source_only": args.source_only, "UPL": args.upl,
                    "CARE_full": args.fcav2},
        "care_row": args.care_key,
        "note": "CARE row reads per_domain_offset of the FCA evaluation of the ablation-suite regret "
                "checkpoint, so the headline table and the ablation's last row are one configuration on "
                "one checkpoint. The Source-only row is the B-1 vanilla U-Net lower bound: same "
                "architecture and source split, no adaptation and no harmonization.",
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    lines = ["# Main table (M-1/M-3)\n",
             "Per-domain Dice / clDice / IoU; mean7 / worst-modality / spread. CARE = regret-noalign (threshold 0.5).\n"]

    for metric in METRICS:
        lines.append(f"\n## {metric}\n")
        header = "| method | " + " | ".join(ORDER) + " | mean | worst | spread |"
        lines.append(header)
        lines.append("|" + "---|" * (len(ORDER) + 4))

        for name, pd in rows.items():
            cells = " | ".join(f"{pd[d][metric]:.3f}" for d in ORDER)
            s = summaries[name][metric]
            lines.append(f"| {name} | {cells} | **{s['mean']:.3f}** | {s['worst']:.3f} | {s['spread']:.3f} |")

    md = "\n".join(lines) + "\n"

    with open(run_dir / "main_table.md", "w", encoding="utf-8") as handle:
        handle.write(md)

    with open(run_dir / "main_table.json", "w", encoding="utf-8") as handle:
        json.dump({"config": config, "rows": rows, "summaries": summaries}, handle, indent=2, ensure_ascii=False)

    print(md)
    print(f"[main-table-fca-v2] done. artifacts under {run_dir}")


if __name__ == "__main__":
    main()
