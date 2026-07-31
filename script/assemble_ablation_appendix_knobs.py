import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.domain_registry import TARGET_DOMAINS
from script.multitarget_results import load_final
from script.multitarget_results import modality_groups
from script.multitarget_results import summarize

RUN_ROOT = Path("result/multi_target_adapt_7targets")
CONDITIONS = {
    "base": {"label": "Base (encoder-LoRA, avg teacher)", "run": "nativeF2_tau10_uniform_noalign_2026-07-23"},
    "decoder_lora": {"label": "+ decoder-LoRA", "run": "A6_decoderlora_tau10_uniform_noalign_2026-07-24"},
    "clean_teacher": {"label": "clean-image teacher pseudo-label", "run": "A6_cleanteacher_tau10_uniform_noalign_2026-07-24"},
}
SUMMARY_KEYS = ["mean", "group_mean", "worst", "spread"]
KEY_LABEL = {"mean": "mean(7)", "group_mean": "mean(5-mod)", "worst": "worst-mod", "spread": "spread"}


def main():
    targets = list(TARGET_DOMAINS)
    groups = modality_groups(targets)
    summ = {}

    for cond in CONDITIONS:
        _, final = load_final(RUN_ROOT / CONDITIONS[cond]["run"])
        summ[cond] = {m: summarize({t: final[m][t] for t in targets}, groups) for m in ("dice", "cldice")}

    lines = ["# A-6 appendix robustness knobs (decoder-LoRA, clean-teacher pseudo-label)", "",
             "Single seed, tau=1 uniform, final @2000. Delta = knob - base.", ""]

    for metric in ("dice", "cldice"):
        lines += [f"## {metric.upper()}", "",
                  "| condition | " + " | ".join(KEY_LABEL[k] for k in SUMMARY_KEYS) + " | Δmean | Δworst |",
                  "|" + "---|" * (len(SUMMARY_KEYS) + 3)]

        for cond in CONDITIONS:
            s = summ[cond][metric]
            cells = [f"{s[k]:.4f}" for k in SUMMARY_KEYS]
            dmean = s["mean"] - summ["base"][metric]["mean"]
            dworst = s["worst"] - summ["base"][metric]["worst"]
            tail = "-- | --" if cond == "base" else f"{dmean:+.4f} | {dworst:+.4f}"
            lines.append(f"| {CONDITIONS[cond]['label']} | " + " | ".join(cells) + f" | {tail} |")

        lines.append("")

    date = datetime.now().strftime("%Y-%m-%d")
    run_dir = Path("result/ablation_7targets") / f"appendix_knobs_{date}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "A-6 appendix robustness knobs (decoder-LoRA, clean-teacher), assembly",
        "conditions": {c: CONDITIONS[c] for c in CONDITIONS},
        "targets": targets,
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    with open(run_dir / "appendix_knobs.json", "w", encoding="utf-8") as handle:
        json.dump({"config": config, "summary": summ}, handle, indent=2, ensure_ascii=False)

    with open(run_dir / "appendix_knobs.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print("\n".join(lines))
    print(f"[A-6] artifacts under {run_dir}")


if __name__ == "__main__":
    main()
