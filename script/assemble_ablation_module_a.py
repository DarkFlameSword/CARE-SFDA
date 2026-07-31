import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.domain_registry import TARGET_DOMAINS
from script.multitarget_results import load_final
from script.multitarget_results import modality_groups
from script.multitarget_results import summarize

RUN_ROOT = Path("result/multi_target_adapt_7targets")

CONDITIONS = {
    "full": {"label": "Full (§3.5 + L_inv)",
             "runs": ["nativeF2_tau10_uniform_noalign_2026-07-23", "nativeF2_tau10_uniform_noalign_s1_2026-07-24"]},
    "noL2": {"label": "-§3.5 Level-2 (MixStyle+Fourier off)",
             "runs": ["A1_noL2_tau10_uniform_noalign_s0_2026-07-24", "A1_noL2_tau10_uniform_noalign_s1_2026-07-24"]},
    "noInv": {"label": "-L_inv (style-invariant consistency off)",
              "runs": ["A1_noinv_tau10_uniform_noalign_s0_2026-07-24", "A1_noinv_tau10_uniform_noalign_s1_2026-07-24"]},
}

SUMMARY_KEYS = ["mean", "group_mean", "worst", "spread"]
KEY_LABEL = {"mean": "mean(7-dom)", "group_mean": "mean(5-mod)", "worst": "worst-mod", "spread": "spread"}


def aggregate_condition(runs, groups):
    seeds = []
    dice_by_seed = []
    cldice_by_seed = []

    for run in runs:
        config, final = load_final(RUN_ROOT / run)
        seeds.append(config["seed"])
        dice_by_seed.append(final["dice"])
        cldice_by_seed.append(final["cldice"])

    domains = list(dice_by_seed[0].keys())
    dice_avg = {t: float(np.mean([s[t] for s in dice_by_seed])) for t in domains}
    cldice_avg = {t: float(np.mean([s[t] for s in cldice_by_seed])) for t in domains}
    out = {"seeds": seeds, "per_domain_dice": dice_avg, "per_domain_cldice": cldice_avg}

    for metric, by_seed in (("dice", dice_by_seed), ("cldice", cldice_by_seed)):
        per_seed = [summarize(s, groups) for s in by_seed]
        stats = {}

        for key in SUMMARY_KEYS:
            values = [ps[key] for ps in per_seed]
            stats[key] = {"seed_mean": float(np.mean(values)), "seed_std": float(np.std(values))}

        out[metric] = {"per_seed": per_seed, "stats": stats}

    return out


def render_markdown(results):
    lines = ["# A-1 Module-A (style-invariant representation) ablation", "",
             "Seed-averaged (2 seeds) final-iteration (@2000) scores; value = seed-mean (seed-std).",
             "Aggregates over the 5 modality groups (color-fundus merged); mean(7-dom) over all targets.", ""]

    for metric in ("dice", "cldice"):
        lines.append(f"## {metric.upper()}")
        lines.append("")
        header = "| condition | " + " | ".join(KEY_LABEL[k] for k in SUMMARY_KEYS) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(SUMMARY_KEYS) + 1))

        for cond in CONDITIONS:
            stats = results[cond][metric]["stats"]
            cells = [f"{stats[k]['seed_mean']:.4f} ({stats[k]['seed_std']:.4f})" for k in SUMMARY_KEYS]
            lines.append(f"| {CONDITIONS[cond]['label']} | " + " | ".join(cells) + " |")

        lines.append("")
        full = results["full"][metric]["stats"]
        lines.append(f"**Delta (Full - knockout)** on worst-mod / mean(7):")

        for cond in ("noL2", "noInv"):
            ko = results[cond][metric]["stats"]
            d_worst = full["worst"]["seed_mean"] - ko["worst"]["seed_mean"]
            d_mean = full["mean"]["seed_mean"] - ko["mean"]["seed_mean"]
            lines.append(f"- vs {cond}: worst-mod {d_worst:+.4f}, mean(7) {d_mean:+.4f}")

        lines.append("")

    return "\n".join(lines)


def plot_ablation(results, out_path):
    conds = list(CONDITIONS.keys())
    bar_keys = ["mean", "group_mean", "worst"]
    colors = {"mean": "#4477AA", "group_mean": "#66CCEE", "worst": "#EE6677"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    for axis, metric in zip(axes, ("dice", "cldice")):
        x = np.arange(len(conds))
        width = 0.26

        for j, key in enumerate(bar_keys):
            means = [results[c][metric]["stats"][key]["seed_mean"] for c in conds]
            stds = [results[c][metric]["stats"][key]["seed_std"] for c in conds]
            axis.bar(x + (j - 1) * width, means, width, yerr=stds, capsize=3,
                     color=colors[key], label=KEY_LABEL[key])

        axis.set_xticks(x)
        axis.set_xticklabels([CONDITIONS[c]["label"] for c in conds], rotation=12, ha="right", fontsize=7)
        axis.set_ylabel(metric.upper())
        axis.set_ylim(0.60, 0.85 if metric == "cldice" else 0.78)
        axis.set_title(f"Module-A ablation ({metric.upper()})")
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def main():
    targets = list(TARGET_DOMAINS)
    groups = modality_groups(targets)
    results = {cond: aggregate_condition(CONDITIONS[cond]["runs"], groups) for cond in CONDITIONS}

    date = datetime.now().strftime("%Y-%m-%d")
    run_dir = Path("result/ablation_7targets") / f"module_a_style_invariance_{date}"
    (run_dir / "visualization").mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "A-1 module-A style-invariance ablation (assembly, no training)",
        "source_runs": {c: CONDITIONS[c]["runs"] for c in CONDITIONS},
        "conditions": {c: CONDITIONS[c]["label"] for c in CONDITIONS},
        "targets": targets,
        "modality_groups": groups,
        "aggregation": "seed-mean(seed-std) of per-seed summary; per-domain = seed mean; @2000 full eval",
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    with open(run_dir / "module_a_ablation.json", "w", encoding="utf-8") as handle:
        json.dump({"config": config, "results": results}, handle, indent=2, ensure_ascii=False)

    report = render_markdown(results)

    with open(run_dir / "module_a_ablation.md", "w", encoding="utf-8") as handle:
        handle.write(report)

    plot_ablation(results, run_dir / "visualization" / "module_a_ablation.png")

    print(report)
    print(f"\n[A-1] artifacts under {run_dir}")


if __name__ == "__main__":
    main()
