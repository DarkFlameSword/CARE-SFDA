import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.domain_registry import TARGET_DOMAINS
from script.multitarget_results import load_final
from script.multitarget_results import modality_groups
from script.multitarget_results import summarize

RUN_ROOT = Path("result/multi_target_adapt_7targets")
SEED0_RUN = "nativeF2_tau10_regret_residual_noalign_2026-07-25"
FCA_ROOT = Path("result/mechanism_fca_offset_7targets")
OFFSET_EVALS = {
    SEED0_RUN: FCA_ROOT / "fca_v2_eval_2026-07-26-suite-regret/metric.json",
    "S1_regret_residual_tau10_noalign_s1_2026-07-24": FCA_ROOT / "fca_v2_eval_2026-07-26-S1-s1-ownsplit/metric.json",
    "S1_regret_residual_tau10_noalign_s2_2026-07-24": FCA_ROOT / "fca_v2_eval_2026-07-26-S1-s2-ownsplit/metric.json",
    "S1_regret_residual_tau10_noalign_s3_2026-07-24": FCA_ROOT / "fca_v2_eval_2026-07-26-S1-s3-ownsplit/metric.json",
    "S1_regret_residual_tau10_noalign_s4_2026-07-24": FCA_ROOT / "fca_v2_eval_2026-07-26-S1-s4-ownsplit/metric.json",
}
SOURCE_REFERENCES = {
    "bound": "result/baselines_7targets/vanilla_unet_source_only_2026-07-25/metric.json",
    "anchor": "result/baselines_7targets/source_only_2026-07-16/metric.json",
}
DOMAIN_ORDER = ["FIVES", "CHASEDB1", "STARE_AH", "IOSTAR", "JRCFA", "JRCFAF", "JRCIR"]


def load_offset_final(run_name):
    payload = json.loads(Path(OFFSET_EVALS[run_name]).read_text(encoding="utf-8"))["per_domain_offset"]

    return {metric: {d: float(payload[d][metric]) for d in payload} for metric in ("dice", "cldice")}


def collect_runs():
    runs = [RUN_ROOT / SEED0_RUN] + sorted(RUN_ROOT.glob("S1_regret_residual_tau10_noalign_s*"))
    loaded = []
    seeds = []

    for run in runs:
        config, _ = load_final(run)
        loaded.append((config, load_offset_final(run.name)))
        seeds.append(config["seed"])

    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds among the collected runs: {seeds}")

    return loaded, seeds


def ci95(values):
    n = len(values)

    if n < 2:
        return 0.0

    return float(stats.t.ppf(0.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n))


def main():
    targets = list(TARGET_DOMAINS)
    groups = modality_groups(targets)
    loaded, seeds = collect_runs()
    n_seeds = len(loaded)

    per_domain = {"dice": {}, "cldice": {}}

    for metric in ("dice", "cldice"):

        for t in targets:
            vals = [final[metric][t] for _, final in loaded]
            per_domain[metric][t] = {"seed_mean": float(np.mean(vals)), "seed_std": float(np.std(vals, ddof=1) if n_seeds > 1 else 0.0),
                                     "ci95": ci95(vals), "seeds": vals}

    overall = {}

    for metric in ("dice", "cldice"):
        per_seed_summ = [summarize({t: final[metric][t] for t in targets}, groups) for _, final in loaded]
        overall[metric] = {}

        for key in ("mean", "group_mean", "worst", "spread"):
            vals = [s[key] for s in per_seed_summ]
            overall[metric][key] = {"seed_mean": float(np.mean(vals)), "ci95": ci95(vals)}

    sources = {name: json.load(open(path, encoding="utf-8"))["per_domain"]
               for name, path in SOURCE_REFERENCES.items()}
    significance = {}

    for metric in ("dice", "cldice"):
        care_by_domain = [per_domain[metric][t]["seed_mean"] for t in DOMAIN_ORDER]
        significance[metric] = {}

        for name, source_rows in sources.items():
            src_by_domain = [source_rows[t][metric] for t in DOMAIN_ORDER]
            diffs = np.array(care_by_domain) - np.array(src_by_domain)
            wil = stats.wilcoxon(care_by_domain, src_by_domain, method="exact")
            tt = stats.ttest_rel(care_by_domain, src_by_domain)
            significance[metric][name] = {
                "mean_diff_vs_source": float(diffs.mean()),
                "domains_care_gt_source": int((diffs > 0).sum()),
                "wilcoxon_stat": float(wil.statistic), "wilcoxon_p": float(wil.pvalue),
                "paired_t_stat": float(tt.statistic), "paired_t_p": float(tt.pvalue),
            }

    date = datetime.now().strftime("%Y-%m-%d")
    run_dir = Path("result/main_results_7targets") / f"seed_significance_{date}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "S-1 five-seed statistics + paired significance (CARE regret tau=1 vs source-only)",
        "runs": [SEED0_RUN] + [p.name for p in sorted(RUN_ROOT.glob("S1_regret_residual_tau10_noalign_s*"))],
        "seeds": seeds,
        "scoring": "class-anchored per-domain offset applied at inference (per_domain_offset of each "
                   "checkpoint's FCA evaluation), i.e. the configuration of the main table",
        "offset_evals": {run: str(path) for run, path in OFFSET_EVALS.items()},
        "source_references": SOURCE_REFERENCES,
        "purpose": "statistical rigor only (5-seed mean +-95% CI + paired test vs the source "
                   "references, n=7 domains); clDice is reported, not claimed as a contribution",
        "caveat": "seed 0 is the ablation-suite checkpoint and seeds 1-4 predate it by one code "
                  "state (about 0.003 mean Dice against a 0.008 seed spread)",
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    payload = {"config": config, "per_domain": per_domain, "overall": overall, "significance": significance}

    with open(run_dir / "seed_significance.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    lines = [f"# S-1 five-seed statistics + paired significance ({n_seeds} seeds: {sorted(seeds)})", "",
             "CARE regret, tau=1, scored with the class-anchored per-domain offset. Per-domain "
             "seed-mean +- 95% CI; paired tests vs the bound (B-1 vanilla) and vs the harmonized "
             "source anchor (B-1r), n=7 domains.", ""]

    for metric in ("cldice", "dice"):
        lines += [f"## {metric.upper()} (per domain)", "",
                  "| domain | bound (B-1) | anchor (B-1r) | CARE seed-mean | 95% CI | seeds |",
                  "|---|---|---|---|---|---|"]

        for t in DOMAIN_ORDER:
            pd = per_domain[metric][t]
            seedstr = ", ".join(f"{v:.3f}" for v in pd["seeds"])
            lines.append(f"| {t} | {sources['bound'][t][metric]:.4f} | {sources['anchor'][t][metric]:.4f} | "
                         f"{pd['seed_mean']:.4f} | ±{pd['ci95']:.4f} | {seedstr} |")

        ov = overall[metric]
        lines += ["",
                  f"- Overall mean(7) {ov['mean']['seed_mean']:.4f} ±{ov['mean']['ci95']:.4f}; "
                  f"worst-mod {ov['worst']['seed_mean']:.4f} ±{ov['worst']['ci95']:.4f}; "
                  f"spread {ov['spread']['seed_mean']:.4f} ±{ov['spread']['ci95']:.4f}."]

        for name in SOURCE_REFERENCES:
            sig = significance[metric][name]
            lines.append(f"- vs {name}: mean diff {sig['mean_diff_vs_source']:+.4f}, "
                         f"CARE>reference on {sig['domains_care_gt_source']}/7 domains; "
                         f"Wilcoxon p={sig['wilcoxon_p']:.4f}, paired-t p={sig['paired_t_p']:.4f}.")

        lines.append("")

    with open(run_dir / "seed_significance.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print("\n".join(lines))
    print(f"[S-1] artifacts under {run_dir}")


if __name__ == "__main__":
    main()
