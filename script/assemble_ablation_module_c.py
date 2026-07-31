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
    "uniform": {"label": "uniform (lambda=1/D)",
                "run": "nativeF2_tau10_uniform_noalign_2026-07-25"},
    "minmax": {"label": "min-max (abs surrogate)",
               "run": "nativeF2_tau10_minmax_noalign_2026-07-26"},
    "regret": {"label": "regret (content-residual)",
               "run": "nativeF2_tau10_regret_residual_noalign_2026-07-25"},
}

OBSOLETE_RUNS = ["F1fix_uniform_s0_2026-07-12", "F1fix_minmax_s0_2026-07-12",
                 "F1fix_regret_s0_2026-07-12"]

SUMMARY_KEYS = ["mean", "group_mean", "worst", "spread"]
KEY_LABEL = {"mean": "mean(7-dom)", "group_mean": "mean(5-mod)", "worst": "worst-mod", "spread": "spread"}


def load_run(run):
    with open(RUN_ROOT / run / "metric.json", encoding="utf-8") as handle:
        payload = json.load(handle)

    config = payload["config"]
    trajectory = payload["trajectory"]

    for key, expected in (("sampling_temperature", 1.0), ("use_sep", False), ("seed", 0),
                          ("hedge_beta", 0.01)):

        if config.get(key) != expected:
            raise ValueError(f"{run}: {key}={config.get(key)} != compliant {expected}")

    return config, trajectory[-1], trajectory


def lambda_table(finals):
    return {rule: finals[rule]["lambda"] for rule in CONDITIONS}


def render_markdown(summaries, lam, groups):
    targets = list(TARGET_DOMAINS)
    lines = ["# A-3 Module-C (regret-equalized fairness) three-way comparison", "",
             "Shared base: tau=1, per-domain unit, use_sep=false, hedge_beta=0.01, seed 0 (**n=1 seed**).",
             "Runs differ only in the `--fairness` rule. Per §3.10/§3.12 this is a **mechanism**",
             "claim (where lambda is allocated), not a worst-domain-accuracy claim.", "",
             "## Primary: per-domain lambda allocation (final @2000)", "",
             "| domain | modality | uniform | min-max | regret |",
             "|---|---|---|---|---|"]

    for t in targets:
        modality = next(g for g, members in groups.items() if t in members)
        cells = " | ".join(f"{lam[rule][t]:.3f}" for rule in ("uniform", "minmax", "regret"))
        lines.append(f"| {t} | {modality} | {cells} |")

    lines += ["", f"Reference weights: uniform 1/D = {1.0 / len(targets):.3f}; anti-starvation floor "
              "lambda_min = 0.020.", "",
              "**Mechanism read-out.** The content-residual signal leaves the vessel-dense, well-served "
              "large domain FIVES near its uniform weight -- its large surrogate is explained by content, "
              "not under-service, so the residual does not penalise it -- and instead down-weights the "
              "genuinely well-served SLO domain IOSTAR and concentrates on the domains whose surrogate "
              "exceeds what their content predicts (JRCFA, JRCFAF). Absolute min-max, by contrast, chases "
              "the noisiest content-dense surrogate (JRCFAF). This is the §3.10 point-of-departure from "
              "absolute worst-group DRO -- a lambda-allocation difference, with no worst-domain-accuracy "
              "claim attached.", "",
              "## Secondary (single-seed diagnostic, NOT an accuracy claim): Dice / clDice triplet", ""]

    for metric in ("dice", "cldice"):
        lines += [f"### {metric.upper()}", "",
                  "| rule | " + " | ".join(KEY_LABEL[k] for k in SUMMARY_KEYS) + " |",
                  "|" + "---|" * (len(SUMMARY_KEYS) + 1)]

        for rule in CONDITIONS:
            stats = summaries[rule][metric]
            cells = [f"{stats[k]:.4f}" for k in SUMMARY_KEYS]
            lines.append(f"| {CONDITIONS[rule]['label']} | " + " | ".join(cells) + " |")

        lines.append("")

    lines += ["> Obsolete (pre-L_sep-removal, tau=0; **not** module-C evidence): "
              + ", ".join(OBSOLETE_RUNS) + ".", ""]

    return "\n".join(lines)


def plot_lambda_allocation(lam, out_path):
    targets = list(TARGET_DOMAINS)
    rules = ["uniform", "minmax", "regret"]
    colors = {"uniform": "#BBBBBB", "minmax": "#EE6677", "regret": "#4477AA"}
    figure, axis = plt.subplots(figsize=(11, 5))
    x = np.arange(len(targets))
    width = 0.26

    for j, rule in enumerate(rules):
        heights = [lam[rule][t] for t in targets]
        axis.bar(x + (j - 1) * width, heights, width, color=colors[rule], label=CONDITIONS[rule]["label"])

    axis.axhline(1.0 / len(targets), color="black", linestyle="--", linewidth=1, alpha=0.7, label="uniform 1/D")
    axis.axhline(0.02, color="green", linestyle=":", linewidth=1, alpha=0.7, label="lambda_min floor")
    axis.set_xticks(x)
    axis.set_xticklabels(targets, rotation=15, ha="right", fontsize=8)
    axis.set_ylabel("final per-domain lambda")
    axis.set_title("Module-C: per-domain lambda allocation by fairness rule (@2000, seed 0)")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def plot_lambda_trajectory(trajectories, out_path):
    targets = list(TARGET_DOMAINS)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for axis, rule in zip(axes, ("regret", "minmax")):
        steps = [r["iter"] for r in trajectories[rule]]

        for t in targets:
            axis.plot(steps, [r["lambda"][t] for r in trajectories[rule]], marker=".", alpha=0.7, label=t)

        axis.axhline(1.0 / len(targets), color="black", linestyle="--", linewidth=1, alpha=0.6)
        axis.set_xlabel("iteration")
        axis.set_title(f"{CONDITIONS[rule]['label']} -- lambda trajectory")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("per-domain lambda")
    axes[1].legend(fontsize=7, ncol=2, loc="best")
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def main():
    targets = list(TARGET_DOMAINS)
    groups = modality_groups(targets)
    configs = {}
    finals = {}
    trajectories = {}

    for rule in CONDITIONS:
        config, final, trajectory = load_run(CONDITIONS[rule]["run"])
        configs[rule] = config
        finals[rule] = final
        trajectories[rule] = trajectory

    summaries = {rule: {"dice": summarize(finals[rule]["dice"], groups),
                        "cldice": summarize(finals[rule]["cldice"], groups)}
                 for rule in CONDITIONS}
    lam = lambda_table(finals)

    date = datetime.now().strftime("%Y-%m-%d")
    run_dir = Path("result/ablation_7targets") / f"module_c_fairness_{date}"
    (run_dir / "visualization").mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "A-3 module-C regret-equalized fairness three-way (assembly, no training)",
        "source_runs": {rule: CONDITIONS[rule]["run"] for rule in CONDITIONS},
        "obsolete_runs": OBSOLETE_RUNS,
        "conditions": {rule: CONDITIONS[rule]["label"] for rule in CONDITIONS},
        "base": "tau=1, per-domain unit, use_sep=false, hedge_beta=0.01, seed 0, n=1 seed",
        "primary_claim": "lambda-allocation mechanism (§3.10); NOT a worst-domain-accuracy claim (§3.12)",
        "targets": targets,
        "modality_groups": groups,
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    payload = {"config": config, "lambda_final": lam,
               "accuracy_triplet": summaries,
               "lambda_trajectory": {rule: [{"iter": r["iter"], "lambda": r["lambda"]}
                                            for r in trajectories[rule]] for rule in CONDITIONS}}

    with open(run_dir / "module_c_fairness.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    report = render_markdown(summaries, lam, groups)

    with open(run_dir / "module_c_fairness.md", "w", encoding="utf-8") as handle:
        handle.write(report)

    plot_lambda_allocation(lam, run_dir / "visualization" / "module_c_lambda_allocation.png")
    plot_lambda_trajectory(trajectories, run_dir / "visualization" / "module_c_lambda_trajectory.png")

    print(report)
    print(f"\n[A-3] artifacts under {run_dir}")


if __name__ == "__main__":
    main()
