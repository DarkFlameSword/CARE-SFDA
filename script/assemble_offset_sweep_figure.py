import argparse
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

DEFAULT_SWEEP = "result/mechanism_fca_offset_7targets/oracle_sweep_2026-07-26-residual/metric.json"
DEFAULT_OFFSET_EVAL = "result/mechanism_fca_offset_7targets/fca_v2_eval_2026-07-26-suite-regret/metric.json"
PANEL_ORDER = ["FIVES", "CHASEDB1", "STARE_AH", "IOSTAR", "JRCFA", "JRCFAF", "JRCIR"]
SHORT_LABEL = {"FIVES": "FIVES", "CHASEDB1": "CHASE", "STARE_AH": "STARE", "IOSTAR": "IOSTAR",
               "JRCFA": "JRCFA", "JRCFAF": "JRCFAF", "JRCIR": "JRCIR"}
COLORS = {"FIVES": "#4477AA", "CHASEDB1": "#66CCEE", "STARE_AH": "#228833", "IOSTAR": "#CCBB44",
          "JRCFA": "#EE6677", "JRCFAF": "#AA3377", "JRCIR": "#777777"}
SOURCE_OPERATING_POINT = 0.5
FIG_W_IN = 3.30
FIG_H_IN = 3.55
FONT_PT = 8
TICK_PT = 6.5


def curve_value(curve, threshold):
    grid = np.array([float(k) for k in curve])
    index = int(np.argmin(np.abs(grid - threshold)))
    key = list(curve)[index]

    return float(curve[key])


def derive_numbers(sweep, tau_table):
    care = sweep["results"]["care_regret"]
    source = sweep["results"]["source_only"]
    rows = {}

    for name in PANEL_ORDER:
        curve = care[name]["curve"]
        tau_clamped = float(tau_table[name]["tau_d"])
        tau_raw = float(tau_table[name]["tau_prior_raw"])
        dice_fixed = float(care[name]["dice_at_0.5"])
        dice_oracle = float(care[name]["oracle_dice"])
        rows[name] = {
            "tau_star_care": float(care[name]["oracle_threshold"]),
            "tau_star_source": float(source[name]["oracle_threshold"]),
            "tau_d_clamped": tau_clamped,
            "tau_prior_raw": tau_raw,
            "dice_at_source_operating_point": dice_fixed,
            "dice_at_tau_star": dice_oracle,
            "dice_at_tau_d": curve_value(curve, tau_clamped),
            "dice_at_tau_prior_raw": curve_value(curve, tau_raw),
            "oracle_gain": dice_oracle - dice_fixed,
            "estimator_gain": curve_value(curve, tau_clamped) - dice_fixed,
        }

    def mean_over(key):
        return float(np.mean([rows[n][key] for n in PANEL_ORDER]))

    oracle_gain = mean_over("dice_at_tau_star") - mean_over("dice_at_source_operating_point")
    estimator_gain = mean_over("dice_at_tau_d") - mean_over("dice_at_source_operating_point")
    summary = {
        "n_targets": len(PANEL_ORDER),
        "n_tau_star_non_negative_care": int(sum(rows[n]["tau_star_care"] >= SOURCE_OPERATING_POINT
                                                for n in PANEL_ORDER)),
        "n_tau_star_non_negative_source": int(sum(rows[n]["tau_star_source"] >= SOURCE_OPERATING_POINT
                                                  for n in PANEL_ORDER)),
        "mean7_dice_at_source_operating_point": mean_over("dice_at_source_operating_point"),
        "mean7_dice_at_tau_d": mean_over("dice_at_tau_d"),
        "mean7_dice_at_tau_prior_raw": mean_over("dice_at_tau_prior_raw"),
        "mean7_dice_at_tau_star": mean_over("dice_at_tau_star"),
        "mean7_oracle_gain": oracle_gain,
        "mean7_estimator_gain": estimator_gain,
        "fraction_of_oracle_gain_recovered": estimator_gain / oracle_gain if oracle_gain else float("nan"),
        "clamp_removal_cost_mean7_dice": mean_over("dice_at_tau_prior_raw") - mean_over("dice_at_tau_d"),
        "n_domains_harmed_by_estimator": int(sum(rows[n]["estimator_gain"] < -1e-6 for n in PANEL_ORDER)),
    }

    return rows, summary


def render(sweep, rows, out_path):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    figure, axes = plt.subplots(2, 1, figsize=(FIG_W_IN, FIG_H_IN), sharex=True)
    panels = [("care_regret", "(a) after adaptation", axes[0]), ("source_only", "(b) un-adapted source", axes[1])]

    for model_key, title, axis in panels:
        for name in PANEL_ORDER:
            curve = sweep["results"][model_key][name]["curve"]
            grid = np.array([float(k) for k in curve])
            values = np.array([float(v) for v in curve.values()])
            axis.plot(grid, values, color=COLORS[name], linewidth=0.9, label=SHORT_LABEL[name], zorder=3)
            tau_star = float(sweep["results"][model_key][name]["oracle_threshold"])
            axis.plot([tau_star], [curve_value(curve, tau_star)], marker="o", markersize=3.2,
                      color=COLORS[name], zorder=5)

            if model_key == "care_regret":
                tau_d = rows[name]["tau_d_clamped"]
                axis.plot([tau_d], [curve_value(curve, tau_d)], marker="o", markersize=4.2,
                          markerfacecolor="white", markeredgecolor=COLORS[name], markeredgewidth=0.9, zorder=6)

        axis.axvline(SOURCE_OPERATING_POINT, color="black", linestyle="--", linewidth=0.7, zorder=2)
        axis.set_title(title, fontsize=FONT_PT, pad=2)
        axis.set_ylabel("Dice", fontsize=FONT_PT)
        axis.tick_params(labelsize=TICK_PT)
        axis.grid(True, alpha=0.22, linewidth=0.5, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].text(SOURCE_OPERATING_POINT - 0.015, axes[0].get_ylim()[0] + 0.02, r"$\tau=0$",
                 fontsize=TICK_PT, ha="right", va="bottom")
    axes[1].set_xlabel(r"decision offset as probability threshold $\sigma(\tau)$", fontsize=FONT_PT)
    axes[1].set_xlim(0.02, 0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, fontsize=TICK_PT, frameon=False, ncol=4, loc="lower center",
                  handlelength=1.1, columnspacing=1.0, borderaxespad=0.0)
    figure.tight_layout(pad=0.3, rect=(0, 0.065, 1, 1))
    figure.savefig(out_path)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Offset-sweep figure and the ablation numbers it reports.")
    parser.add_argument("--sweep", type=str, default=DEFAULT_SWEEP)
    parser.add_argument("--offset-eval", type=str, default=DEFAULT_OFFSET_EVAL)
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))

    args = parser.parse_args()
    sweep = json.loads(Path(args.sweep).read_text(encoding="utf-8"))
    offset_eval = json.loads(Path(args.offset_eval).read_text(encoding="utf-8"))
    rows, summary = derive_numbers(sweep, offset_eval["tau"])
    run_dir = Path("result/mechanism_fca_offset_7targets") / f"offset_sweep_figure_{args.date}"
    (run_dir / "visualization").mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "Class-anchored decision offset: sign structure of tau*, oracle recovery of the "
                      "prior-anchored estimator, and the cost of dropping the one-sided clamp",
        "sources": {"oracle_sweep": args.sweep, "offset_eval": args.offset_eval},
        "checkpoints": sweep["config"]["models"],
        "reading": "every Dice is read off the sweep curve of the same evaluation pass; tau_d and "
                   "tau_prior_raw come from the label-free estimator on the unlabeled adapt pool",
        "source_operating_point": SOURCE_OPERATING_POINT,
        "figure": {"width_in": FIG_W_IN, "height_in": FIG_H_IN, "font_pt": FONT_PT,
                   "latex_width": "\\columnwidth", "format": "pdf (vector)"},
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    render(sweep, rows, run_dir / "visualization" / "offset_sweep.pdf")

    with open(run_dir / "metric.json", "w", encoding="utf-8") as handle:
        json.dump({"config": config, "per_domain": rows, "summary": summary}, handle, indent=2, ensure_ascii=False)

    print(f"{'domain':10s} {'tau*(care)':>10s} {'tau*(src)':>10s} {'tau_d':>7s} {'raw':>6s} "
          f"{'@0.5':>7s} {'@tau_d':>7s} {'@raw':>7s} {'@tau*':>7s}", flush=True)

    for name in PANEL_ORDER:
        row = rows[name]
        print(f"{name:10s} {row['tau_star_care']:10.2f} {row['tau_star_source']:10.2f} "
              f"{row['tau_d_clamped']:7.2f} {row['tau_prior_raw']:6.2f} "
              f"{row['dice_at_source_operating_point']:7.4f} {row['dice_at_tau_d']:7.4f} "
              f"{row['dice_at_tau_prior_raw']:7.4f} {row['dice_at_tau_star']:7.4f}", flush=True)

    print(f"\n  tau* >= source operating point: care {summary['n_tau_star_non_negative_care']}/7, "
          f"source {summary['n_tau_star_non_negative_source']}/7", flush=True)
    print(f"  mean7 Dice  @0.5 {summary['mean7_dice_at_source_operating_point']:.4f}  "
          f"@tau_d {summary['mean7_dice_at_tau_d']:.4f}  @raw {summary['mean7_dice_at_tau_prior_raw']:.4f}  "
          f"@tau* {summary['mean7_dice_at_tau_star']:.4f}", flush=True)
    print(f"  recovered fraction of the oracle gain: {summary['fraction_of_oracle_gain_recovered']:.3f} "
          f"({summary['mean7_estimator_gain']:+.4f} of {summary['mean7_oracle_gain']:+.4f})", flush=True)
    print(f"  clamp removal costs {summary['clamp_removal_cost_mean7_dice']:+.4f} mean Dice", flush=True)
    print(f"[offset-sweep] artifacts under {run_dir}", flush=True)


if __name__ == "__main__":
    main()
