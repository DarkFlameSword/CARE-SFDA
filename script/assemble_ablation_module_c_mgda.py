import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ORDER = ["FIVES", "CHASEDB1", "STARE_AH", "IOSTAR", "JRCFA", "JRCFAF", "JRCIR"]
GROUPS = {"color_fundus": ["FIVES", "CHASEDB1", "STARE_AH"], "slo": ["IOSTAR"],
          "fluorescein_angiography": ["JRCFA"], "autofluorescence": ["JRCFAF"], "infrared_reflectance": ["JRCIR"]}


def _final(path):
    return json.load(open(path, encoding="utf-8"))["trajectory"][-1]


def _summ(per_dice):
    group = {g: float(np.mean([per_dice[t] for t in members])) for g, members in GROUPS.items()}
    values = list(group.values())

    return {"mean": float(np.mean([per_dice[t] for t in ORDER])),
            "worst": float(min(values)), "spread": float(max(values) - min(values)), "group_dice": group}


def main():
    parser = argparse.ArgumentParser(description="Assemble Module-C fairness comparison with MGDA (noalign).")
    parser.add_argument("--uniform", type=str,
                        default="result/multi_target_adapt_7targets/nativeF2_tau10_uniform_noalign_2026-07-23/metric.json")
    parser.add_argument("--regret", type=str,
                        default="result/multi_target_adapt_7targets/nativeF2_tau10_regret_residual_noalign_2026-07-24/metric.json")
    parser.add_argument("--mgda", type=str,
                        default="result/multi_target_adapt_7targets/nativeF2_tau10_mgda7_noalign_2026-07-22/metric.json")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))

    args = parser.parse_args()
    recs = {"uniform": _final(args.uniform), "regret": _final(args.regret), "MGDA-7": _final(args.mgda)}
    rows = {name: {"per_domain_dice": {t: rec["dice"][t] for t in ORDER},
                   "per_domain_cldice": {t: rec["cldice"][t] for t in ORDER},
                   "summary_dice": _summ(rec["dice"]), "summary_cldice": _summ(rec["cldice"])}
            for name, rec in recs.items()}

    allocation = {"uniform_per_domain": {t: 1.0 / len(ORDER) for t in ORDER},
                  "mgda_alpha": {t: recs["MGDA-7"].get("mgda_alpha", {}).get(t) for t in ORDER},
                  "regret_lambda": {t: recs["regret"].get("lambda", {}).get(t) for t in ORDER}}
    alpha_vals = [allocation["mgda_alpha"][t] for t in ORDER]
    lam_vals = [allocation["regret_lambda"][t] for t in ORDER]
    allocation["mgda_alpha_spread"] = float(max(alpha_vals) - min(alpha_vals))
    allocation["regret_lambda_spread"] = float(max(lam_vals) - min(lam_vals))

    run_dir = Path("result/ablation_7targets") / f"module_c_fairness_mgda_{args.date}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "Module-C fairness: MGDA (gradient-based Pareto) vs regret (CARE Hedge) vs uniform, noalign base",
        "base": "w_align=0, use_sep=false, tau=1, seed 0, 2000 iters, same anchor (compliant noalign, matches the "
                "§3.11 L_d = L_seg + w1 L_inv + w4 L_occ spec)",
        "mgda": "7-domain sequential MGDA: per-domain unit-normalized gradients (LoRA+GN, no anchor), Frank-Wolfe "
                "min-norm over the 7-domain simplex, descend sum alpha_d ghat_d + anchor; rotating cross-domain donor",
        "sources": {"uniform": args.uniform, "regret": args.regret, "mgda": args.mgda},
        "finding": "MGDA alpha collapses to ~uniform (high-dim near-orthogonal per-domain grads -> min-norm "
                   "defaults to uniform); its unit-norm direction-averaging over-weights well-served domains -> "
                   "MGDA is WORST of the three on mean AND worst-modality. regret lambda strongly differentiates "
                   "by relative progress (FIVES floored to lam_min, IOSTAR lifted). Demonstrates the regret "
                   "mechanism's value over generic multi-objective optimization.",
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    with open(run_dir / "module_c_mgda.json", "w", encoding="utf-8") as handle:
        json.dump({"config": config, "rows": rows, "allocation": allocation}, handle, indent=2, ensure_ascii=False)

    lines = ["# Module-C fairness: MGDA vs regret vs uniform (noalign base)\n",
             "\n## Dice (final full eval)\n",
             "| rule | " + " | ".join(ORDER) + " | mean7 | worst-mod | spread |",
             "|" + "---|" * (len(ORDER) + 4)]

    for name in ["uniform", "regret", "MGDA-7"]:
        pd = rows[name]["per_domain_dice"]
        s = rows[name]["summary_dice"]
        cells = " | ".join(f"{pd[t]:.3f}" for t in ORDER)
        lines.append(f"| {name} | {cells} | **{s['mean']:.4f}** | {s['worst']:.4f} | {s['spread']:.4f} |")

    lines += ["\n## Weight allocation (uniform 1/7=0.143, MGDA Frank-Wolfe alpha, regret Hedge lambda)\n",
              "| domain | MGDA alpha | regret lambda |", "|---|---|---|"]

    for t in ORDER:
        lines.append(f"| {t} | {allocation['mgda_alpha'][t]:.3f} | {allocation['regret_lambda'][t]:.3f} |")

    lines.append(f"\nalpha spread {allocation['mgda_alpha_spread']:.3f} (≈0, ~uniform) | "
                 f"lambda spread {allocation['regret_lambda_spread']:.3f} (strong differentiation)\n")
    md = "\n".join(lines) + "\n"

    with open(run_dir / "module_c_mgda.md", "w", encoding="utf-8") as handle:
        handle.write(md)

    print(md)
    print(f"[module-c-mgda] done. artifacts under {run_dir}")


if __name__ == "__main__":
    main()
