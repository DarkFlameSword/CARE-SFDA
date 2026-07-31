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


def parse_args():
    parser = argparse.ArgumentParser(description="Assemble the C-1/F-3 trust-region drift appendix figure.")
    parser.add_argument("--target", type=str, default="JRCFA")
    parser.add_argument("--glob", type=str, default="full_2026-07-17*",
                        help="Run-folder glob selecting this drift study's runs.")
    parser.add_argument("--result-root", type=str, default="result")

    return parser.parse_args()


def dw_of(record):
    for key in ("dw", "dw_norm"):

        if key in record:
            return record[key]

    return float("nan")


def collect(args):
    root = Path(args.result_root) / f"stage_c_adapt_{args.target}"
    runs = {}

    for folder in sorted(root.glob(args.glob)):
        config_path = folder / "config.yaml"
        metric_path = folder / "metric.json"

        if not (config_path.exists() and metric_path.exists()):
            continue

        config = yaml.safe_load(open(config_path, encoding="utf-8"))
        payload = json.load(open(metric_path, encoding="utf-8"))
        trajectory = payload["trajectory"]

        if not trajectory or trajectory[-1]["iter"] != config["iters"]:
            print(f"  [skip] {folder.name}: incomplete "
                  f"(last iter {trajectory[-1]['iter'] if trajectory else None} != {config['iters']})")
            continue

        if config.get("use_sep"):
            print(f"  [warn] {folder.name}: use_sep=true (L_sep-era run) — excluded from the clean study")
            continue

        runs[float(config["w_anc"])] = {
            "run": folder.name,
            "final_dw": dw_of(trajectory[-1]),
            "iters": config["iters"],
            "trajectory": [{"iter": r["iter"], "dw": dw_of(r)} for r in trajectory],
        }

    return dict(sorted(runs.items()))


def plot(runs, out_path):
    w5s = sorted(runs)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(w5s)))
    figure, axis = plt.subplots(figsize=(6.5, 4.6))

    for w5, color in zip(w5s, colors):
        traj = runs[w5]["trajectory"]
        axis.plot([r["iter"] for r in traj], [r["dw"] for r in traj], color=color, label=f"w5={w5:g}")

    axis.set_xlabel("iteration")
    axis.set_ylabel(r"$\|\Delta W\|$")
    axis.set_yscale("log")
    axis.set_title("Trust region bounds anchor drift\n"
                   r"(weak w5: $\|\Delta W\|$ diverges; tight w5: plateaus)")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def main():
    args = parse_args()
    runs = collect(args)

    if not runs:
        raise SystemExit("No complete clean runs found.")

    w5s = sorted(runs)
    date = datetime.now().strftime("%Y-%m-%d")
    run_dir = Path(args.result_root) / "mechanism_trust_region" / f"drift_{date}"
    (run_dir / "visualization").mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "C-1/F-3 trust-region anchor-drift appendix figure (Dice-free)",
        "target": args.target,
        "runs": {f"{w:g}": runs[w]["run"] for w in w5s},
        "base": "use_sep=false (L_sep removed from the method), iters=2000, variant=full, seed 0",
        "role": "APPENDIX drift mechanism only; w5 is a fixed §4 constant (1e-1), not selected here. "
                "No w5 sweep-selection and no Dice/clDice-vs-w5 correlation is presented "
                "(AAAI final-draft decision).",
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    sweep = {f"{w:g}": {"run": runs[w]["run"], "final_dw": runs[w]["final_dw"], "iters": runs[w]["iters"]}
             for w in w5s}

    with open(run_dir / "drift_data.json", "w", encoding="utf-8") as handle:
        json.dump({"config": config, "sweep": sweep}, handle, indent=2, ensure_ascii=False)

    plot(runs, run_dir / "visualization" / "trust_region_drift.png")

    lines = [f"# C-1/F-3 trust-region anchor drift ({args.target}, clean base: use_sep=false, 2000 it)", "",
             "Appendix drift-mechanism figure only. **w5 is a fixed §4 constant (1e-1, deliberately "
             "tight); it is not selected here.** No w5 sweep-selection and no Dice/clDice-vs-w5 "
             "correlation is presented (AAAI final-draft decision).", "",
             "| w5 | final ||dW|| |", "|---|---|"]

    for w in w5s:
        lines.append(f"| {w:g} | {runs[w]['final_dw']:.3f} |")

    lines += ["", "A weak anchor (small w5) lets ||dW|| grow without bound, so the teacher-student loop "
              "can drift self-consistently away from the source prior; the tight adopted anchor holds "
              "||dW|| near the source, which is what bounds how far a self-reinforcing error can travel "
              "(§3.4). The claim here is exactly this drift bound -- decoupled from segmentation accuracy."]

    with open(run_dir / "drift_report.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\n[C-1/F-3] drift figure under {run_dir}")


if __name__ == "__main__":
    main()
