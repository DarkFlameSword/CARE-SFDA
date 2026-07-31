import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RUNNER = "script/multi_target_adapt.py"
SHARED = ["--sampling-temperature", "1.0", "--iters", "2000", "--eval-interval", "200", "--seed", "0",
          "--variance-gate"]

ARMS = [
    ("nativeF2_tau10_uniform_noalign", ["--fairness", "uniform"],
     "control / module-table row M-b"),
    ("nativeF2_tau10_regret_residual_noalign", ["--fairness", "regret_residual"],
     "regret-equalized fairness / row M-c"),
    ("nativeF2_tau10_minmax_noalign", ["--fairness", "minmax"],
     "absolute worst-group reweighting, the A-3 contrast to the content-adjusted residual"),
    ("A1_nostyle_tau10_uniform_noalign",
     ["--fairness", "uniform", "--intervention-strength", "0", "--no-mixstyle", "--no-fourier",
      "--w-inv", "0"],
     "no style robustness at all / row M-a (compound knockout, see views_degenerate)"),
    ("A1_novargate_tau10_uniform_noalign", ["--fairness", "uniform", "--no-variance-gate"],
     "the reliability weight e^{-u}, since dropped from the method on this arm's own evidence"),
    ("A5_trainablehead_tau10_uniform_noalign", ["--fairness", "uniform", "--trainable-head"],
     "the frozen class anchor (§3.12 non-circularity control)"),
    ("A4_occoff_tau10_uniform_noalign", ["--fairness", "uniform", "--w-occ", "0"],
     "the §3.8 soft-skeleton occupancy barrier"),
    ("FDH_nostage2_tau10_uniform_noalign",
     ["--fairness", "uniform", "--intervention-strength", "0", "--no-mixstyle", "--no-fourier"],
     "harmonization stage 2 as a whole: both intervention levels off, L_inv kept"),
    ("FDH_noimagelevel_tau10_uniform_noalign",
     ["--fairness", "uniform", "--intervention-strength", "0"],
     "harmonization stage 2, image level only: photometric interventions off"),
    ("FDH_nofeaturelevel_tau10_uniform_noalign",
     ["--fairness", "uniform", "--no-mixstyle", "--no-fourier"],
     "harmonization stage 2, feature level only: MixStyle and Fourier off"),
]


def main():
    parser = argparse.ArgumentParser(description="Serial module-ablation suite.")
    parser.add_argument("--suite-dates", type=str, default=datetime.now().strftime("%Y-%m-%d"),
                        help="Comma-separated date suffixes that belong to THIS suite. Only these "
                             "count as done; a run from an earlier code state must not be reused, "
                             "and a suite that crosses midnight needs both dates listed.")
    parser.add_argument("--result-root", type=str, default="result")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    suite_dates = [d.strip() for d in args.suite_dates.split(",") if d.strip()]
    adapt_root = Path(args.result_root) / "multi_target_adapt_7targets"

    for index, (note, extra, purpose) in enumerate(ARMS, start=1):
        done = [adapt_root / f"{note}_{d}" for d in suite_dates
                if (adapt_root / f"{note}_{d}" / "student_final.pt").exists()]

        if done:
            print(f"[suite] {index}/{len(ARMS)} {note}: already done ({done[0].name}), skipping",
                  flush=True)
            continue

        print(f"[suite] {index}/{len(ARMS)} {note} -- {purpose}", flush=True)
        command = [sys.executable, os.path.abspath(RUNNER), *SHARED, *extra,
                   "--note", note, "--result-root", args.result_root, "--device", args.device]
        subprocess.run(command, check=True)

    print(f"[suite] all {len(ARMS)} arms present under {adapt_root}", flush=True)


if __name__ == "__main__":
    main()
