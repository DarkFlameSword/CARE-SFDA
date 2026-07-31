import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SOURCE_RUNNER = "script/train_source_anchor.py"
ADAPT_RUNNER = "script/multi_target_adapt.py"
SOURCE_THEME = "result/stage_a_source_driven_to_7targets"
ADAPT_THEME = "result/multi_target_adapt_7targets"
DEPLOYED_ANCHOR = "result/stage_a_source_driven_to_7targets/robust_2026-07-09/checkpoint/best_model.pt"

SOURCE_SHARED = ["--arm", "robust", "--iters", "6000", "--eval-interval", "500", "--seed", "0", "--amp"]
ADAPT_SHARED = ["--sampling-temperature", "1.0", "--iters", "2000", "--eval-interval", "200",
                "--seed", "0", "--variance-gate", "--fairness", "uniform"]

ARMS = [
    ("fdhfull", [], True,
     "every axis on: the reference this block is read against"),
    ("nocaliber", ["--no-fdh-caliber"], True,
     "no resampling to the common FOV caliber"),
    ("nozscore", ["--fdh-norm", "unit"], True,
     "no per-image FOV z-score, plain unit scaling instead"),
    ("luminance", ["--fdh-channel", "luminance"], True,
     "luminance instead of the green working channel"),
    ("nopolarity", ["--no-fdh-polarity"], False,
     "no per-domain polarity sign, so the angiography inversion is left undone"),
]


def adapt_note(note):
    return f"FDH1_{note}_tau10_uniform"


def find_done(theme, pattern, marker):
    finished = [d for d in sorted(Path(theme).glob(pattern)) if (d / marker).exists()]

    return finished[-1] if finished else None


def main():
    parser = argparse.ArgumentParser(description="Per-axis stage-1 harmonization suite.")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    for index, (note, axis_flags, retrain_source, purpose) in enumerate(ARMS, start=1):
        print(f"\n[fdh-suite] {index}/{len(ARMS)} {note} -- {purpose}", flush=True)
        anchor = DEPLOYED_ANCHOR

        if retrain_source:
            source_dir = find_done(SOURCE_THEME, f"robust_{note}_*", "checkpoint/best_model.pt")

            if source_dir is None:
                command = [sys.executable, os.path.abspath(SOURCE_RUNNER), *SOURCE_SHARED, *axis_flags,
                           "--note", note, "--device", args.device]
                subprocess.run(command, check=True)
                source_dir = find_done(SOURCE_THEME, f"robust_{note}_*", "checkpoint/best_model.pt")

            else:
                print(f"[fdh-suite]   source already trained ({source_dir.name}), skipping", flush=True)

            anchor = str(source_dir / "checkpoint" / "best_model.pt")

        else:
            print("[fdh-suite]   axis does not touch source training, reusing the deployed anchor",
                  flush=True)

        adapt_dir = find_done(ADAPT_THEME, f"{adapt_note(note)}_*", "student_final.pt")

        if adapt_dir is not None:
            print(f"[fdh-suite]   adaptation already done ({adapt_dir.name}), skipping", flush=True)
            continue

        command = [sys.executable, os.path.abspath(ADAPT_RUNNER), *ADAPT_SHARED, *axis_flags,
                   "--anchor", anchor, "--note", adapt_note(note), "--device", args.device]
        subprocess.run(command, check=True)

    print(f"\n[fdh-suite] all {len(ARMS)} arms present", flush=True)


if __name__ == "__main__":
    main()
