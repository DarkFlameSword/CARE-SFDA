# CARE-SFDA

Source code for **CARE-SFDA**: Class-Anchored, Regret-Equalized Source-Free Domain Adaptation for
cross-modality retinal vessel segmentation. One source model, trained on colour-fundus DRIVE, is
adapted jointly to seven unlabelled targets spanning five imaging modalities.

## Layout

| Path | Contents |
| --- | --- |
| `dataset/` | Domain registry, feature-distribution harmonization, splits, FOV handling, overlap-tiled native-resolution inference |
| `adaptation/` | LoRA + GroupNorm trust region, per-domain objectives, two-level style interventions, Hedge fairness, class-anchored decision offset |
| `model/` | GroupNorm U-Net with the bias-free single-logit head |
| `loss/`, `util/`, `evaluation/` | Dice/BCE losses, Dice / clDice / IoU metrics, FOV-masked evaluator |
| `script/` | Training, adaptation, evaluation, ablation launchers and result assemblers |
| `baseline/` | Source-only bounds, TENT, and the six source-free competitors |

## Environment

```
python 3.9        cuda 12.8
pip install -r requirements.txt
```

## Data

Place the seven target datasets and the DRIVE source under `data/` at the repository root. Each
domain folder follows the layout declared in `dataset/domain_registry.py`:

```
data/
  DRIVEN/     train/{img,label,mask}
  FIVES/      train/{img,label}   test/{img,label}
  CHASEDB1/   {img,label,mask}
  IOSTAR/     {img,label,mask}
  JRCFA/      {img,label}
  JRCFAF/     {img,label}
  JRCIR/      {img,label}
  STARE_AH/   {img,label}
```

Labels of the seven targets are used for evaluation only. No target label enters adaptation or
model selection.

## Reproducing the main result

All runs write to `result/`; the adaptation step reads the checkpoint the first step produces.

**1. Source anchor** — supervised pre-training on the DRIVE training split.

```
python script/train_source_anchor.py --arm robust --iters 6000 --batch-size 2 --patch-size 512 --lr 1e-3 --seed 0
```

**2. Multi-target adaptation** — one model jointly adapted to all seven targets.

```
python script/multi_target_adapt.py \
    --anchor result/stage_a_source_driven_to_7targets/robust_<date>/checkpoint/best_model.pt \
    --fairness regret_residual --fairness-unit domain \
    --iters 2000 --sampling-temperature 1.0 --seed 0
```

Remaining hyper-parameters take the defaults, which are the values reported in the paper.

**3. Evaluation at the class-anchored operating point.**

```
python script/evaluate_with_offset.py --regret result/multi_target_adapt_7targets/<run>
```

`script/evaluate_checkpoint_on_targets.py` scores any checkpoint at the fixed 0.5 threshold, which
is the setting every baseline is evaluated under.

## Ablations

`script/run_ablation_suite.py` launches the module ablation, `script/run_fdh_stage1_suite.py` the
per-axis ablation of harmonization stage 1. Every arm must be produced by the same code state: a
numerically equivalent edit to the adaptation runner shifts the trajectory by about one seed
standard deviation. The `script/assemble_*.py` and `script/measure_*.py` modules turn the recorded
runs into the tables and mechanism quantities the paper reports.

## Baselines

`baseline/` holds the orchestration; the competitors' own networks and losses are imported verbatim
from their official releases, which are not redistributed here. Clone each into `model/` under the
directory name below before running the corresponding script:

| Directory | Upstream |
| --- | --- |
| `model/AIF-SFDA` | https://github.com/JingHuaMan/AIF-SFDA |
| `model/ProSFDA` | https://github.com/ShishuaiHu/ProSFDA |
| `model/SFDA` | https://github.com/mathilde-b/SFDA |
| `model/SFDA-CBMT` | https://github.com/lloongx/SFDA-CBMT |
| `model/SFDA-DPL` | https://github.com/cchen-cc/SFDA-DPL |
| `model/UPL-SFDA` | https://github.com/HiLab-git/UPL-SFDA |

`baseline/vanilla_unet_source_only.py` and `script/run_tent_baseline.py` are self-contained and need
none of the above.
