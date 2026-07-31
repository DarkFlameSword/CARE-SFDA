import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptation.objectives import invariance_loss
from adaptation.objectives import occupancy_loss
from adaptation.objectives import percentile_gate
from adaptation.objectives import prototype_sep_loss
from adaptation.objectives import soft_cldice_loss
from adaptation.objectives import soft_occupancy
from adaptation.peft import anchor_loss
from adaptation.peft import configure_peft
from adaptation.peft import trainable_parameters
from dataset.domain_registry import SOURCE_DOMAIN
from dataset.retinal_vessel_dataset import EvalDataset
from dataset.retinal_vessel_dataset import SourceDataset
from dataset.retinal_vessel_dataset import TargetAdaptDataset
from dataset.splits import domain_split
from evaluation.evaluator import Evaluator
from model.unet import UNet

DEFAULT_ANCHOR = "result/stage_a_source_driven_to_7targets/robust_2026-07-09/checkpoint/best_model.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="CARE-SFDA Stage-C single-target adaptation.")
    parser.add_argument("--variant", choices=["full", "no_occ", "no_sep", "stress1", "stress2", "prevent"],
                        default="full")
    parser.add_argument("--anchor", type=str, default=DEFAULT_ANCHOR)
    parser.add_argument("--target", type=str, default="CHASEDB1")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--k-views", type=int, default=2)
    parser.add_argument("--intervention-strength", type=float, default=1.0,
                        help="Photometric intervention strength for §3.5 views (1.0 = §3.3 source default).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--w-inv", type=float, default=1.0)
    parser.add_argument("--w-sep", type=float, default=0.1,
                        help="[DEPRECATED] L_sep prototype-separation barrier weight (off by default).")
    parser.add_argument("--w-occ", type=float, default=0.5)
    parser.add_argument("--w-anc", type=float, default=1e-1)
    parser.add_argument("--lambda-gn", type=float, default=1e-3)
    parser.add_argument("--eta-conn", type=float, default=0.5)
    parser.add_argument("--m0", type=float, default=1.0)
    parser.add_argument("--rho-vessel", type=float, default=0.4)
    parser.add_argument("--teacher-m", type=float, default=0.99)
    parser.add_argument("--eval-fraction", type=float, default=0.30, help="Per-target held-out eval fraction.")
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--adv-iters", type=int, default=40, help="all-bg pre-steps for stress2.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--result-root", type=str, default="result")
    return parser.parse_args()


def build_model(args):
    model = UNet(in_channels=1, n_classes=1, base_channels=args.base_channels, norm="group", head_bias=False)
    model.load_state_dict(torch.load(args.anchor, map_location="cpu", weights_only=False)["model"])
    model.to(args.device)
    peft = configure_peft(model, r=args.lora_rank, alpha=args.lora_alpha)

    return model, peft


def source_vessel_prototype(model, args, n_batches=4):
    dataset = SourceDataset(SOURCE_DOMAIN, split="train", patch_size=args.patch_size,
                            augment=False, photometric=False)
    stream = iter(DataLoader(dataset, batch_size=1, shuffle=True))
    accum = None
    denom = 0.0

    for _ in range(n_batches):
        batch = next(stream)
        image = batch["image"].to(args.device)
        weight = (batch["label"] * batch["fov"]).to(args.device).reshape(-1)

        with torch.no_grad():
            z = model.forward_features(image)

        c = z.shape[1]
        contrib = (z.permute(0, 2, 3, 1).reshape(-1, c) * weight.unsqueeze(1)).sum(0)
        accum = contrib if accum is None else accum + contrib
        denom = denom + float(weight.sum())

    return (accum / max(denom, 1e-6)).detach()


@torch.no_grad()
def ema_update(teacher, student, m):
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.mul_(m).add_(p_s, alpha=1.0 - m)


def teacher_target(teacher, views):
    probs = []

    for k in range(views.shape[1]):
        probs.append(torch.sigmoid(teacher(views[:, k])))

    stack = torch.stack(probs, dim=0)

    return stack.mean(0), stack.var(0, unbiased=False)


def monitors(student, teacher, clean, fov, gate, p_bar):
    with torch.no_grad():
        logit = student(clean)
        prob = torch.sigmoid(logit)
        m = fov
        fg_soft = float((prob * m).sum() / m.sum().clamp_min(1.0))
        fg_hard = float(((prob > 0.5).float() * m).sum() / m.sum().clamp_min(1.0))
        z = student.forward_features(clean)
        ves = (p_bar > 0.5)
        phi = torch.maximum(p_bar, 1.0 - p_bar)
        c = z.shape[1]
        zf = z.permute(0, 2, 3, 1).reshape(-1, c)
        w = phi.reshape(-1)
        wv = w * ves.reshape(-1).float()
        wb = w * (~ves).reshape(-1).float()
        mu_v = (zf * wv.unsqueeze(1)).sum(0) / wv.sum().clamp_min(1e-6)
        mu_b = (zf * wb.unsqueeze(1)).sum(0) / wb.sum().clamp_min(1e-6)
        protsep = float((mu_v - mu_b).norm())
        agree = float((((logit > 0) == ves) & (m > 0.5)).float().sum() / m.sum().clamp_min(1.0))

    return {"fg_soft": fg_soft, "fg_hard": fg_hard, "protsep": protsep, "agree": agree}


def adversarial_collapse(student, params, loader, args):
    opt = torch.optim.Adam(params, lr=1e-3)
    it = iter(loader)

    for _ in range(args.adv_iters):

        try:
            batch = next(it)

        except StopIteration:
            it = iter(loader)
            batch = next(it)

        clean = batch["image"].to(args.device)
        fov = batch["fov"].to(args.device)
        prob = torch.sigmoid(student(clean))
        loss = (prob * fov).sum() / fov.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_map = {"full": (True, False), "no_occ": (False, False), "no_sep": (True, False),
               "stress1": (False, False), "stress2": (True, False), "prevent": (True, False)}
    use_occ, use_sep = use_map[args.variant]

    adapt_ids, eval_ids = domain_split(args.target, eval_fraction=args.eval_fraction, seed=args.seed)
    dataset = TargetAdaptDataset(args.target, split="train", patch_size=args.patch_size,
                                 sample_ids=adapt_ids, n_views=args.k_views,
                                 intervention_strength=args.intervention_strength)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=2, persistent_workers=True)

    def infinite():

        while True:

            for b in loader:
                yield b

    stream = infinite()

    student, peft = build_model(args)
    teacher, _ = build_model(args)
    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad_(False)

    params = trainable_parameters(student)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    mu_v_src = source_vessel_prototype(student, args) if use_sep else None

    q_accum = []

    for _ in range(5):
        b = next(stream)

        with torch.no_grad():
            q_accum.append(float(soft_occupancy(torch.sigmoid(student(b["image"].to(args.device))),
                                                b["fov"].to(args.device)).item()))

    q_src = float(np.mean(q_accum))

    if args.variant in ("stress2", "prevent"):
        adversarial_collapse(student, params, loader, args)

        if args.variant == "stress2":
            teacher.load_state_dict(student.state_dict())

    date = datetime.now().strftime("%Y-%m-%d")
    run_dir = Path(args.result_root) / f"stage_c_adapt_{args.target}" / f"{args.variant}_{date}"

    if run_dir.exists():
        run_dir = run_dir.parent / f"{args.variant}_{date}_{datetime.now().strftime('%H%M%S')}"

    (run_dir / "visualization").mkdir(parents=True)
    config = {**vars(args), "q_src": q_src, "use_occ": use_occ, "use_sep": use_sep,
              "n_adapt": len(adapt_ids), "n_eval": len(eval_ids),
              "launch_timestamp": datetime.now().isoformat(timespec="seconds"), "code_commit_hash": "None"}

    with open(run_dir / "config.yaml", "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    evaluator = Evaluator(SOURCE_DOMAIN, [args.target], device=args.device, tile=args.patch_size,
                          use_amp=False, datasets={args.target: EvalDataset(args.target, split="train",
                                                                            sample_ids=eval_ids)})
    records = []
    carry = {"u_mean": float("nan")}

    def run_eval(step, mon):
        res = evaluator.evaluate(student)[args.target]

        with torch.no_grad():
            dw_norm = float(torch.sqrt(sum(l.delta_w_fro_sq() for l in peft["lora"])).item())

        rec = {"iter": step, "dice": res["dice"], "cldice": res["cldice"],
               "u_mean": carry["u_mean"], "dw_norm": dw_norm, **mon}
        records.append(rec)

        with open(run_dir / "metric.json", "w") as handle:
            json.dump({"config": config, "trajectory": records}, handle, indent=2)

        print(f"[stage_c:{args.variant}] iter {step}: dice {res['dice']:.4f} cldice {res['cldice']:.4f} | "
              f"FG_soft {mon['fg_soft']:.4f} u_mean {carry['u_mean']:.4f} agree {mon['agree']:.3f} "
              f"protsep {mon['protsep']:.3f} dW {dw_norm:.3f} (q_src {q_src:.4f})")
        return rec

    for step in range(1, args.iters + 1):
        batch = next(stream)
        clean = batch["image"].to(args.device)
        views = batch["views"].to(args.device)
        fov = batch["fov"].to(args.device)

        with torch.no_grad():
            p_bar, u = teacher_target(teacher, views)
            ves = (p_bar > 0.5)
            phi = torch.maximum(p_bar, 1.0 - p_bar)
            gate = percentile_gate(phi, ves, fov, rho_vessel=args.rho_vessel)
            y_float = ves.float()
            carry["u_mean"] = float((u * fov).sum() / fov.sum().clamp_min(1.0))

        z_clean = student.forward_features(clean)
        logit_clean = student.forward_head(z_clean)
        prob_clean = torch.sigmoid(logit_clean)
        student_views = [torch.sigmoid(student(views[:, k])) for k in range(views.shape[1])]

        l_inv = invariance_loss(student_views, p_bar, u, fov)
        ce = F.binary_cross_entropy_with_logits(logit_clean, y_float, reduction="none")
        gate_f = gate.float()
        l_ce = (ce * gate_f).sum() / gate_f.sum().clamp_min(1.0)
        l_conn = soft_cldice_loss(prob_clean, y_float, fov)
        l_seg = l_ce + args.eta_conn * l_conn
        l_anc = anchor_loss(peft, args.lambda_gn)
        ramp = min(1.0, step / max(1, args.warmup_iters))
        loss = l_seg + args.w_inv * l_inv + args.w_anc * l_anc

        if use_sep:
            l_sep = prototype_sep_loss(z_clean, ves, phi, m0=args.m0, mu_v_fallback=mu_v_src)
            loss = loss + ramp * args.w_sep * l_sep

        if use_occ:
            l_occ = occupancy_loss(prob_clean, fov, q_src)
            loss = loss + ramp * args.w_occ * l_occ

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema_update(teacher, student, args.teacher_m)

        if step % args.eval_interval == 0 or step == 1:
            mon = monitors(student, teacher, clean, fov, gate, p_bar)
            run_eval(step, mon)

    iters = [r["iter"] for r in records]
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(iters, [r["dice"] for r in records], marker="o", label="Dice")
    axis.plot(iters, [r["fg_hard"] for r in records], marker="s", label="FG_hard")
    axis.axhline(q_src, color="gray", linestyle=":", label="q_src")
    axis.set_xlabel("iteration")
    axis.legend(loc="best", fontsize=8)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(run_dir / "visualization" / "adapt_curves.png", dpi=150)
    plt.close(figure)
    torch.save({"student": student.state_dict(), "config": config}, run_dir / "student_final.pt")
    print(f"[stage_c:{args.variant}] done. artifacts under {run_dir}")


if __name__ == "__main__":
    main()
