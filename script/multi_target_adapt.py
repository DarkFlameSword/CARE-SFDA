import argparse
import json
import math
import os
import sys
from collections import defaultdict
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

from adaptation.fairness import HedgeFairness
from adaptation.interventions import CrossDomainMixStyle
from adaptation.interventions import cross_domain_perm
from adaptation.interventions import fourier_amplitude_swap
from adaptation.objectives import invariance_loss
from adaptation.objectives import occupancy_loss
from adaptation.objectives import percentile_gate
from adaptation.objectives import prototype_sep_loss
from adaptation.objectives import soft_cldice_loss
from adaptation.objectives import soft_occupancy
from adaptation.objectives import teacher_student_disagreement
from adaptation.peft import anchor_loss
from adaptation.peft import configure_peft
from adaptation.peft import trainable_parameters
from dataset.domain_registry import DOMAINS
from dataset.domain_registry import SOURCE_DOMAIN
from dataset.domain_registry import TARGET_DOMAINS
from dataset.multi_domain import DomainStratifiedBatchSampler
from dataset.multi_domain import MultiDomainAdaptDataset
from dataset.retinal_vessel_dataset import fdh_dataset_kwargs
from dataset.retinal_vessel_dataset import EvalDataset
from dataset.retinal_vessel_dataset import SourceDataset
from dataset.splits import domain_split
from evaluation.evaluator import Evaluator
from model.unet import UNet

DEFAULT_ANCHOR = "result/stage_a_source_driven_to_7targets/robust_2026-07-09/checkpoint/best_model.pt"


def _nan_to_none(store):
    return {key: (None if value != value else value) for key, value in store.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="CARE-SFDA multi-target joint adaptation (§3.5/§3.11).")
    parser.add_argument("--anchor", type=str, default=DEFAULT_ANCHOR)
    parser.add_argument("--targets", type=str, default=",".join(TARGET_DOMAINS),
                        help="Comma-separated target domains (default: all 7).")
    parser.add_argument("--note", type=str, default="uniform", help="Run-folder note tag.")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-max-images", type=int, default=8,
                        help="Per-domain image cap for periodic eval (final eval uses all).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--domains-per-batch", type=int, default=2)
    parser.add_argument("--starve-domain", type=str, default=None,
                        help="F2: down-weight this domain's sampling to manufacture under-service (§3.10 F2 test).")
    parser.add_argument("--starve-factor", type=float, default=0.33,
                        help="Sampling weight of the starved domain relative to 1.0 for the rest.")
    parser.add_argument("--sampling-temperature", type=float, default=0.0,
                        help="Domain-sampling temperature tau: the per-batch domain-draw probability is "
                             "proportional to (adaptation-pool size)^tau. tau=0 (default) reproduces the "
                             "per-domain-uniform draw; tau=1 is fully size-proportional (large domains "
                             "dominate and small / rare-modality domains are natively under-served); "
                             "tau=0.5 is tempered. This is the native-heterogeneity axis for the §3.10 "
                             "fairness test (a realistic alternative to the artificial --starve-domain).")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--fdh-norm", choices=["zscore", "unit"], default="zscore",
                        help="Intensity normalization axis of the deterministic alignment.")
    parser.add_argument("--fdh-channel", choices=["green", "luminance"], default="green",
                        help="Working-channel axis of the deterministic alignment.")
    parser.add_argument("--fdh-caliber", action=argparse.BooleanOptionalAction, default=True,
                        help="Resample each domain to the common FOV caliber.")
    parser.add_argument("--fdh-polarity", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply the per-domain polarity sign (the FA inversion).")
    parser.add_argument("--k-views", type=int, default=2)
    parser.add_argument("--intervention-strength", type=float, default=1.0)
    parser.add_argument("--mixstyle", action=argparse.BooleanOptionalAction, default=True,
                        help="Cross-domain MixStyle on the K teacher + student intervention forwards (§3.5 Level-2 i).")
    parser.add_argument("--fourier", action=argparse.BooleanOptionalAction, default=True,
                        help="Cross-domain Fourier amplitude swap on the views (§3.5 Level-2 ii).")
    parser.add_argument("--mixstyle-p", type=float, default=0.5)
    parser.add_argument("--mixstyle-alpha", type=float, default=0.1)
    parser.add_argument("--beta-fda", type=float, default=0.01)
    parser.add_argument("--p-fourier", type=float, default=0.5,
                        help="Per-view probability of the Fourier swap (independent per view => cross-view variance).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--w-inv", type=float, default=1.0)
    parser.add_argument("--variance-gate", action=argparse.BooleanOptionalAction, default=False,
                        help="Re-weight L_inv by e^{-u}. Dropped from the method (measured range "
                             "0.998-1.000, ablation +0.0002 Dice); kept to reproduce the runs that "
                             "carried it.")
    parser.add_argument("--trainable-head", action="store_true",
                        help="§3.12 non-circularity control: unfreeze the 1x1 decision head so the "
                             "class anchor is no longer frozen.")
    parser.add_argument("--w-sep", type=float, default=0.1,
                        help="[DEPRECATED] Weight of the L_sep prototype-separation barrier (off by default).")
    parser.add_argument("--w-occ", type=float, default=0.5)
    parser.add_argument("--w-anc", type=float, default=1e-1)
    parser.add_argument("--lambda-gn", type=float, default=1e-3)
    parser.add_argument("--eta-conn", type=float, default=0.5)
    parser.add_argument("--m0", type=float, default=1.0)
    parser.add_argument("--rho-vessel", type=float, default=0.4)
    parser.add_argument("--gamma1", type=float, default=1.0, help="rho_d L_inv weight (§3.10).")
    parser.add_argument("--gamma2", type=float, default=0.5, help="rho_d L_occ weight (§3.10).")
    parser.add_argument("--fairness", choices=["uniform", "groupfrozen", "minmax", "regret", "regret_residual", "mgda"],
                        default="uniform",
                        help="Reweighting rule (§3.10). uniform=1/D per domain; groupfrozen=1/G per group fixed; "
                             "minmax=Hedge on abs surrogate; regret=Hedge on relative regret (ratio to source floor); "
                             "regret_residual=Hedge on the content-regression residual of rho_d (additive; uses the "
                             "source content descriptor instead of a floor); "
                             "mgda=7-domain sequential Multiple-Gradient-Descent (Frank-Wolfe min-norm on "
                             "unit-normalized per-domain gradients; gradient-based Pareto baseline vs the "
                             "loss-reweighting rules).")
    parser.add_argument("--fairness-unit", choices=["domain", "modality"], default="domain",
                        help="Hedge reweighting unit. domain=per-target (worst domain surfaces); modality=§1.3 groups "
                             "(merges color-fundus, hiding the worst domain -- only valid once §3.9 verifies duplicates).")
    parser.add_argument("--floor", choices=["source", "confidence"], default="source",
                        help="Difficulty floor rho_easy (§3.10). source=rho_d(theta_o) frozen source-model surrogate "
                             "(content-magnitude cancels; F0-validated); confidence=surrogate on top-decile-confident images.")
    parser.add_argument("--floor-calibrate", choices=["anchor", "warmup_end"], default="warmup_end",
                        help="Reference iterate at which the source floor rho_d^src is measured (§3.10; only affects "
                             "floor=source). anchor=at the frozen pre-adaptation anchor theta_o (the original "
                             "definition): there the teacher--student disagreement D(f_theta_o, f_theta_o) is 0 and "
                             "the occupancy hinge L_occ is 0 by the q_src construction, so the floor degenerates to "
                             "the style-inconsistency term gamma1*L_inv alone. warmup_end (default)=on the student at "
                             "the end of the warmup ramp (iter warmup_iters), where all surrogate terms are "
                             "non-degenerate, then frozen for the rest of adaptation.")
    parser.add_argument("--hedge-beta", type=float, default=0.01, help="Hedge step on the smoothed signal (§3.10).")
    parser.add_argument("--hedge-ema", type=float, default=0.9, help="EMA smoothing of the Hedge signal (slow timescale).")
    parser.add_argument("--lam-min", type=float, default=0.02, help="Lower floor on each unit's weight (anti-starvation).")
    parser.add_argument("--regret-clip", type=float, nargs=2, default=(0.25, 4.0),
                        help="Clip bounds on the relative regret (Hedge bounded-loss assumption).")
    parser.add_argument("--kref-steps", type=int, default=500,
                        help="Confidence-floor recompute interval (§3.10 K_ref); source floor is frozen.")
    parser.add_argument("--teacher-m", type=float, default=0.99)
    parser.add_argument("--ema-domain", type=float, default=0.95, help="Per-domain rho_d/monitor EMA rate.")
    parser.add_argument("--eval-fraction", type=float, default=0.30)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--no-occ", action="store_true", help="Disable the occupancy barrier.")
    parser.add_argument("--use-sep", action="store_true",
                        help="[DEPRECATED] Enable the L_sep prototype-separation barrier (redundant by construction "
                             "-- the frozen +-w rays already separate the classes; off by default, ablation only).")
    parser.add_argument("--decoder-lora", action="store_true",
                        help="A-6: also LoRA-adapt the decoder convs + unfreeze their GN affine (head stays frozen).")
    parser.add_argument("--gn-only", action="store_true",
                        help="Inject no LoRA; adapt only the encoder GroupNorm affine (isolates whether the "
                             "encoder LoRA contributes anything at multi-target). dW is then identically 0.")
    parser.add_argument("--clean-teacher-pseudolabel", action="store_true",
                        help="A-6: derive the self-training pseudo-label from the teacher's single clean-image "
                             "pass instead of the intervention-averaged posterior p_bar (§3.8 avg-vs-clean).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--result-root", type=str, default="result")

    return parser.parse_args()


def build_model(args):
    model = UNet(in_channels=1, n_classes=1, base_channels=args.base_channels, norm="group", head_bias=False)
    model.load_state_dict(torch.load(args.anchor, map_location="cpu", weights_only=False)["model"])
    model.to(args.device)
    peft = configure_peft(model, r=args.lora_rank, alpha=args.lora_alpha,
                          include_decoder=args.decoder_lora, use_lora=not args.gn_only,
                          train_head=args.trainable_head)

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


def per_domain_q_src(model, multi_dataset, targets, args, n_images=5):
    q_src = {}

    for domain in targets:
        dataset = multi_dataset.datasets[domain]
        values = []

        for i in range(min(n_images, len(dataset))):
            item = dataset[i]
            image = item["image"].unsqueeze(0).to(args.device)
            fov = item["fov"].unsqueeze(0).to(args.device)

            with torch.no_grad():
                values.append(float(soft_occupancy(torch.sigmoid(model(image)), fov).item()))

        q_src[domain] = float(np.mean(values))

    return q_src


def per_domain_content_fg(model, multi_dataset, targets, args, n_images=8):
    content = {}

    for domain in targets:
        dataset = multi_dataset.datasets[domain]
        values = []

        for i in range(min(n_images, len(dataset))):
            item = dataset[i]
            image = item["image"].unsqueeze(0).to(args.device)
            fov = item["fov"].unsqueeze(0).to(args.device)

            with torch.no_grad():
                prob = torch.sigmoid(model(image))
                values.append(float(((prob > 0.5).float() * fov).sum() / fov.sum().clamp_min(1.0)))

        content[domain] = float(np.mean(values))

    return content


def compute_difficulty_floor(student, teacher, mixstyle, multi_dataset, targets, args,
                             q_src, use_occ, pool=40, decile=0.1, min_images=2):
    device = args.device
    floor = {}

    for di, domain in enumerate(targets):
        dataset = multi_dataset.datasets[domain]
        n_all = len(dataset)
        rng = np.random.default_rng(1000 * args.seed + di)

        if n_all > pool:
            idxs = sorted(int(x) for x in rng.choice(n_all, size=pool, replace=False))

        else:
            idxs = list(range(n_all))

        items = [dataset[i] for i in idxs]
        n = len(items)
        confs = []

        for item in items:
            image = item["image"].unsqueeze(0).to(device)
            fov = item["fov"].unsqueeze(0).to(device)

            with torch.no_grad():
                prob = torch.sigmoid(teacher(image))
                confs.append(float((torch.maximum(prob, 1.0 - prob) * fov).sum() / fov.sum().clamp_min(1.0)))

        k = min(n, max(min_images, math.ceil(decile * n)))
        top = sorted(range(n), key=lambda i: confs[i], reverse=True)[:k]
        clean = torch.stack([items[i]["image"] for i in top]).to(device)
        views = torch.stack([items[i]["views"] for i in top]).to(device)
        fov = torch.stack([items[i]["fov"] for i in top]).to(device)
        perm = torch.arange(k - 1, -1, -1, device=device)
        mixstyle.perm = perm
        swapped = build_views(views, perm, fov, args.fourier, args.beta_fda, args.p_fourier)

        with torch.no_grad():
            mixstyle.enabled = args.mixstyle
            stack = torch.stack([torch.sigmoid(teacher(sv)) for sv in swapped], dim=0)
            p_bar = stack.mean(0)
            u = stack.var(0, unbiased=False)
            mixstyle.enabled = False
            p_teacher_clean = torch.sigmoid(teacher(clean))
            z_clean = student.forward_features(clean)
            prob_clean = torch.sigmoid(student.forward_head(z_clean))
            mixstyle.enabled = args.mixstyle
            student_views = [torch.sigmoid(student(sv)) for sv in swapped]
            mixstyle.enabled = False
            values = []

            for i in range(k):
                sl = slice(i, i + 1)
                l_inv_i = invariance_loss([sv[sl] for sv in student_views], p_bar[sl], u[sl], fov[sl])
                l_occ_i = occupancy_loss(prob_clean[sl], fov[sl], q_src[domain]) if use_occ else 0.0
                disagree = teacher_student_disagreement(prob_clean[sl], p_teacher_clean[sl], fov[sl])
                values.append(float(disagree + args.gamma1 * l_inv_i + args.gamma2 * l_occ_i))

        floor[domain] = sum(values) / len(values)

    return floor


def modality_groups(targets):
    groups = defaultdict(list)

    for domain in targets:
        groups[DOMAINS[domain].modality].append(domain)

    return dict(groups)


def group_aggregates(per_domain_dice, groups):
    group_dice = {}

    for modality, members in groups.items():
        group_dice[modality] = float(np.mean([per_domain_dice[m] for m in members]))

    values = list(group_dice.values())

    return min(values), max(values) - min(values), group_dice


@torch.no_grad()
def ema_update(teacher, student, m):
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.mul_(m).add_(p_s, alpha=1.0 - m)


def build_views(views, perm, fov, use_fourier, beta, p_fourier):
    out = []

    for k in range(views.shape[1]):
        view = views[:, k]

        if use_fourier and float(torch.rand(())) < p_fourier:
            view = fourier_amplitude_swap(view, view[perm], beta=beta, fov=fov)

        out.append(view)

    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    use_occ = not args.no_occ
    use_sep = args.use_sep
    device = args.device

    splits = {t: domain_split(t, eval_fraction=args.eval_fraction, seed=args.seed) for t in targets}
    adapt_ids = {t: splits[t][0] for t in targets}
    eval_ids = {t: splits[t][1] for t in targets}
    fdh = fdh_dataset_kwargs(norm=args.fdh_norm, channel=args.fdh_channel,
                             caliber=args.fdh_caliber, polarity=args.fdh_polarity)
    multi_dataset = MultiDomainAdaptDataset(adapt_ids, patch_size=args.patch_size, n_views=args.k_views,
                                            intervention_strength=args.intervention_strength, fdh=fdh)
    if args.sampling_temperature != 0.0:
        sampling_weights = {t: float(len(adapt_ids[t])) ** args.sampling_temperature for t in targets}

    else:
        sampling_weights = None

    if args.starve_domain:
        sampling_weights = sampling_weights or {t: 1.0 for t in targets}
        sampling_weights[args.starve_domain] *= args.starve_factor

    sampler = DomainStratifiedBatchSampler(multi_dataset.domain_positions, batch_size=args.batch_size,
                                           n_batches=args.iters, domains_per_batch=args.domains_per_batch,
                                           seed=args.seed, sampling_weights=sampling_weights)
    loader = DataLoader(multi_dataset, batch_sampler=sampler, num_workers=4, persistent_workers=True)

    student, peft = build_model(args)
    teacher, _ = build_model(args)
    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad_(False)

    mixstyle = CrossDomainMixStyle(p=args.mixstyle_p, alpha=args.mixstyle_alpha)
    mixstyle.install(student)
    mixstyle.install(teacher)
    w_vec = student.head.weight.detach().clone().view(-1)
    params = trainable_parameters(student)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    mu_v_src = source_vessel_prototype(student, args) if use_sep else None
    q_src = per_domain_q_src(student, multi_dataset, targets, args)
    content = per_domain_content_fg(student, multi_dataset, targets, args) if args.fairness == "regret_residual" else None
    groups = modality_groups(targets)
    game_groups = {t: [t] for t in targets} if args.fairness_unit == "domain" else groups
    fair = HedgeFairness(game_groups, mode=args.fairness, beta=args.hedge_beta, ema=args.hedge_ema,
                         clip=tuple(args.regret_clip), lam_min=args.lam_min, content=content)
    uses_floor = args.fairness == "regret"
    floor_decile = 1.0 if args.floor == "source" else 0.1
    floor_pool = 8 if args.floor == "source" else 40
    rho_easy = (compute_difficulty_floor(student, teacher, mixstyle, multi_dataset, targets, args,
                                         q_src, use_occ, pool=floor_pool, decile=floor_decile)
                if uses_floor else None)

    def calibrate_floor_at_warmup_end(step, current_floor):
        if uses_floor and args.floor == "source" and args.floor_calibrate == "warmup_end" and step == args.warmup_iters:

            return compute_difficulty_floor(student, teacher, mixstyle, multi_dataset, targets, args,
                                            q_src, use_occ, pool=floor_pool, decile=floor_decile)

        return current_floor

    rho_d = {t: float("nan") for t in targets}
    u_d = {t: float("nan") for t in targets}
    linv_d = {t: float("nan") for t in targets}

    def ema(store, key, value):
        prev = store[key]
        store[key] = value if (prev != prev) else args.ema_domain * prev + (1.0 - args.ema_domain) * value

    mgda = args.fairness == "mgda"
    alpha_ema = {t: float("nan") for t in targets}
    mgda_rng = np.random.default_rng(args.seed)

    def _flat_grad(ps):
        return torch.cat([(p.grad.detach().flatten() if p.grad is not None
                           else torch.zeros(p.numel(), device=p.device)) for p in ps])

    def _write_grad(ps, flat):
        offset = 0

        for p in ps:
            n = p.numel()
            p.grad = flat[offset:offset + n].view_as(p).clone()
            offset += n

    def frank_wolfe_min_norm(vecs, iters=60, tol=1e-8):
        matrix = torch.stack(vecs)
        gram = matrix @ matrix.t()
        n = gram.shape[0]
        alpha = torch.full((n,), 1.0 / n, device=gram.device, dtype=gram.dtype)

        for _ in range(iters):
            grad = gram @ alpha
            i = int(torch.argmin(grad))
            v1 = float(alpha @ grad)
            v2 = float(grad[i])
            v3 = float(gram[i, i])
            denom = v1 - 2.0 * v2 + v3
            gamma = 0.0 if denom <= tol else min(max((v1 - v2) / denom, 0.0), 1.0)
            nxt = (1.0 - gamma) * alpha
            nxt[i] += gamma

            if float((nxt - alpha).abs().max()) < tol:
                alpha = nxt
                break

            alpha = nxt

        return alpha

    def _domain_grad(item_d, item_donor, domain, ramp):
        clean = torch.stack([torch.as_tensor(item_d["image"]), torch.as_tensor(item_donor["image"])]).to(device)
        views = torch.stack([torch.as_tensor(item_d["views"]), torch.as_tensor(item_donor["views"])]).to(device)
        fov = torch.stack([torch.as_tensor(item_d["fov"]), torch.as_tensor(item_donor["fov"])]).to(device)
        perm = torch.tensor([1, 0], device=device)
        mixstyle.perm = perm
        swapped = build_views(views, perm, fov, args.fourier, args.beta_fda, args.p_fourier)

        with torch.no_grad():
            mixstyle.enabled = args.mixstyle
            stack = torch.stack([torch.sigmoid(teacher(sv)) for sv in swapped], dim=0)
            p_bar = stack.mean(0)
            u = stack.var(0, unbiased=False)
            mixstyle.enabled = False
            p_teacher_clean = torch.sigmoid(teacher(clean))
            pseudo = p_teacher_clean if args.clean_teacher_pseudolabel else p_bar
            ves = pseudo > 0.5
            phi = torch.maximum(pseudo, 1.0 - pseudo)
            y_float = ves.float()

        z_clean = student.forward_features(clean)
        logit_clean = student.forward_head(z_clean)
        prob_clean = torch.sigmoid(logit_clean)
        mixstyle.enabled = args.mixstyle
        student_views = [torch.sigmoid(student(sv)) for sv in swapped]
        mixstyle.enabled = False

        sl = slice(0, 1)
        gate = percentile_gate(phi[sl], ves[sl], fov[sl], rho_vessel=args.rho_vessel).float()
        l_inv = invariance_loss([sv[sl] for sv in student_views], p_bar[sl], u[sl], fov[sl])
        ce = F.binary_cross_entropy_with_logits(logit_clean[sl], y_float[sl], reduction="none")
        l_ce = (ce * gate).sum() / gate.sum().clamp_min(1.0)
        l_conn = soft_cldice_loss(prob_clean[sl], y_float[sl], fov[sl])
        l_d = l_ce + args.eta_conn * l_conn + args.w_inv * l_inv
        l_occ = torch.zeros((), device=device)

        if use_occ:
            l_occ = occupancy_loss(prob_clean[sl], fov[sl], q_src[domain])
            l_d = l_d + ramp * args.w_occ * l_occ

        optimizer.zero_grad(set_to_none=True)
        l_d.backward()
        grad = _flat_grad(params)

        with torch.no_grad():
            disagree = teacher_student_disagreement(prob_clean[sl], p_teacher_clean[sl], fov[sl])
            ema(rho_d, domain, float(disagree + args.gamma1 * l_inv + args.gamma2 * l_occ))
            ema(u_d, domain, float((u[sl] * fov[sl]).sum() / fov[sl].sum().clamp_min(1.0)))
            ema(linv_d, domain, float(l_inv))

        return grad

    def mgda7_step(step):
        ramp = min(1.0, step / max(1, args.warmup_iters))
        order = list(targets)
        items = {d: multi_dataset.datasets[d][int(mgda_rng.integers(len(multi_dataset.datasets[d])))]
                 for d in order}
        ghat = {}

        for idx, domain in enumerate(order):
            donor = order[(idx + 1 + int(mgda_rng.integers(len(order) - 1))) % len(order)]
            grad = _domain_grad(items[domain], items[donor], domain, ramp)
            ghat[domain] = grad / (grad.norm() + 1e-12)

        alpha = frank_wolfe_min_norm([ghat[d] for d in order])
        combined = sum(float(alpha[i]) * ghat[order[i]] for i in range(len(order)))

        for i, domain in enumerate(order):
            ema(alpha_ema, domain, float(alpha[i]))

        optimizer.zero_grad(set_to_none=True)
        (args.w_anc * anchor_loss(peft, args.lambda_gn)).backward()
        _write_grad(params, combined + _flat_grad(params))
        optimizer.step()

    date = datetime.now().strftime("%Y-%m-%d")
    run_dir = Path(args.result_root) / "multi_target_adapt_7targets" / f"{args.note}_{date}"

    if run_dir.exists():
        run_dir = run_dir.parent / f"{args.note}_{date}_{datetime.now().strftime('%H%M%S')}"

    (run_dir / "visualization").mkdir(parents=True)
    views_degenerate = bool((args.intervention_strength == 0.0 and not args.mixstyle
                             and not args.fourier) or args.k_views == 1)
    config = {**vars(args), "targets": targets, "q_src": q_src, "content_fg": content,
              "use_occ": use_occ, "use_sep": use_sep,
              "views_degenerate": views_degenerate,
              "pseudolabel_source_effective": ("clean" if (views_degenerate
                                                           or args.clean_teacher_pseudolabel)
                                               else "intervention_averaged"),
              "variance_gate_effective": ("off" if (views_degenerate or not args.variance_gate)
                                          else "on"),
              "sampling_weights": sampling_weights,
              "n_adapt": {t: len(adapt_ids[t]) for t in targets}, "n_eval": {t: len(eval_ids[t]) for t in targets},
              "groups": groups, "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
              "code_commit_hash": "None"}

    with open(run_dir / "config.yaml", "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    eval_datasets = {t: EvalDataset(t, split="train", sample_ids=eval_ids[t], **fdh) for t in targets}
    evaluator = Evaluator(SOURCE_DOMAIN, targets, device=device, tile=args.patch_size, use_amp=False,
                          max_images=args.eval_max_images, datasets=eval_datasets)
    records = []

    def run_eval(step):
        results = evaluator.evaluate(student)
        per_dice = {t: results[t]["dice"] for t in targets}
        per_cldice = {t: results[t]["cldice"] for t in targets}
        worst, spread, group_dice = group_aggregates(per_dice, groups)

        with torch.no_grad():
            dw_sq = sum((l.delta_w_fro_sq() for l in peft["lora"]), torch.zeros((), device=device))
            dw_norm = float(torch.sqrt(dw_sq).item())
            w_now = student.head.weight.detach().view(-1)
            w_cos = float(F.cosine_similarity(w_now, w_vec, dim=0))
            w_rel_drift = float((w_now - w_vec).norm() / w_vec.norm().clamp_min(1e-12))

        r_tilde = {t: (rho_d[t] / (rho_easy[t] + fair.eps) if (rho_easy is not None and rho_d[t] == rho_d[t]) else None)
                   for t in targets}
        rec = {"iter": step, "dice": per_dice, "cldice": per_cldice, "worst_dice": worst, "spread": spread,
               "group_dice": group_dice, "dw_norm": dw_norm,
               "w_cos": w_cos, "w_rel_drift": w_rel_drift, "rho_d": _nan_to_none(rho_d),
               "u_d": _nan_to_none(u_d), "l_inv_d": _nan_to_none(linv_d),
               "rho_easy": (_nan_to_none(rho_easy) if rho_easy is not None else None),
               "r_tilde": r_tilde, "hedge_signal": _nan_to_none(fair.signal),
               "lambda": fair.per_domain_lambda(), "lambda_group": dict(fair.lam),
               "mgda_alpha": _nan_to_none(alpha_ema)}
        records.append(rec)

        with open(run_dir / "metric.json", "w") as handle:
            json.dump({"config": config, "trajectory": records}, handle, indent=2)

        mean_dice = float(np.mean(list(per_dice.values())))
        mean_u = float(np.nanmean(list(u_d.values())))
        print(f"[mtda:{args.note}] iter {step}: meanDice {mean_dice:.4f} worstDice {worst:.4f} "
              f"spread {spread:.4f} | u_mean {mean_u:.5f} dW {dw_norm:.3f}")

        for t in targets:
            print(f"    {t:>9}: dice {per_dice[t]:.4f} cldice {per_cldice[t]:.4f} "
                  f"u {u_d[t]:.5f} l_inv {linv_d[t]:.4f} rho {rho_d[t]:.4f} (q_src {q_src[t]:.4f})")

        return rec

    stream = iter(loader)

    for step in range(1, args.iters + 1):

        if mgda:
            mgda7_step(step)
            ema_update(teacher, student, args.teacher_m)
            rho_easy = calibrate_floor_at_warmup_end(step, rho_easy)

            if uses_floor and args.floor == "confidence" and step % args.kref_steps == 0:
                rho_easy = compute_difficulty_floor(student, teacher, mixstyle, multi_dataset, targets, args,
                                                    q_src, use_occ, pool=40, decile=0.1)

            if step == 1 or (step % args.eval_interval == 0 and step != args.iters):
                run_eval(step)

            continue

        batch = next(stream)
        clean = batch["image"].to(device)
        views = batch["views"].to(device)
        fov = batch["fov"].to(device)
        domains = list(batch["domain"])
        perm = cross_domain_perm(domains).to(device)
        mixstyle.perm = perm
        swapped = build_views(views, perm, fov, args.fourier, args.beta_fda, args.p_fourier)

        with torch.no_grad():
            mixstyle.enabled = args.mixstyle
            stack = torch.stack([torch.sigmoid(teacher(sv)) for sv in swapped], dim=0)
            p_bar = stack.mean(0)
            u = stack.var(0, unbiased=False)
            mixstyle.enabled = False
            p_teacher_clean = torch.sigmoid(teacher(clean))
            pseudo = p_teacher_clean if args.clean_teacher_pseudolabel else p_bar
            ves = pseudo > 0.5
            phi = torch.maximum(pseudo, 1.0 - pseudo)
            y_float = ves.float()

        z_clean = student.forward_features(clean)
        logit_clean = student.forward_head(z_clean)
        prob_clean = torch.sigmoid(logit_clean)

        mixstyle.enabled = args.mixstyle
        student_views = [torch.sigmoid(student(sv)) for sv in swapped]
        mixstyle.enabled = False

        ramp = min(1.0, step / max(1, args.warmup_iters))
        total = 0.0

        for i in range(clean.shape[0]):
            d = domains[i]
            sl = slice(i, i + 1)
            gate_i = percentile_gate(phi[sl], ves[sl], fov[sl], rho_vessel=args.rho_vessel).float()
            l_inv_i = invariance_loss([sv[sl] for sv in student_views], p_bar[sl], u[sl], fov[sl],
                                      variance_gate=args.variance_gate)
            ce_i = F.binary_cross_entropy_with_logits(logit_clean[sl], y_float[sl], reduction="none")
            l_ce_i = (ce_i * gate_i).sum() / gate_i.sum().clamp_min(1.0)
            l_conn_i = soft_cldice_loss(prob_clean[sl], y_float[sl], fov[sl])
            l_seg_i = l_ce_i + args.eta_conn * l_conn_i
            l_d = l_seg_i + args.w_inv * l_inv_i
            l_sep_i = torch.zeros((), device=device)
            l_occ_i = torch.zeros((), device=device)

            if use_sep:
                l_sep_i = prototype_sep_loss(z_clean[sl], ves[sl], phi[sl], m0=args.m0, mu_v_fallback=mu_v_src)
                l_d = l_d + ramp * args.w_sep * l_sep_i

            if use_occ:
                l_occ_i = occupancy_loss(prob_clean[sl], fov[sl], q_src[d])
                l_d = l_d + ramp * args.w_occ * l_occ_i

            total = total + fair.weight(d) * l_d

            with torch.no_grad():
                disagree = teacher_student_disagreement(prob_clean[sl], p_teacher_clean[sl], fov[sl])
                surrogate = float(disagree + args.gamma1 * l_inv_i + args.gamma2 * l_occ_i)
                ema(rho_d, d, surrogate)
                ema(u_d, d, float((u[sl] * fov[sl]).sum() / fov[sl].sum().clamp_min(1.0)))
                ema(linv_d, d, float(l_inv_i))

        loss = total / (clean.shape[0] / fair.n_domains) + args.w_anc * anchor_loss(peft, args.lambda_gn)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema_update(teacher, student, args.teacher_m)
        rho_easy = calibrate_floor_at_warmup_end(step, rho_easy)

        if all(rho_d[t] == rho_d[t] for t in targets):
            fair.update(rho_d, rho_easy)

        if uses_floor and args.floor == "confidence" and step % args.kref_steps == 0:
            rho_easy = compute_difficulty_floor(student, teacher, mixstyle, multi_dataset, targets, args,
                                                q_src, use_occ, pool=40, decile=0.1)

        if step == 1 or (step % args.eval_interval == 0 and step != args.iters):
            run_eval(step)

    evaluator.max_images = None
    run_eval(args.iters)

    steps = [r["iter"] for r in records]
    figure, axis = plt.subplots(figsize=(10, 6))

    for t in targets:
        axis.plot(steps, [r["dice"][t] for r in records], marker=".", alpha=0.6, label=t)

    axis.plot(steps, [r["worst_dice"] for r in records], marker="o", color="black", linewidth=2, label="worstDice")
    axis.set_xlabel("iteration")
    axis.set_ylabel("Dice")
    axis.legend(loc="best", fontsize=7, ncol=2)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(run_dir / "visualization" / "multitarget_curves.png", dpi=150)
    plt.close(figure)
    torch.save({"student": student.state_dict(), "config": config}, run_dir / "student_final.pt")
    print(f"[mtda:{args.note}] done. artifacts under {run_dir}")


if __name__ == "__main__":
    main()
