import torch
import torch.nn.functional as F

__all__ = [
    "soft_skeleton",
    "soft_occupancy",
    "occupancy_loss",
    "signed_margins",
    "invariance_loss",
    "percentile_gate",
    "prototype_sep_loss",
    "soft_cldice_loss",
    "teacher_student_disagreement",
]


def _soft_erode(x):
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x):
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x):
    return _soft_dilate(_soft_erode(x))


def soft_skeleton(prob, iters=10):
    x = prob
    opened = _soft_open(x)
    skel = F.relu(x - opened)

    for _ in range(iters):
        x = _soft_erode(x)
        opened = _soft_open(x)
        delta = F.relu(x - opened)
        skel = skel + F.relu(delta - skel * delta)

    return skel


def soft_occupancy(prob, fov, iters=10):
    skel = soft_skeleton(prob, iters)
    m = fov.to(skel.dtype)

    return (skel * m).sum() / m.sum().clamp_min(1.0)


def occupancy_loss(prob, fov, q_src, iters=10):
    q = soft_occupancy(prob, fov, iters)

    return F.relu(q_src - q)


def signed_margins(logit, sign):
    vessel = sign > 0
    bg = sign < 0
    margin_v = logit[vessel].mean().item() if vessel.any() else float("nan")
    margin_bg = logit[bg].mean().item() if bg.any() else float("nan")

    return margin_v, margin_bg


def _bernoulli_sym_kl(p, q, eps=1e-6):
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    kl_pq = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
    kl_qp = q * torch.log(q / p) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - p))

    return kl_pq + kl_qp


def _soft_dice(p, q, fov, eps=1.0):
    m = fov.to(p.dtype)
    inter = (p * q * m).sum()
    denom = (p * m).sum() + (q * m).sum()

    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def invariance_loss(student_view_probs, teacher_prob, u, fov, eps=1.0, variance_gate=False):
    reliability = torch.exp(-u) if variance_gate else torch.ones_like(u)
    weight = reliability * fov.to(u.dtype)
    denom = weight.sum().clamp_min(1.0)
    total = 0.0

    for sp in student_view_probs:
        pixel = (weight * _bernoulli_sym_kl(sp, teacher_prob)).sum() / denom
        region = _soft_dice(sp, teacher_prob, fov, eps)
        total = total + pixel + region

    return total / len(student_view_probs)


def teacher_student_disagreement(student_prob, teacher_prob, fov):
    m = fov.to(student_prob.dtype)
    pixel = (_bernoulli_sym_kl(student_prob, teacher_prob) * m).sum() / m.sum().clamp_min(1.0)
    region = _soft_dice(student_prob, teacher_prob, fov)

    return pixel + region


def percentile_gate(phi, pred_vessel, fov, rho_vessel=0.4, rho_bg=0.6):
    fov_b = fov > 0.5
    vessel = pred_vessel & fov_b
    bg = (~pred_vessel) & fov_b
    gate = torch.zeros_like(fov_b)

    if vessel.any():
        gate = gate | (vessel & (phi >= torch.quantile(phi[vessel], 1.0 - rho_vessel)))

    if bg.any():
        gate = gate | (bg & (phi >= torch.quantile(phi[bg], 1.0 - rho_bg)))

    return gate


def prototype_sep_loss(z, pred_vessel, phi, m0=1.0, mu_v_fallback=None, min_weight=1.0, eps=1e-6):
    c = z.shape[1]
    z_flat = z.permute(0, 2, 3, 1).reshape(-1, c)
    weight = phi.reshape(-1)
    w_v = weight * pred_vessel.reshape(-1).to(weight.dtype)
    w_b = weight * (~pred_vessel).reshape(-1).to(weight.dtype)
    w_v_sum = w_v.sum()

    if mu_v_fallback is not None and float(w_v_sum) < min_weight:
        mu_v = mu_v_fallback

    else:
        mu_v = (z_flat * w_v.unsqueeze(1)).sum(0) / w_v_sum.clamp_min(eps)

    mu_b = (z_flat * w_b.unsqueeze(1)).sum(0) / w_b.sum().clamp_min(eps)

    return F.relu(m0 - (mu_v - mu_b).norm())


def soft_cldice_loss(pred_prob, target, fov, iters=10, eps=1e-6):
    skel_p = soft_skeleton(pred_prob, iters)
    skel_t = soft_skeleton(target, iters)
    m = fov.to(pred_prob.dtype)
    t_prec = (skel_p * target * m).sum() / (skel_p * m).sum().clamp_min(eps)
    t_sens = (skel_t * pred_prob * m).sum() / (skel_t * m).sum().clamp_min(eps)

    return 1.0 - 2.0 * t_prec * t_sens / (t_prec + t_sens + eps)
