import torch
import torch.nn as nn

__all__ = ["cross_domain_perm", "fourier_amplitude_swap", "CrossDomainMixStyle"]


def cross_domain_perm(domains, generator=None):
    n = len(domains)
    perm = torch.arange(n)

    for i in range(n):
        others = [j for j in range(n) if domains[j] != domains[i]]

        if others:
            pick = int(torch.randint(len(others), (1,), generator=generator).item())
            perm[i] = others[pick]

    return perm


def fourier_amplitude_swap(x, x_ref, beta=0.01, fov=None):
    height = x.shape[-2]
    width = x.shape[-1]
    fx = torch.fft.fft2(x, norm="ortho")
    fr = torch.fft.fft2(x_ref, norm="ortho")
    amp_x = torch.fft.fftshift(fx.abs(), dim=(-2, -1))
    amp_r = torch.fft.fftshift(fr.abs(), dim=(-2, -1))
    phase_x = fx.angle()
    cy = height // 2
    cx = width // 2
    half = max(1, int(round(beta * min(height, width))))
    amp_new = amp_x.clone()
    amp_new[..., cy - half:cy + half + 1, cx - half:cx + half + 1] = \
        amp_r[..., cy - half:cy + half + 1, cx - half:cx + half + 1]
    amp_new = torch.fft.ifftshift(amp_new, dim=(-2, -1))
    out = torch.fft.ifft2(torch.polar(amp_new, phase_x), norm="ortho").real

    if fov is not None:
        out = out * fov.to(out.dtype)

    return out


class CrossDomainMixStyle:
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.enabled = False
        self.perm = None
        self._beta = torch.distributions.Beta(alpha, alpha)
        self._handles = []

    def _mix(self, h):
        n = h.shape[0]

        if not self.enabled or n < 2 or float(torch.rand(())) > self.p:
            return h

        perm = self.perm.to(h.device) if self.perm is not None else torch.arange(n - 1, -1, -1, device=h.device)
        mu = h.mean(dim=(2, 3), keepdim=True)
        sigma = (h.var(dim=(2, 3), keepdim=True) + self.eps).sqrt()
        mu = mu.detach()
        sigma = sigma.detach()
        normed = (h - mu) / sigma
        lam = self._beta.sample((n, 1, 1, 1)).to(h.device)
        mu_mix = lam * mu + (1.0 - lam) * mu[perm]
        sigma_mix = lam * sigma + (1.0 - lam) * sigma[perm]

        return normed * sigma_mix + mu_mix

    def install(self, model):
        def hook(module, inputs, output):

            return self._mix(output)

        self._handles.append(model.inc.register_forward_hook(hook))
        self._handles.append(model.down1.conv.register_forward_hook(hook))

        return self

    def remove(self):
        for handle in self._handles:
            handle.remove()

        self._handles = []
