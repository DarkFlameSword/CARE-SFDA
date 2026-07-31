import math

__all__ = ["HedgeFairness"]


class HedgeFairness:
    def __init__(self, groups, mode="uniform", beta=0.5, ema=0.9, clip=(0.25, 4.0), eps=1e-3,
                 lam_min=1e-3, content=None):
        self.groups = groups
        self.group_names = list(groups.keys())
        self.domain_to_group = {d: g for g, members in groups.items() for d in members}
        self.group_size = {g: len(members) for g, members in groups.items()}
        self.n_domains = sum(len(members) for members in groups.values())
        self.content = content
        self.group_content = ({g: sum(content[d] for d in members) / len(members)
                               for g, members in groups.items()} if content is not None else None)
        self.mode = mode
        self.beta = beta
        self.ema = ema
        self.clip = clip
        self.eps = eps
        self.lam_min = lam_min
        self.lam = {g: 1.0 / len(self.group_names) for g in self.group_names}
        self.signal = {g: float("nan") for g in self.group_names}

    def weight(self, domain):
        if self.mode == "uniform":

            return 1.0 / self.n_domains

        group = self.domain_to_group[domain]

        return self.lam[group] / self.group_size[group]

    def _group_signal(self, rho_d, rho_easy_d):
        signal = {}

        for group, members in self.groups.items():
            rho_g = sum(rho_d[d] for d in members) / len(members)

            if self.mode == "minmax":
                value = rho_g

            else:
                easy_g = sum(rho_easy_d[d] for d in members) / len(members)
                value = rho_g / (easy_g + self.eps)

            signal[group] = min(max(value, self.clip[0]), self.clip[1])

        return signal

    def _residual_signal(self, rho_d):
        names = self.group_names
        rho = [sum(rho_d[d] for d in self.groups[g]) / self.group_size[g] for g in names]
        con = [self.group_content[g] for g in names]
        n = len(names)
        c_bar = sum(con) / n
        r_bar = sum(rho) / n
        s_cc = sum((c - c_bar) ** 2 for c in con)
        slope = (sum((con[i] - c_bar) * (rho[i] - r_bar) for i in range(n)) / s_cc) if s_cc > 1e-12 else 0.0
        intercept = r_bar - slope * c_bar
        resid = [rho[i] - (intercept + slope * con[i]) for i in range(n)]
        scale = r_bar + self.eps

        return {names[i]: min(max(resid[i] / scale, -1.0), 1.0) for i in range(n)}

    def update(self, rho_d, rho_easy_d):
        if self.mode in ("uniform", "groupfrozen"):

            return

        raw = self._residual_signal(rho_d) if self.mode == "regret_residual" else self._group_signal(rho_d, rho_easy_d)

        for group in self.group_names:
            value = raw[group]
            prev = self.signal[group]
            self.signal[group] = value if (prev != prev) else self.ema * prev + (1.0 - self.ema) * value

        updated = {g: self.lam[g] * math.exp(self.beta * self.signal[g]) for g in self.group_names}
        total = sum(updated.values())
        floored = {g: max(updated[g] / total, self.lam_min) for g in self.group_names}
        renorm = sum(floored.values())

        for group in self.group_names:
            self.lam[group] = floored[group] / renorm

    def per_domain_lambda(self):
        return {d: self.weight(d) for d in self.domain_to_group}
