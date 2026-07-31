import numpy as np

from dataset.domain_registry import index_domain

__all__ = ["domain_split"]


def domain_split(name, eval_fraction=0.30, seed=0, split="train"):
    samples = index_domain(name, split=split, require_label=False)
    ids = sorted(s.sample_id for s in samples)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    n_eval = max(1, int(round(len(ids) * eval_fraction)))
    eval_positions = set(int(i) for i in order[:n_eval])
    adapt_ids = [ids[i] for i in range(len(ids)) if i not in eval_positions]
    eval_ids = [ids[i] for i in range(len(ids)) if i in eval_positions]

    return sorted(adapt_ids), sorted(eval_ids)
