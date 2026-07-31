import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dataset.domain_registry import DOMAINS

__all__ = ["modality_groups", "load_final", "group_mean", "summarize"]


def modality_groups(domains):
    groups = defaultdict(list)

    for domain in domains:
        groups[DOMAINS[domain].modality].append(domain)

    return dict(groups)


def load_final(run_dir):
    with open(Path(run_dir) / "metric.json", encoding="utf-8") as handle:
        payload = json.load(handle)

    return payload["config"], payload["trajectory"][-1]


def group_mean(per_domain, groups):
    return {modality: float(np.mean([per_domain[m] for m in members]))
            for modality, members in groups.items()}


def summarize(per_domain, groups):
    grouped = group_mean(per_domain, groups)
    values = list(grouped.values())

    return {
        "mean": float(np.mean(list(per_domain.values()))),
        "group_mean": float(np.mean(values)),
        "worst": float(min(values)),
        "spread": float(max(values) - min(values)),
    }
