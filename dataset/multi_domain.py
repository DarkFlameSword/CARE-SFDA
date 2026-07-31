import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from dataset.retinal_vessel_dataset import TargetAdaptDataset

__all__ = ["MultiDomainAdaptDataset", "DomainStratifiedBatchSampler"]


class MultiDomainAdaptDataset(Dataset):
    def __init__(self, adapt_ids, patch_size=512, n_views=2, intervention_strength=1.0,
                 target_diameter=1024.0, fdh=None):
        self.domains = list(adapt_ids.keys())
        self.datasets = {}
        self.index = []
        self.domain_positions = {}
        fdh = dict(fdh or {})

        for domain in self.domains:
            dataset = TargetAdaptDataset(domain, split="train", patch_size=patch_size,
                                         sample_ids=adapt_ids[domain], n_views=n_views,
                                         intervention_strength=intervention_strength,
                                         target_diameter=target_diameter, **fdh)
            self.datasets[domain] = dataset
            positions = []

            for local_idx in range(len(dataset)):
                positions.append(len(self.index))
                self.index.append((domain, local_idx))

            self.domain_positions[domain] = positions

        for domain in self.domains:

            if not self.domain_positions[domain]:
                raise ValueError(f"Domain {domain!r} has an empty adaptation pool (no unlabeled ids).")

    def __len__(self):

        return len(self.index)

    def __getitem__(self, global_idx):
        domain, local_idx = self.index[global_idx]

        return self.datasets[domain][local_idx]


class DomainStratifiedBatchSampler(Sampler):
    def __init__(self, domain_positions, batch_size, n_batches, domains_per_batch=2, seed=0,
                 sampling_weights=None):
        self.domain_positions = domain_positions
        self.domains = list(domain_positions.keys())
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.domains_per_batch = min(domains_per_batch, len(self.domains))
        self.rng = np.random.default_rng(seed)

        if sampling_weights is None:
            self.p = None

        else:
            weights = np.array([float(sampling_weights.get(d, 1.0)) for d in self.domains])
            self.p = weights / weights.sum()

        if self.domains_per_batch < 2:
            raise ValueError("Cross-domain §3.5 interventions require >=2 distinct domains per batch; "
                             "pass >=2 targets and --domains-per-batch >=2.")

        if batch_size < self.domains_per_batch:
            raise ValueError("batch_size must be >= domains_per_batch to place each chosen domain in the batch.")

    def __len__(self):

        return self.n_batches

    def __iter__(self):

        for _ in range(self.n_batches):
            chosen = self.rng.choice(len(self.domains), size=self.domains_per_batch, replace=False, p=self.p)
            picked = [self.domains[c] for c in chosen]
            batch = []

            for slot in range(self.batch_size):
                domain = picked[slot % self.domains_per_batch]
                positions = self.domain_positions[domain]
                batch.append(int(positions[self.rng.integers(len(positions))]))

            yield batch
