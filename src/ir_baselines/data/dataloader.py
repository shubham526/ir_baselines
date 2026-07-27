from typing import Optional

import torch
from torch.utils.data import DataLoader

from .dataset import RankingDataset


def _seed_worker(worker_id: int) -> None:
    import random

    import numpy as np
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class RankingDataLoader(DataLoader):
    """
    Adds a seeded generator and seeded workers so that a shuffled training
    order is reproducible across runs, and `drop_last` so in-batch-negative
    training never sees a ragged final batch.
    """

    def __init__(self,
                 dataset: RankingDataset,
                 batch_size: int,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 seed: Optional[int] = None,
                 drop_last: bool = False,
                 pin_memory: bool = True,
                 ) -> None:
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate,
            generator=generator,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )