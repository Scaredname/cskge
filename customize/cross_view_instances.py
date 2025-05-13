import math
from collections.abc import Iterable, Iterator

import torch
from pykeen.triples.instances import SLCWABatch
from torch.utils import data

from .cross_view_negative_sampler import CorssViewNegativeSampler


class BatchedCrossViewInstances(data.IterableDataset[SLCWABatch]):
    """
    Pre-batched training instances for the sLCWA training loop.

    .. note ::
        this class is intended to be used with automatic batching disabled, i.e., both parameters `batch_size` and
        `batch_sampler` of torch.utils.data.DataLoader` are set to `None`.
    """

    def __init__(
        self,
        mapped_pairs: torch.LongTensor,
        num_categorized_entities: int,
        num_entities: int,
        num_categories: int,
        batch_size: int = 1,
        drop_last: bool = True,
        num_negs_per_pos: int = 1,
        sample_all_entities: bool = False,
    ):
        """
        Initialize the cross-view instances.

        :param mapped_pairs: shape: (num_instances, 2)
        """
        self.mapped_pairs = mapped_pairs
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.negative_sampler = CorssViewNegativeSampler(
            mapped_pairs=mapped_pairs,
            num_categorized_entities=num_categorized_entities,
            num_entities=num_entities,
            num_categories=num_categories,
            num_negs_per_pos=num_negs_per_pos,
            filtered=True,
            sample_all_entities=sample_all_entities,
        )

    def __getitem__(self, item: list[int]) -> SLCWABatch:
        """Get a batch from the given list of positive triple IDs."""
        positive_batch = self.mapped_pairs[item]
        negative_batch, masks = self.negative_sampler.sample(
            positive_batch=positive_batch
        )
        return SLCWABatch(
            positives=positive_batch, negatives=negative_batch, masks=masks
        )

    def split_workload(self, n: int) -> range:
        """Split workload for multi-processing."""
        # cf. https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            workload = range(n)
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id  # 1-based
            start = math.ceil(n / num_workers * worker_id)
            stop = math.ceil(n / num_workers * (worker_id + 1))
            workload = range(start, stop)
        return workload

    def iter_pair_ids(self) -> Iterable[list[int]]:
        """Iterate over batches of IDs of positive triples."""
        yield from data.BatchSampler(
            sampler=data.RandomSampler(
                data_source=self.split_workload(len(self.mapped_pairs))
            ),
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )

    def __iter__(self) -> Iterator[SLCWABatch]:
        """Iterate over batches."""
        for triple_ids in self.iter_pair_ids():
            yield self[triple_ids]

    def __len__(self) -> int:
        """Return the number of batches."""
        num_batches, remainder = divmod(len(self.mapped_pairs), self.batch_size)
        if remainder and not self.drop_last:
            num_batches += 1
        return num_batches
