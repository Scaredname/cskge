import math
from collections.abc import Mapping
from typing import Any, ClassVar, Optional

from class_resolver import normalize_string
from pykeen.sampling.basic_negative_sampler import random_replacement_
from pykeen.sampling.filtering import Filterer
from pykeen.typing import BoolTensor, LongTensor
from torch import nn

from .cross_view_filter import CrossViewPythonSetFilterer


class CorssViewNegativeSampler(nn.Module):
    """A negative sampler."""

    #: The default strategy for optimizing the negative sampler's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Mapping[str, Any]]] = dict(
        num_negs_per_pos=dict(type=int, low=1, high=100, log=True),
    )

    #: A filterer for negative batches
    filterer: Optional[Filterer]

    num_categorized_entities: int
    num_categories: int
    num_negs_per_pos: int

    def __init__(
        self,
        *,
        mapped_pairs,
        num_categorized_entities: Optional[int],
        num_entities: Optional[int],
        num_categories: Optional[int],
        num_negs_per_pos: Optional[int] = None,
        filtered: bool = False,
        sample_all_entities: bool = False,
    ) -> None:
        """Initialize the negative sampler with the given entities.

        :param mapped_triples:
            the positive training triples
        :param num_entities:
            the number of entities. If None, will be inferred from the triples.
        :param num_relations:
            the number of relations. If None, will be inferred from the triples.
        :param num_negs_per_pos:
            number of negative samples to make per positive triple. Defaults to 1.
        :param filtered: Whether proposed corrupted triples that are in the training data should be filtered.
            Defaults to False. See explanation in :func:`filter_negative_triples` for why this is
            a reasonable default.
        :param filterer: If filtered is set to True, this can be used to choose which filter module from
            :mod:`pykeen.sampling.filtering` is used.
        :param filterer_kwargs:
            Additional keyword-based arguments passed to the filterer upon construction.
        """
        super().__init__()
        self.num_categorized_entities = num_categorized_entities
        self.num_entities = num_entities
        self.num_categories = num_categories
        self.num_negs_per_pos = num_negs_per_pos if num_negs_per_pos is not None else 1
        self.sample_all_entities = sample_all_entities

        self.filterer = (
            CrossViewPythonSetFilterer(mapped_pairs=mapped_pairs) if filtered else None
        )

        self._corruption_indices = [0, 1]

    @classmethod
    def get_normalized_name(cls) -> str:
        """Get the normalized name of the negative sampler."""
        return normalize_string(cls.__name__, suffix=CorssViewNegativeSampler.__name__)

    def sample(
        self, positive_batch: LongTensor
    ) -> tuple[LongTensor, Optional[BoolTensor]]:
        """
        Generate negative samples from the positive batch.

        :param positive_batch: shape: (batch_size, 2)
            The positive triples.

        :return:
            A pair `(negative_batch, filter_mask)` where

            1. `negative_batch`: shape: (batch_size, num_negatives, 2)
               The negative batch. ``negative_batch[i, :, :]`` contains the negative examples generated from
               ``positive_batch[i, :]``.
            2. filter_mask: shape: (batch_size, num_negatives)
               An optional filter mask. True where negative samples are valid.
        """
        # create unfiltered negative batch by corruption
        negative_batch = self.corrupt_batch(positive_batch=positive_batch)

        if self.filterer is None:
            return negative_batch, None

        # If filtering is activated, all negative triples that are positive in the training dataset will be removed
        return negative_batch, self.filterer(negative_batch=negative_batch)

    def corrupt_batch(self, positive_batch: LongTensor) -> LongTensor:
        """
        Generate negative samples from the positive batch without application of any filter.

        :param positive_batch: shape: `(*batch_dims, 2)`
            The positive triples.

        :return: shape: `(*batch_dims, num_negs_per_pos, 2)`
            The negative triples. ``result[*bi, :, :]`` contains the negative examples generated from
            ``positive_batch[*bi, :]``.
        """
        batch_shape = positive_batch.shape[:-1]

        # clone positive batch for corruption (.repeat_interleave creates a copy)
        negative_batch = positive_batch.view(-1, 2).repeat_interleave(
            self.num_negs_per_pos, dim=0
        )

        # Bind the total number of negatives to sample in this batch
        total_num_negatives = negative_batch.shape[0]

        # Allocate negative examples based on the ratio of entities to categories.
        split_idx = int(
            math.ceil(
                total_num_negatives
                * (
                    self.num_categorized_entities
                    / (self.num_categorized_entities + self.num_categories)
                )
            )
        )

        max_entities_id = (
            self.num_categorized_entities
            if not self.sample_all_entities
            else self.num_entities
        )

        # Do not detach, as no gradients should flow into the indices.
        for index, start in zip(
            self._corruption_indices, range(0, total_num_negatives, split_idx)
        ):
            stop = min(start + split_idx, total_num_negatives)
            random_replacement_(
                batch=negative_batch,
                index=index,
                selection=slice(start, stop),
                size=stop - start,
                max_index=(self.num_categories if index == 1 else max_entities_id),
            )

        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 2)
