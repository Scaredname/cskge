import math
from abc import abstractmethod
from collections.abc import Iterable

import torch
from class_resolver import ClassResolver
from pykeen.sampling.filtering import Filterer
from pykeen.typing import BoolTensor, LongTensor, MappedTriples
from pykeen.utils import triple_tensor_to_set
from torch import nn


class CrossViewPythonSetFilterer(Filterer):
    """A filterer using Python sets for filtering.

    This filterer is expected to be rather slow due to the conversion from torch long tensors to Python tuples. It can
    still serve as a baseline for performance comparison.
    """

    def __init__(self, mapped_pairs):
        """Initialize the filterer.

        :param mapped_triples:
            The ID-based triples.
        """
        super().__init__()
        # store set of triples
        self.pairs = triple_tensor_to_set(mapped_pairs)

    # docstr-coverage: inherited
    def contains(self, batch: MappedTriples) -> BoolTensor:  # noqa: D102
        return torch.as_tensor(
            data=[tuple(pair) in self.pairs for pair in batch.view(-1, 2).tolist()],
            dtype=torch.bool,
            device=batch.device,
        ).view(*batch.shape[:-1])
