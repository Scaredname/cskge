from typing import Optional, Sequence

import torch
from class_resolver.utils import (
    HintOrType,
    OneOrManyHintOrType,
    OneOrManyOptionalKwargs,
    OptionalKwargs,
)
from pykeen.models import ERModel
from pykeen.models.nbase import _prepare_representation_module_list, repeat_if_necessary
from pykeen.nn.init import (
    PretrainedInitializer,
    init_phases,
    xavier_uniform_,
    xavier_uniform_norm_,
)
from pykeen.nn.modules import RotatEInteraction, TransEInteraction, parallel_unsqueeze
from pykeen.nn.representation import Representation
from pykeen.regularizers import Regularizer
from pykeen.typing import Constrainer, Hint, InductiveMode, Initializer
from pykeen.utils import complex_normalize
from torch.nn.functional import sigmoid

from .category_triple_factory import CategoryTriplesFactory


class CategorySupplementedModel(ERModel):
    def __init__(
        self,
        *,
        triples_factory: CategoryTriplesFactory,
        interaction,
        loss,
        interaction_kwargs: OptionalKwargs = None,
        entity_representations: OneOrManyHintOrType[Representation] = None,
        entity_representations_kwargs: OneOrManyOptionalKwargs = None,
        relation_representations: OneOrManyHintOrType[Representation] = None,
        relation_representations_kwargs: OneOrManyOptionalKwargs = None,
        category_representations: OneOrManyHintOrType[Representation] = None,
        category_representations_kwargs: OneOrManyOptionalKwargs = None,
        skip_checks: bool = False,
        entity_preference_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
            entity_representations=entity_representations,
            entity_representations_kwargs=entity_representations_kwargs,
            relation_representations=relation_representations,
            relation_representations_kwargs=relation_representations_kwargs,
            skip_checks=skip_checks,
            loss=loss,
            **kwargs,
        )

        self.triples_factory = triples_factory

        self.category_representations = self._customize_build_representations(
            shape=("d",),
            max_id=triples_factory.num_categories,
            representations=category_representations,
            label="category",
            representations_kwargs=dict(
                shape=category_representations_kwargs["shape"],
                initializer=category_representations_kwargs["initializer"],
                constrainer=category_representations_kwargs["constrainer"],
                dtype=category_representations_kwargs["dtype"],
                trainable=True,
            ),
        )
        ent_category_weight = triples_factory.ents_cates_adj_matrix
        # self.ent_wrt_cat_weight = torch.nn.Parameter(
        #     1 / (torch.tensor(triples_factory.entity_frequency) + 1e-1), True
        # )
        # self.ent_wrt_cat_weight = torch.nn.Parameter(
        #     torch.FloatTensor(triples_factory.entity_frequency)
        #     - torch.mean(torch.FloatTensor(triples_factory.entity_frequency)),
        #     True,
        # )
        adj_sum = self.triples_factory.ents_cates_adj_matrix.sum(dim=1)
        # Make sure adj_sum is a float tensor if needed:
        adj_sum = adj_sum.float()

        # (1 - adj_sum) will be broadcasted elementwise;
        self.ent_wrt_cat_weight = torch.nn.Parameter(
            (1 - adj_sum) * 10, requires_grad=True
        )

        self.project_cat_to_ent_space = torch.nn.Sequential(
            torch.nn.Linear(
                category_representations_kwargs["shape"],
                entity_representations_kwargs["shape"],
            ),
            torch.nn.Tanh(),
        )

        self.entity_preference_weight = entity_preference_weight
        normal_noise = torch.empty_like(
            self.relation_representations[0]._embeddings.weight
        )
        torch.nn.init.normal_(normal_noise, mean=0, std=10)
        self.register_buffer("ent_category_weight", ent_category_weight)
        self.register_buffer("normal_noise", normal_noise)
        self.reset_parameters_()

    def _customize_build_representations(
        self,
        shape: Sequence[str],
        max_id: int,
        label: str,
        representations: OneOrManyHintOrType[Representation] = None,
        representations_kwargs: OneOrManyOptionalKwargs = None,
        **kwargs,
    ) -> Sequence[Representation]:
        return _prepare_representation_module_list(
            representations=representations,
            representations_kwargs=representations_kwargs,
            max_id=max_id,
            shapes=shape,
            label=label,
            **kwargs,
        )

    def score_cross_view(self, ent_cat_pair_batch):
        ent_index = ent_cat_pair_batch[:, 0]
        cat_index = ent_cat_pair_batch[:, 1]

        ent_emb = self.entity_representations[0](ent_index)
        cat_emb = self.category_representations[0](cat_index)

        return -(
            torch.norm((ent_emb - self.project_cat_to_ent_space(cat_emb)))
            / ent_emb.shape[0]  # 除以batch_size 得到 平均误差
        )

    def get_weighted_category_emb_wrt_ent(self, ent_index):
        category_emb = self.category_representations[0]._embeddings.weight
        return self.project_cat_to_ent_space(
            torch.matmul(
                self.ent_category_weight[ent_index],
                category_emb,
            )
        )

    def distance_between_cat_and_ent(self, ent_index):

        ent_index = ent_index.to(self.device)
        ent_emb = self.entity_representations[0]._embeddings(ent_index).detach()
        cat_wrt_ent_emb = self.get_weighted_category_emb_wrt_ent(ent_index)

        weight = self.ent_wrt_cat_weight[ent_index]

        # return torch.nn.MSELoss()(ent_emb, cat_wrt_ent_emb)
        return torch.norm(
            (1 - sigmoid(weight.unsqueeze(dim=-1))) * (ent_emb - cat_wrt_ent_emb)
        )

    def score_hrt(
        self, hrt_batch: torch.LongTensor, *, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:
        """Forward pass.

        This method takes head, relation and tail of each triple and calculates the corresponding score.

        :param hrt_batch: shape: (batch_size, 3), dtype: long
            The indices of (head, relation, tail) triples.
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.

        :return: shape: (batch_size, 1), dtype: float
            The score for each triple.
        """
        # Note: slicing cannot be used here: the indices for score_hrt only have a batch
        # dimension, and slicing along this dimension is already considered by sub-batching.
        # Note: we do not delegate to the general method for performance reasons
        # Note: repetition is not necessary here
        h_index = hrt_batch[:, 0]
        r_index = hrt_batch[:, 1]
        t_index = hrt_batch[:, 2]

        h, r, t = self._get_representations(h=h_index, r=r_index, t=t_index, mode=mode)

        return self.interaction.score_hrt(h=h, r=r, t=t)

    def score_hrt_with_cat(
        self, hrt_batch: torch.LongTensor, *, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:
        """Forward pass.

        This method takes head, relation and tail of each triple and calculates the corresponding score.

        :param hrt_batch: shape: (batch_size, 3), dtype: long
            The indices of (head, relation, tail) triples.
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.

        :return: shape: (batch_size, 1), dtype: float
            The score for each triple.
        """
        # Note: slicing cannot be used here: the indices for score_hrt only have a batch
        # dimension, and slicing along this dimension is already considered by sub-batching.
        # Note: we do not delegate to the general method for performance reasons
        # Note: repetition is not necessary here
        h_index = hrt_batch[:, 0]
        r_index = hrt_batch[:, 1]
        t_index = hrt_batch[:, 2]

        h, r, t = self._get_representations(h=h_index, r=r_index, t=t_index, mode=mode)

        head_cat_emb = self.get_weighted_category_emb_wrt_ent(h_index)
        tail_cat_emb = self.get_weighted_category_emb_wrt_ent(t_index)
        head_weight = sigmoid(self.ent_wrt_cat_weight[h_index].unsqueeze(dim=-1))
        tail_weight = sigmoid(self.ent_wrt_cat_weight[t_index].unsqueeze(dim=-1))

        h_score = head_weight * self.interaction.score_hrt(h=h, r=r, t=t) + (
            1 - head_weight
        ) * self.interaction.score_hrt(h=head_cat_emb, r=r, t=t)

        t_score = tail_weight * self.interaction.score_hrt(h=h, r=r, t=t) + (
            1 - tail_weight
        ) * self.interaction.score_hrt(h=h, r=r, t=tail_cat_emb)

        return 0.5 * (h_score + t_score)

    def score_t(
        self,
        hr_batch: torch.LongTensor,
        *,
        slice_size: Optional[int] = None,
        mode: Optional[InductiveMode] = None,
        tails: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        self._check_slicing(slice_size=slice_size)
        # add broadcast dimension
        hr_batch = hr_batch.unsqueeze(dim=1)
        h, r, t = self._get_representations(
            h=hr_batch[..., 0], r=hr_batch[..., 1], t=tails, mode=mode
        )

        head_cat_emb = self.get_weighted_category_emb_wrt_ent(hr_batch[..., 0])

        head_cat_emb = head_cat_emb.view(h.shape[0], h.shape[1], -1)

        # 确保t和t_type_emb的shape一致
        t = t.unsqueeze(dim=0).repeat(h.shape[0], 1, 1)

        head_cat_weight = sigmoid(self.ent_wrt_cat_weight[hr_batch[..., 0]])

        # unsqueeze if necessary
        if tails is None or tails.ndimension() == 1:
            if not len(t.shape) > 2:
                t = parallel_unsqueeze(t, dim=0)

        scores = head_cat_weight * self.interaction.score(
            h=h, r=r, t=t, slice_size=slice_size, slice_dim=1
        ).view(-1, self.num_entities) + (1 - head_cat_weight) * self.interaction.score(
            h=head_cat_emb, r=r, t=t, slice_size=slice_size, slice_dim=1
        ).view(
            -1, self.num_entities
        )

        # scores = self.interaction.score(
        #     h=h, r=r, t=t, slice_size=slice_size, slice_dim=1
        # ).view(-1, self.num_entities)

        return repeat_if_necessary(
            # score shape: (batch_size, num_entities)
            scores=scores,  # 会出现测试批度为1的特例，所以调整一下score的shape
            representations=self.entity_representations,
            num=self._get_entity_len(mode=mode) if tails is None else tails.shape[-1],
        )

    def score_h(
        self,
        rt_batch: torch.LongTensor,
        *,
        slice_size: Optional[int] = None,
        mode: Optional[InductiveMode] = None,
        heads: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        self._check_slicing(slice_size=slice_size)
        # add broadcast dimension
        rt_batch = rt_batch.unsqueeze(dim=1)
        h, r, t = self._get_representations(
            h=heads, r=rt_batch[..., 0], t=rt_batch[..., 1], mode=mode
        )

        tail_cat_emb = self.get_weighted_category_emb_wrt_ent(rt_batch[..., 1])

        tail_cat_weight = sigmoid(self.ent_wrt_cat_weight[rt_batch[..., 1]])

        tail_cat_emb = tail_cat_emb.view(t.shape[0], t.shape[1], -1)
        h = h.unsqueeze(dim=0).repeat(t.shape[0], 1, 1)

        if heads is None or heads.ndimension() == 1:
            if not len(h.shape) > 2:
                h = parallel_unsqueeze(h, dim=0)

        scores = tail_cat_weight * self.interaction.score(
            h=h, r=r, t=t, slice_size=slice_size, slice_dim=1
        ).view(-1, self.num_entities) + (1 - tail_cat_weight) * self.interaction.score(
            h=h, r=r, t=tail_cat_emb, slice_size=slice_size, slice_dim=1
        ).view(
            -1, self.num_entities
        )

        # scores = self.interaction.score(
        #     h=h, r=r, t=t, slice_size=slice_size, slice_dim=1
        # ).view(-1, self.num_entities)

        return repeat_if_necessary(
            scores=scores,  # 会出现测试批度为1的特例，所以调整一下score的shape
            representations=self.entity_representations,
            num=(self._get_entity_len(mode=mode) if heads is None else heads.shape[-1]),
        )


class CST(CategorySupplementedModel):
    def __init__(
        self,
        *,
        triples_factory: CategoryTriplesFactory,
        entity_representations=None,
        relation_representations=None,
        category_representations=None,
        skip_checks: bool = False,
        ent_dim: int = 50,
        rel_dim: int = 50,
        cat_dim: int = 20,
        scoring_fct_norm: int = 1,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = torch.nn.functional.normalize,
        relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
        relation_constrainer: Hint[Constrainer] = None,
        category_initializer: Hint[Initializer] = xavier_uniform_norm_,
        regularizer: HintOrType[Regularizer] = None,
        regularizer_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            interaction=TransEInteraction,
            interaction_kwargs=dict(p=scoring_fct_norm),
            entity_representations=entity_representations,
            entity_representations_kwargs=dict(
                shape=ent_dim,
                initializer=entity_initializer,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
                constrainer=entity_constrainer,
            ),
            relation_representations=relation_representations,
            relation_representations_kwargs=dict(
                shape=rel_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
            ),
            category_representations=category_representations,
            category_representations_kwargs=dict(
                shape=cat_dim,
                initializer=category_initializer,
                constrainer=torch.nn.functional.normalize,
                dtype=torch.float,
            ),
            skip_checks=skip_checks,
            **kwargs,
        )


class CSR(CategorySupplementedModel):
    def __init__(
        self,
        *,
        triples_factory: CategoryTriplesFactory,
        entity_representations=None,
        relation_representations=None,
        category_representations=None,
        skip_checks: bool = False,
        ent_dim: int = 50,
        rel_dim: int = 50,
        cat_dim: int = 20,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        relation_initializer: Hint[Initializer] = init_phases,
        relation_constrainer: Hint[Constrainer] = complex_normalize,
        category_initializer: Hint[Initializer] = xavier_uniform_norm_,
        regularizer: HintOrType[Regularizer] = None,
        regularizer_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            interaction=RotatEInteraction,
            entity_representations=entity_representations,
            entity_representations_kwargs=dict(
                shape=ent_dim,
                initializer=entity_initializer,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
                dtype=torch.float,
            ),
            relation_representations=relation_representations,
            relation_representations_kwargs=dict(
                shape=rel_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
                dtype=torch.float,
            ),
            category_representations=category_representations,
            category_representations_kwargs=dict(
                shape=cat_dim,
                initializer=category_initializer,
                constrainer=complex_normalize,
                dtype=torch.float,
            ),
            skip_checks=skip_checks,
            **kwargs,
        )
