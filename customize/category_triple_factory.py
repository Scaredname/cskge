import logging
import pathlib
import re
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import (
    Any,
    ClassVar,
    Collection,
    Dict,
    Mapping,
    MutableMapping,
    Optional,
    TextIO,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch
from pykeen.constants import COLUMN_LABELS
from pykeen.triples.analysis import get_entity_counts
from pykeen.triples.triples_factory import (
    INVERSE_SUFFIX,
    TriplesFactory,
    _map_triples_elements_to_ids,
)
from pykeen.typing import (
    LABEL_HEAD,
    LABEL_RELATION,
    LABEL_TAIL,
    EntityMapping,
    LabeledTriples,
    RelationMapping,
)
from pykeen.utils import compact_mapping
from torch.utils.data import Dataset

from .cross_view_instances import BatchedCrossViewInstances

logger = logging.getLogger(__name__)


def create_categorized_entity_mapping(
    category_triples: LabeledTriples,
) -> EntityMapping:
    """Create mapping from entity labels to IDs.

    :param triples: shape: (n, 3), dtype: str
    :returns:
        A mapping of entity labels to indices
    """
    # Split triples
    ents, cats = category_triples[:, 0], category_triples[:, 2]
    # Sorting ensures consistent results when the triples are permuted
    entity_labels = sorted(set(ents))
    category_labels = sorted(set(cats))

    ent_mapping = {label: i for i, label in enumerate(entity_labels)}
    cat_mapping = {label: i for i, label in enumerate(category_labels)}
    # Create mapping
    return ent_mapping, cat_mapping


def create_entity_mapping(
    triples: LabeledTriples, categorized_ent_ids=None
) -> EntityMapping:
    """Create mapping from entity labels to IDs.

    :param triples: shape: (n, 3), dtype: str
    :returns:
        A mapping of entity labels to indices
    """
    # Split triples
    heads, tails = triples[:, 0], triples[:, 2]
    # Sorting ensures consistent results when the triples are permuted
    entity_labels = sorted(set(heads).union(tails))

    # Create mapping
    if categorized_ent_ids is not None:
        if len(categorized_ent_ids) >= len(entity_labels):
            entity_labels_dict = dict()
            for label in categorized_ent_ids:
                if label in entity_labels:
                    entity_labels_dict[label] = len(entity_labels_dict)
            return entity_labels_dict
        else:
            for label in entity_labels:
                if label not in categorized_ent_ids:
                    categorized_ent_ids[label] = len(categorized_ent_ids)
            return categorized_ent_ids
    else:
        return {str(label): i for (i, label) in enumerate(entity_labels)}


def create_relation_mapping(relations: Iterable[str]) -> RelationMapping:
    """Create mapping from relation labels to IDs.

    :param relations: A set of relation labels
    :returns:
        A mapping of relation labels to indices
    """
    # Sorting ensures consistent results when the triples are permuted
    relation_labels = sorted(
        set(relations),
        key=lambda x: (
            re.sub(f"{INVERSE_SUFFIX}$", "", x),
            x.endswith(f"{INVERSE_SUFFIX}"),  # 有后缀的会被放在后面
        ),
    )
    # Create mapping
    return {str(label): i for (i, label) in enumerate(relation_labels)}


def L1_normalize_each_rows_of_matrix(matrix: np.array) -> np.array:
    """
    description:
    param matrix: np.float32
    return {np.float32}
    """

    for i in range(matrix.shape[0]):
        if np.sum(abs(matrix[i]), dtype=np.float32) > 0:
            matrix[i] = matrix[i] / np.sum(abs(matrix[i]), dtype=np.float32)
    return matrix


def category_to_id(cate_triples: np.array):

    categories = np.unique(np.ndarray.flatten(cate_triples[:, 2]))
    category_to_id: Dict[str, int] = {
        value: key for key, value in enumerate(categories)
    }
    return category_to_id


def create_adjacency_matrix_of_entities_relations(
    mapped_triples: np.array,
    ent_num,
    rel_num,
):
    adjacency_matrix = np.zeros([2, ent_num, rel_num], dtype=np.float32)
    for h, r, t in mapped_triples:
        adjacency_matrix[0, h, r] = 1
        adjacency_matrix[1, t, r] = 1

    return adjacency_matrix


def create_adjacency_matrix_of_entities_categories(
    cate_triples: np.array,
    entity_to_id: EntityMapping,
    category_to_id,
    introducting_uncategory: bool = False,
):
    """
    Create matrix where each row corresponds to an entity and each column to a cate.
    """
    # Prepare literal matrix, set every cate to zero, and afterwards fill in the corresponding value if available
    ents_cates_adj_matrix = np.zeros(
        [len(entity_to_id), len(category_to_id)], dtype=np.float32
    )
    if introducting_uncategory:
        uncategorized_column = np.ones(
            (ents_cates_adj_matrix.shape[0], 1), dtype=np.float32
        )
        ents_cates_adj_matrix = np.hstack((ents_cates_adj_matrix, uncategorized_column))

    for ent, _, cate in cate_triples:
        # row define entity, and column the cate
        try:
            ents_cates_adj_matrix[entity_to_id[ent], category_to_id[cate]] = 1
            # if there are any categories, it is not uncategory
            if introducting_uncategory:
                ents_cates_adj_matrix[entity_to_id[ent], -1] = 0
        except:
            # There are some entities not in training set
            pass

    return ents_cates_adj_matrix


def create_adjacency_matrix_of_relations_categories(
    ents_cates_adj_matrix,
    rel_num,
    cat_num,
    entities_relations_adjacency,
    introducting_uncategory: bool = False,
):
    if introducting_uncategory:
        rels_cats_adj_matrix = np.zeros([2, rel_num, cat_num + 1], dtype=np.float32)
    else:
        rels_cats_adj_matrix = np.zeros([2, rel_num, cat_num], dtype=np.float32)
    for e in range(ents_cates_adj_matrix.shape[0]):
        for c in np.where(ents_cates_adj_matrix[e, :] != 0)[0]:
            for r in np.where(entities_relations_adjacency[0, e, :] == 1)[0]:
                rels_cats_adj_matrix[0, r, c] += 1
            for r in np.where(entities_relations_adjacency[1, e, :] == 1)[0]:
                rels_cats_adj_matrix[1, r, c] += 1

    return rels_cats_adj_matrix


def id_ent2cat_cat2ent(ents_cates_adj_matrix):
    ent2cat = {}
    ent_cat_pairs = []
    for i, row in enumerate(ents_cates_adj_matrix):
        cat_ids = list(np.where(row)[0])
        ent2cat[i] = cat_ids
        for cat_id in cat_ids:
            ent_cat_pairs.append((i, cat_id))

    ent2cat = {i: list(np.where(row)[0])}
    cat2ent = {
        j: list(np.where(ents_cates_adj_matrix[:, j])[0])
        for j in range(ents_cates_adj_matrix.shape[1])
    }
    return ent2cat, cat2ent, ent_cat_pairs


def calculate_injective_confidence(
    df: pd.DataFrame,
    source: str,
    target: str,
) -> Tuple[int, float]:
    """
    Calcualte the confidence of wheter there is injective mapping from source to target.

    :param df:
        The dataframe.
    :param source:
        The source column.
    :param target:
        The target column.

    :return:
        the relative frequency of unique target per source.
    """
    grouped = df.groupby(by=source)
    n_unique = grouped.agg({target: "nunique"})[target]
    conf = (n_unique <= 1).mean()
    return conf


def maximum_reciprocal(
    df: pd.DataFrame,
    source: str,
    target: str,
    mu: int = 3,
) -> Tuple[int, float]:
    """
    description:
        calculate the reciprocal of maximum of target per source
    param df:
        The dataframe.
    param source:
        The source column.
    param target:
        The target column.
    return {*}
    """
    grouped = df.groupby(by=source)
    n_unique = grouped.agg({target: "nunique"})[target]
    return 1 / n_unique.max() ** mu


def average_target(
    df: pd.DataFrame,
    source: str,
    target: str,
) -> Tuple[int, float]:
    """
    description:
        calculate the mean of target per source
    param df:
        The dataframe.
    param source:
        The source column.
    param target:
        The target column.
    return {*}
    """
    grouped = df.groupby(by=source)
    n_unique = grouped.agg({target: "nunique"})[target]
    return n_unique.mean()


def create_relation_injective_confidence(
    mapped_triples: Collection[Tuple[int, int, int]], our_mu: int = 3
):
    """
    return: injective_confidence: [head2tail, tail2head], showcase: Dataframe
    """
    injective_confidence = list()
    df = pd.DataFrame(data=mapped_triples, columns=COLUMN_LABELS)
    showcase_df = defaultdict(list)
    for relation, group in df.groupby(by=LABEL_RELATION):
        # groupby relation_id, from 0 to relations_num
        h_IJC = calculate_injective_confidence(
            df=group, source=LABEL_TAIL, target=LABEL_HEAD
        )
        t_IJC = calculate_injective_confidence(
            df=group, source=LABEL_HEAD, target=LABEL_TAIL
        )

        h_RMT = maximum_reciprocal(
            df=group, source=LABEL_TAIL, target=LABEL_HEAD, mu=our_mu
        )
        t_RMT = maximum_reciprocal(
            df=group, source=LABEL_HEAD, target=LABEL_TAIL, mu=our_mu
        )
        h_AT = average_target(df=group, source=LABEL_TAIL, target=LABEL_HEAD)
        t_AT = average_target(df=group, source=LABEL_HEAD, target=LABEL_TAIL)

        if h_IJC > (1 - h_RMT):
            h_confi = h_IJC
        else:
            h_confi = min(h_IJC, h_RMT)
        if t_IJC > (1 - t_RMT):
            t_confi = t_IJC
        else:
            t_confi = min(t_IJC, t_RMT)

        injective_confidence.append((h_confi, t_confi))
        showcase_df["h_IJC"].append(h_IJC)
        showcase_df["h_AT"].append(h_AT)
        showcase_df["h_confi"].append(h_confi)
        showcase_df["t_IJC"].append(t_IJC)
        showcase_df["t_AT"].append(t_AT)
        showcase_df["t_confi"].append(t_confi)

    return np.array(injective_confidence), pd.DataFrame(showcase_df)


class CategoryTriplesFactory(TriplesFactory):
    file_name_category_to_id: ClassVar[str] = "cates_to_id"
    file_name_ents_cates: ClassVar[str] = "ents_cates"
    file_name_rels_cates: ClassVar[str] = "rels_cates"
    cate_triples_file_name: ClassVar[str] = "cate_triples"
    file_name_entity_to_id: ClassVar[str] = "entity_to_id"
    file_name_relation_to_id: ClassVar[str] = "relation_to_id"
    file_name_relation_injective_confidence: ClassVar[str] = (
        "relation_injective_confidence"
    )
    file_name_id_ent2cat_cat2ent: ClassVar[str] = "id_ent2cat_cat2ent"

    def __init__(
        self,
        *,
        ents_cates_adj_matrix: np.ndarray,
        rels_cats_adj_matrix=np.ndarray,
        categories_to_ids: Mapping[str, int],
        id_ent2cat_cat2ent: Tuple[dict, dict],
        relation_injective_confidence: Collection[Tuple[float, float]],
        ent_cat_pairs: Collection[Tuple[int, int]],
        categorized_ent_num: int,
        cross_view_sample_all_entities: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.categories_to_ids = categories_to_ids
        self.ents_cates_adj_matrix = torch.from_numpy(ents_cates_adj_matrix)
        self.rels_cats_adj_matrix = torch.from_numpy(rels_cats_adj_matrix)
        self.ent2cat = id_ent2cat_cat2ent[0]
        self.cat2ent = id_ent2cat_cat2ent[1]
        self.relation_injective_confidence = relation_injective_confidence

        entity_freq_df = get_entity_counts(self.mapped_triples)
        summed_freq = entity_freq_df.groupby("entity_id", as_index=False)["count"].sum()
        self.entity_frequency = list(summed_freq["count"])
        self.ents_cates_pairs = torch.LongTensor(ent_cat_pairs)
        self.categorized_ent_num = categorized_ent_num
        self.cv_sample_all_entities = cross_view_sample_all_entities

    @classmethod
    def from_labeled_triples(
        cls,
        triples: LabeledTriples,
        create_inverse_triples=False,
        introducing_uncategory=False,
        entity_to_id: Optional[EntityMapping] = None,
        relation_to_id: Optional[RelationMapping] = None,
        compact_id: bool = True,
        *,
        cate_triples: LabeledTriples = None,
        filter_out_candidate_inverse_relations: bool = True,
        cross_view_sample_all_entities: bool = False,
    ) -> "TriplescatesFactory":
        if cate_triples is None:
            raise ValueError(f"{cls.__name__} requires cate_triples.")

        if filter_out_candidate_inverse_relations:
            unique_relations, inverse = np.unique(triples[:, 1], return_inverse=True)
            suspected_to_be_inverse_relations = {
                r for r in unique_relations if r.endswith(INVERSE_SUFFIX)
            }
            if len(suspected_to_be_inverse_relations) > 0:
                logger.warning(
                    f"Some triples already have the inverse relation suffix {INVERSE_SUFFIX}. "
                    f"Re-creating inverse triples to ensure consistency. You may disable this behaviour by passing "
                    f"filter_out_candidate_inverse_relations=False",
                )
                relation_ids_to_remove = [
                    i
                    for i, r in enumerate(unique_relations.tolist())
                    if r in suspected_to_be_inverse_relations
                ]
                mask = np.isin(
                    element=inverse, test_elements=relation_ids_to_remove, invert=True
                )
                logger.info(f"keeping {mask.sum() / mask.shape[0]} triples.")
                triples = triples[mask]

        # Generate entity mapping if necessary
        categorized_ent_num = 0
        if entity_to_id is None:
            if cate_triples is not None:
                categorized_ent_id, categories_to_ids = (
                    create_categorized_entity_mapping(category_triples=cate_triples)
                )
                categorized_ent_num = len(categorized_ent_id)
                entity_to_id = create_entity_mapping(
                    triples=triples, categorized_ent_ids=categorized_ent_id
                )
            else:
                entity_to_id = create_entity_mapping(triples=triples)
        if compact_id:
            entity_to_id = compact_mapping(mapping=entity_to_id)[0]

        if relation_to_id is None:
            relation_to_id = create_relation_mapping(triples[:, 1])
        if compact_id:
            relation_to_id = compact_mapping(mapping=relation_to_id)[0]

        mapped_triples = _map_triples_elements_to_ids(
            triples=triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

        # get entity and relation adjacency matrix
        if entity_to_id is not None:
            categories_to_ids = category_to_id(cate_triples)
        ents_cates_adj_matrix = create_adjacency_matrix_of_entities_categories(
            cate_triples=cate_triples,
            entity_to_id=entity_to_id,
            category_to_id=categories_to_ids,
            introducting_uncategory=introducing_uncategory,
        )
        ent2cat, cat2ent, ent_cat_pairs = id_ent2cat_cat2ent(ents_cates_adj_matrix)

        ent_rel_adj = create_adjacency_matrix_of_entities_relations(
            mapped_triples=mapped_triples,
            ent_num=len(entity_to_id),
            rel_num=len(relation_to_id),
        )

        rels_cats_adj_matrix = create_adjacency_matrix_of_relations_categories(
            ents_cates_adj_matrix=ents_cates_adj_matrix,
            rel_num=len(relation_to_id),
            cat_num=len(categories_to_ids),
            entities_relations_adjacency=ent_rel_adj,
            introducting_uncategory=introducing_uncategory,
        )

        if introducing_uncategory:
            categories_to_ids["uncategory"] = len(categories_to_ids)

        # Calculate the proportion of each cate.
        ents_cates_adj_matrix = L1_normalize_each_rows_of_matrix(ents_cates_adj_matrix)
        rels_cats_adj_matrix[0] = L1_normalize_each_rows_of_matrix(
            rels_cats_adj_matrix[0]
        )
        rels_cats_adj_matrix[1] = L1_normalize_each_rows_of_matrix(
            rels_cats_adj_matrix[1]
        )

        (
            relation_injective_confidence,
            injective_confidence_showcase,
        ) = create_relation_injective_confidence(mapped_triples)

        # TODO: Mapped cate_triples

        return cls(
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            mapped_triples=mapped_triples,
            create_inverse_triples=create_inverse_triples,
            ents_cates_adj_matrix=ents_cates_adj_matrix,
            rels_cats_adj_matrix=rels_cats_adj_matrix,
            categories_to_ids=categories_to_ids,
            id_ent2cat_cat2ent=(ent2cat, cat2ent),
            relation_injective_confidence=relation_injective_confidence,
            ent_cat_pairs=ent_cat_pairs,
            categorized_ent_num=min(categorized_ent_num, len(entity_to_id)),
            cross_view_sample_all_entities=cross_view_sample_all_entities,
        )

    @property
    def category_shape(self) -> Tuple[int, ...]:
        """Return the shape of the cates."""
        return self.ents_cates_adj_matrix.shape[1:]

    @property
    def num_categories(self) -> int:
        """Return the number of cates."""
        return self.ents_cates_adj_matrix.shape[1]

    def create_cross_view_instances(self, **kwargs) -> Dataset:
        """Create sLCWA instances for this factory's triples."""
        cls = BatchedCrossViewInstances
        if "shuffle" in kwargs:
            if kwargs.pop("shuffle"):
                warnings.warn(
                    "Training instances are always shuffled.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                raise AssertionError("If shuffle is provided, it must be True.")
        return cls(
            mapped_pairs=self.ents_cates_pairs,
            num_categorized_entities=self.categorized_ent_num,
            num_entities=self.num_entities,
            num_categories=self.num_categories,
            sample_all_entities=self.cv_sample_all_entities,
            **kwargs,
        )

    def to_path_binary(
        self, path: Union[str, pathlib.Path, TextIO]
    ) -> pathlib.Path:  # noqa: D102
        path = super().to_path_binary(path=path)
        # store entity/relation to ID
        for name, data in (
            (
                self.file_name_category_to_id,
                self.categories_to_ids,
            ),
        ):
            pd.DataFrame(
                data=data.items(),
                columns=["label", "id"],
            ).sort_values(
                by="id"
            ).set_index("id").to_csv(
                path.joinpath(f"{name}.tsv.gz"),
                sep="\t",
            )

        np.savez_compressed(
            path.joinpath(f"{self.file_name_ents_cates}.npz"),
            self.ents_cates_adj_matrix,
        )
        np.savez_compressed(
            path.joinpath(f"{self.file_name_relation_injective_confidence}.npz"),
            self.relation_injective_confidence,
        )

        np.savez_compressed(
            path.joinpath(f"{self.file_name_id_ent2cat_cat2ent}.npz"),
            np.array([self.ent2cat, self.cat2ent]),
        )

        return path

    @classmethod
    def _from_path_binary(cls, path: pathlib.Path) -> MutableMapping[str, Any]:
        data = super()._from_path_binary(path)
        # load entity/relation to ID
        for name in [
            cls.file_name_category_to_id,
        ]:
            df = pd.read_csv(
                path.joinpath(f"{name}.tsv.gz"),
                sep="\t",
            )
            data[name] = dict(zip(df["label"], df["id"]))

        data[cls.file_name_ents_cates] = np.load(
            path.joinpath(f"{cls.file_name_ents_cates}.npz")
        )["arr_0"]

        if "categories_to_ids" not in data:
            data["categories_to_ids"] = data.pop("cates_to_id")
        if "ents_cates_adj_matrix" not in data:
            data["ents_cates_adj_matrix"] = data.pop("ents_cates")
        if "relation_injective_confidence" not in data:
            data["relation_injective_confidence"] = (
                create_relation_injective_confidence(data["mapped_triples"])
            )
        if "id_ent2cat_cat2ent" not in data:
            data["id_ent2cat_cat2ent"] = id_ent2cat_cat2ent(
                data["ents_cates_adj_matrix"]
            )

        # print(list(data.keys()))

        return data
