import os
from pykeen.triples import TriplesFactory
from customize.category_triple_factory import CategoryTriplesFactory


def get_key(dict, va):
    return [k for k, v in dict.item() if v == va]


def split_type_data(data: TriplesFactory):
    relations = list(data.relation_to_id.keys())
    relations.remove("category")

    return (
        data.label_triples(
            data.new_with_restriction(relations=relations).mapped_triples
        ),
        data.label_triples(
            data.new_with_restriction(relations=["category"]).mapped_triples
        ),
    )


def read_data(
    data_name,
    data_pro_func=split_type_data,
    create_inverse_triples=False,
    introducing_uncategory=False,
    cross_view_sample_all_entities=False,
):
    """
    @Params: data_name, data_pro_func, create_inverse_triples, type_position
    @Return: Train, Test, Valid
    """
    data_path = os.path.join(os.getcwd(), "./data/")

    train_path = os.path.join(data_path, "%s/" % (data_name), "train_cate.txt")
    valid_path = os.path.join(data_path, "%s/" % (data_name), "valid.txt")
    test_path = os.path.join(data_path, "%s/" % (data_name), "test.txt")

    training = TriplesFactory.from_path(
        train_path,
        create_inverse_triples=create_inverse_triples,
    )

    (
        training_triples,
        category_triples,
    ) = data_pro_func(training)
    training_data = CategoryTriplesFactory.from_labeled_triples(
        triples=training_triples,
        cate_triples=category_triples,
        introducing_uncategory=introducing_uncategory,
        cross_view_sample_all_entities=cross_view_sample_all_entities,
    )

    validation = TriplesFactory.from_path(
        valid_path,
        entity_to_id=training_data.entity_to_id,
        relation_to_id=training_data.relation_to_id,
        create_inverse_triples=create_inverse_triples,
    )
    testing = TriplesFactory.from_path(
        test_path,
        entity_to_id=training_data.entity_to_id,
        relation_to_id=training_data.relation_to_id,
        create_inverse_triples=create_inverse_triples,
    )

    return training_data, validation, testing
