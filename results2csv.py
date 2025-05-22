import ast
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import pandas as pd

# "{'embedding_dim': 512, 'random_seed': 3119831628, 'loss': NSSALoss(), 'entity_initializer': <function xavier_uniform_ at 0x7fc03c7503a0>, 'relation_initializer': <function init_phases at 0x7fc03c7245e0>, 'relation_constrainer': <function complex_normalize at 0x7fc03d7421f0>, 'regularizer': None, 'regularizer_kwargs': None}",

EMB_PATTERN = r"\'embedding_dim\': (\d+)\,"
EMB_PATTERN1 = r"\(_embeddings\): Embedding\((.*)\)"
SUB_PATTERN = r"(?<=, )\'.*?\),"
SUB_PATTERN1 = r"'[^']*'\s*:\s*<[^>]+>,"
BASE_PATTERN = r"(?<=\('interaction', )\S*(?=\(\))"


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_experiment_result_paths(
    results_root: str,
) -> Dict[Tuple[str, str, str], List[str]]:
    experiment_result_paths: Dict[Tuple[str, str, str], List[str]] = {}
    for dataset_name in os.listdir(results_root):
        dataset_path = os.path.join(results_root, dataset_name)
        if os.path.isdir(dataset_path):
            for model_name in os.listdir(dataset_path):
                model_path = os.path.join(dataset_path, model_name)

                if os.path.isdir(model_path):
                    for date in os.listdir(model_path):
                        date_path = os.path.join(model_path, date)

                        if os.path.isdir(date_path):
                            results_json_path = os.path.join(date_path, "results.json")
                            config_json_path = os.path.join(date_path, "config.json")

                            experiment_result_paths[
                                (dataset_name, model_name, date)
                            ] = [
                                results_json_path,
                                config_json_path,
                            ]

    return experiment_result_paths


def read_KGE_config(config: Dict[str, str]):
    contents = dict()
    contents["learning-rate"] = config["optimizer_kwargs"].split(",")[0].split(":")[1]

    try:
        model_dims = re.findall(EMB_PATTERN1, config["model_kwargs"])[0].split(",")[1]
    except:
        model_dims = eval(config["model_kwargs"])["embedding_dim"]

    contents["ent-dim"] = int(model_dims)
    contents["rel-dim"] = int(model_dims)

    contents["num_negs_per_pos"] = int(
        re.findall(r"'num_negs_per_pos': (\d+)", config["training_loop_kwargs"])[0]
    )
    contents["batch-size"] = config["batch_size"]
    if "loss" in config:
        contents["loss"] = str(config["loss"])
    else:
        contents["loss"] = "NSSALoss"

    contents["lr_scheduler"] = config["lr_scheduler"]
    if contents["lr_scheduler"] == "OCLR":
        scheduler_kwargs = eval(config["lr_scheduler_kwargs"])
        contents["max_lr"] = scheduler_kwargs["max_lr"]
        contents["start_lr"] = (
            scheduler_kwargs["max_lr"] / scheduler_kwargs["div_factor"]
        )
        contents["end_lr"] = contents["start_lr"] / (
            scheduler_kwargs["final_div_factor"]
        )
        contents["turning_epoch"] = (
            int(config["num_epochs"]) * scheduler_kwargs["pct_start"]
        )
        contents["anneal_strategy"] = scheduler_kwargs["anneal_strategy"]
    else:
        contents["max_lr"] = 0
        contents["start_lr"] = 0
        contents["end_lr"] = 0
        contents["anneal_strategy"] = ""

    loss_kwargs = re.sub(SUB_PATTERN, "", config["loss_kwargs"])
    loss_kwargs = re.sub(SUB_PATTERN1, "", loss_kwargs)
    loss_kwargs = ast.literal_eval(loss_kwargs)

    main_loss_kwargs = ["margin", "inverse_softmax_temperature"]
    if loss_kwargs:
        for kwarg in main_loss_kwargs:
            try:
                contents[kwarg] = float(loss_kwargs[kwarg])
            except:
                pass

        contents["other-loss-kwargs"] = str(
            {
                kwarg: loss_kwargs[kwarg]
                for kwarg in loss_kwargs
                if kwarg not in main_loss_kwargs
            }
        )
    else:
        for kwarg in main_loss_kwargs:
            contents[kwarg] = float(0)
        contents["other-loss-kwargs"] = ""

    if "description" in config:
        contents["description"] = config["description"]
    else:
        contents["description"] = ""
    return contents


def read_new_category_model_config(config: Dict[str, str]):
    contents = dict()
    contents["emb_dim"] = eval(config["model_kwargs"])["embedding_dim"]
    try:
        contents["cat_dim"] = eval(config["model_kwargs"])["cat_embedding_dim"]
    except:
        contents["cat_dim"] = contents["emb_dim"]

    try:
        contents["num_negs_per_pos"] = int(
            re.findall(r"'num_negs_per_pos': (\d+)", config["training_loop_kwargs"])[0]
        )
    except:
        contents["num_negs_per_pos"] = int(
            re.findall(r"'num_negs_per_pos': (\d+)", config["training_kwargs"])[0]
        )

    try:
        contents["num_negs_cross"] = int(
            re.findall(r"'num_negs_cross_view': (\d+)", config["training_kwargs"])[0]
        )
    except:
        contents["num_negs_cross"] = "-"

    contents["batch-size"] = config["batch_size"]
    if "loss" in config:
        contents["loss"] = str(config["loss"])
    else:
        contents["loss"] = "-"

    loss_kwargs = re.sub(SUB_PATTERN, "", config["loss_kwargs"])
    loss_kwargs = re.sub(SUB_PATTERN1, "", loss_kwargs)
    loss_kwargs = ast.literal_eval(loss_kwargs)

    main_loss_kwargs = ["margin", "inverse_softmax_temperature"]
    if loss_kwargs:
        for kwarg in main_loss_kwargs:
            try:
                contents[kwarg] = float(loss_kwargs[kwarg])
            except:
                pass

        contents["other-loss-kwargs"] = str(
            {
                kwarg: loss_kwargs[kwarg]
                for kwarg in loss_kwargs
                if kwarg not in main_loss_kwargs
            }
        )
    else:
        for kwarg in main_loss_kwargs:
            contents[kwarg] = float(0)
        contents["other-loss-kwargs"] = ""

    if "description" in config:
        contents["description"] = config["description"]
    else:
        contents["description"] = ""

    optimizer_outer_match = re.search(
        r"'optimizer_outer':\s*'([^']+)'", config["training_kwargs"]
    )
    optimizer_inner_match = re.search(
        r"'optimizer_inner':\s*'([^']+)'", config["training_kwargs"]
    )
    outer_lr_match = re.search(
        r"'optimizer_outer_kwargs':\s*\{\s*'lr':\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*\}",
        config["training_kwargs"],
    )
    inner_lr_match = re.search(
        r"'optimizer_inner_kwargs':\s*\{\s*'lr':\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*\}",
        config["training_kwargs"],
    )

    cv_lr = re.search(
        r"'cv_lr':\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        config["training_kwargs"],
    )

    inner_percentage = re.search(
        r"'inner_percentage':\s*([0-9.]*\.?[0-9]+)",
        config["training_kwargs"],
    )

    cv_ent_lr = re.search(
        r"'cv_ent_lr':\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        config["training_kwargs"],
    )

    optimizer_outer = optimizer_outer_match.group(1) if optimizer_outer_match else None
    optimizer_inner = optimizer_inner_match.group(1) if optimizer_inner_match else None
    outer_lr = float(outer_lr_match.group(1)) if outer_lr_match else None
    inner_lr = float(inner_lr_match.group(1)) if inner_lr_match else None
    cv_lr = float(cv_lr.group(1)) if cv_lr else None
    cv_ent_lr = float(cv_ent_lr.group(1)) if cv_ent_lr else None
    contents["inner_percentage"] = (
        float(inner_percentage.group(1)) if inner_percentage else None
    )

    contents["optimizer_outer"] = optimizer_outer
    contents["outer_lr"] = outer_lr
    contents["optimizer_inner"] = optimizer_inner
    contents["inner_lr"] = inner_lr
    contents["cv_lr"] = cv_lr
    contents["cv_ent_lr"] = cv_ent_lr

    return contents


def read_results_content(results: Dict[str, str]) -> List[str]:
    contents = dict()

    if "stopper" in results:
        contents["valid-mrr"] = round(float(results["stopper"]["best_metric"]), 3)
    else:
        contents["valid-mrr"] = 0

    contents["test-mrr"] = round(
        float(results["metrics"]["both"]["realistic"]["inverse_harmonic_mean_rank"]),
        3,
    )

    contents["mr"] = round(
        float(results["metrics"]["both"]["realistic"]["arithmetic_mean_rank"]),
        1,
    )

    contents["hits@1"] = round(
        float(results["metrics"]["both"]["realistic"]["hits_at_1"]),
        3,
    )

    contents["hits@3"] = round(
        float(results["metrics"]["both"]["realistic"]["hits_at_3"]),
        3,
    )
    contents["hits@10"] = round(
        float(results["metrics"]["both"]["realistic"]["hits_at_10"]),
        3,
    )

    return contents


def reindex_df(df: pd.DataFrame, prefix=["dataset", "model", "date"]):
    new_column_index = prefix + [
        index for index in list(df.columns) if index not in prefix
    ]
    return df[new_column_index]


if __name__ == "__main__":
    results_root = "./models/"

    # /home/ni/code/new_idea/models/FB15k237/rotate/20240623-204301/results.json

    experiment_result_paths = collect_experiment_result_paths(results_root)

    results_list = list()
    kge_results_config = list()
    category_framework_results_config = list()
    category_model_results_config = list()
    cs_model_results_config = list()

    save_config = True

    for dataset_name, m_name, date in experiment_result_paths:
        results = read_json(experiment_result_paths[(dataset_name, m_name, date)][0])
        config = read_json(experiment_result_paths[(dataset_name, m_name, date)][1])
        if "model" in config:
            model_name = config["model"]
        else:
            model_name = m_name

        results_contents = read_results_content(results)

        results_contents["dataset"] = dataset_name
        results_contents["model"] = model_name
        results_contents["date"] = date

        if "num_parameters" in config:
            results_contents["num_parameters"] = config["num_parameters"]
        else:
            results_contents["num_parameters"] = "-"

        results_list.append(results_contents.copy())

        if save_config:
            if "cs" in model_name.lower():
                cs_config = read_new_category_model_config(config)
                for c in cs_config:
                    results_contents[c] = cs_config[c]
                cs_model_results_config.append(results_contents.copy())
            else:
                kge_config = read_KGE_config(config)
                for c in kge_config:
                    results_contents[c] = kge_config[c]

                kge_results_config.append(results_contents.copy())

    # 修改为一个函数
    if not len(results_list):
        raise ValueError("No results!")
    all_results_df = reindex_df(pd.DataFrame(results_list))

    save_path = "./result/{name}.csv"
    if not os.path.exists("./result/"):
        os.mkdir("./result/")
    all_results_df = all_results_df.sort_values("date", ascending=False)
    all_results_df.to_csv(save_path.format(name="all_results"), index=False)

    if len(kge_results_config):
        kge_results = reindex_df(pd.DataFrame(kge_results_config))
        kge_results = kge_results.sort_values("date", ascending=False)
        kge_results.to_csv(save_path.format(name="kge_results"), index=False)
    else:
        print("no kge results")

    if len(cs_model_results_config):
        cs_model_results = reindex_df(pd.DataFrame(cs_model_results_config))
        cs_model_results = cs_model_results.sort_values("date", ascending=False)
        cs_model_results.to_csv(save_path.format(name="cs_model_results"), index=False)
    else:
        print("no cs model results")
