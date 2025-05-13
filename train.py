# ==== 标准库 ====
import argparse
import datetime
import json
import logging
import os

# ==== 第三方库 ====
import torch

# ==== PyKEEN 核心 ====
import pykeen
import pykeen.datasets
from pykeen.datasets import get_dataset
from pykeen.evaluation import RankBasedEvaluator
from pykeen.losses import MarginPairwiseLoss, NSSALoss
from pykeen.models import model_resolver
from pykeen.sampling import negative_sampler_resolver

# ==== 自定义模块 ====
from customize.new_pipeline import pipeline
from customize.cs_model import CSR, CST
from customize.category_training_loop import (
    SLCWAWithReduceLROnPlateauLRScheduler,
    CategorySupplementarySLCWATrainingLoop,
)
from customize.stopper import EarlyStopperWithTrainingResults, PostponeEarlyStopper
from utilities import read_data


parser = argparse.ArgumentParser()

# === Dataset & Model Settings ===
parser.add_argument("-d", "--dataset", type=str, default="FB15k237")
parser.add_argument("-m", "--model", type=str, default="rotate")
parser.add_argument(
    "-lo", "--loss", type=str, choices=["nssa_loss", "mp_loss"], default="nssa_loss"
)
parser.add_argument(
    "-ed", "--emb_dim", type=int, default=50, help="entity embedding dimension"
)
parser.add_argument(
    "-ced", "--cat_emb_dim", type=int, default=5, help="category embedding dimension"
)
parser.add_argument("-lm", "--loss_margin", type=float, default=9.0)
parser.add_argument("-at", "--adversarial_temperature", type=float, default=1.0)
parser.add_argument("-i_per", "--inner_percentage", type=float, default=0.8)

# === Optimizer & Learning Rate Settings ===
parser.add_argument(
    "-o", "--optimizer", type=str, default="adam", help="optimizer for all stages"
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=0.001,
    help="learning rate for stage I",
)
parser.add_argument(
    "-lr_eta",
    "--learning_rate_eta",
    type=float,
    default=0.001,
    help="learning rate for stage III",
)
parser.add_argument(
    "-lr_beta",
    "--learning_rate_beta",
    type=float,
    default=0.001,
    help="learning rate for stage II (β)",
)
parser.add_argument(
    "-lr_kappa",
    "--learning_rate_kappa",
    type=float,
    default=0.001,
    help="learning rate for stage II (κ)",
)

parser.add_argument(
    "-lr_sch",
    "--lr_scheduler",
    type=str,
    default=None,
    choices=["OCLR", "RLRP"],
    help="learning rate scheduler: OCLR = One Cycle LR, RLRP = Reduce LR on Plateau",
)

# === Training Loop Settings ===
parser.add_argument("-train", "--training_loop", type=str, default="slcwa")
parser.add_argument("-b", "--batch_size", type=int, default=256)
parser.add_argument("-e", "--epochs", type=int, default=1000)
parser.add_argument(
    "-si",
    "--store_intermediate_results",
    action="store_true",
    default=False,
    help="store intermediate results for each epoch",
)

# === Negative Sampling ===
parser.add_argument("-neg", "--negative_sampler", type=str, default=None)
parser.add_argument(
    "-nen",
    "--num_negs_one",
    type=int,
    default=256,
    help="negatives per positive in stage I",
)
parser.add_argument(
    "-nenT",
    "--num_negs_two",
    type=int,
    default=16,
    help="negatives per positive in stage II",
)

# === Evaluation & Filtering ===
parser.add_argument("-ef", "--filtered", type=bool, default=True)
parser.add_argument(
    "-eot", "--evaluate_on_training", action="store_true", default=False
)
parser.add_argument("-eb", "--evaluator_batch_size", type=int, default=128)

# === Early Stopping & Regularization ===
parser.add_argument(
    "-stop", "--stopper", type=str, choices=["early", "nop"], default="early"
)
parser.add_argument("-r", "--regularization", type=float, default=0.0)
parser.add_argument("-rn", "--regularization_norm", type=int, default=2)

# === Memory & Logging ===
parser.add_argument(
    "-mf",
    "--memory_fraction",
    type=float,
    default=1.0,
    help="fraction of GPU memory to use",
)
parser.add_argument(
    "-de", "--description", type=str, default="", help="description of the experiment"
)

# === Misc ===
parser.add_argument("--random_seed", type=int, default=None)


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)

    if args.regularization > 0 and args.regularization_norm > 0:
        regularizer_kwargs = (
            dict(
                weight=args.regularization,
                p=args.regularization_norm,
                normalize=True,
            ),
        )
    else:
        regularizer_kwargs = None

    loss_resolver = dict(
        nssa_loss=NSSALoss,
        mp_loss=MarginPairwiseLoss,
    )

    if args.loss != "mp_loss":
        loss = loss_resolver[args.loss](
            reduction="mean",
            adversarial_temperature=args.adversarial_temperature,
            margin=args.loss_margin,
        )
    else:
        loss = loss_resolver[args.loss](
            reduction="mean",
            margin=args.loss_margin,
        )

    pipeline_config = dict(
        training_loop=args.training_loop,
        training_kwargs=dict(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
        ),
        negative_sampler=args.negative_sampler,
        negative_sampler_kwargs=dict(
            num_negs_per_pos=args.num_negs_one,
        ),
        evaluator="RankBasedEvaluator",
        evaluator_kwargs=dict(
            filtered=args.filtered,
            batch_size=args.evaluator_batch_size,
        ),
        stopper=args.stopper,
        stopper_kwargs=dict(
            frequency=10,
            patience=10,
            relative_delta=0.0001,
            metric="mean_reciprocal_rank",
            evaluation_batch_size=args.evaluator_batch_size,
        ),
        device="cuda",
        loss=loss,
        regularizer_kwargs=regularizer_kwargs,
        random_seed=args.random_seed,
    )

    date_time = "/%s/%s/%s" % (
        args.dataset,
        args.model,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    if args.dataset in [
        "yago_new",
        "NELL-995_new",
        "DB_new",
        "FB_new",
    ]:
        training, validation, testing = read_data(args.dataset)
        dataset = get_dataset(training=training, validation=validation, testing=testing)
    else:
        print(f"using dataset {args.dataset} without categories")
        dataset = pykeen.datasets.utils.get_dataset(dataset=args.dataset)

    evaluator_instance = RankBasedEvaluator(
        filtered=args.filtered,
        batch_size=args.evaluator_batch_size,
    )
    pipeline_config["evaluator"] = evaluator_instance

    # ===== read model =====
    is_cs_model = args.model.startswith("cs-")
    base_model_name = args.model.replace("cs-", "") if is_cs_model else args.model

    # ===== create model =====
    if is_cs_model:
        cs_model_resolver = {
            "transe": CST,
            "rotate": CSR,
        }
        model = cs_model_resolver[base_model_name](
            triples_factory=dataset.training,
            ent_dim=args.emb_dim,
            rel_dim=args.emb_dim,
            cat_dim=args.cat_emb_dim,
            loss=loss,
        )
    else:
        model = model_resolver.make(
            base_model_name,
            dict(
                embedding_dim=args.emb_dim,
                triples_factory=dataset.training,
                loss=loss,
            ),
        )

    # ===== prepare kwargs =====
    if is_cs_model:
        negative_sampler_cls = negative_sampler_resolver.lookup(
            pipeline_config["negative_sampler"]
        )
        training_kwargs = dict(
            model=model,
            triples_factory=training,
            num_negs_cross_view=args.num_negs_two,
            optimizer_outer=args.optimizer,
            optimizer_outer_kwargs=dict(lr=args.learning_rate_eta),
            optimizer_inner=args.optimizer,
            optimizer_inner_kwargs=dict(lr=args.learning_rate),
            inner_percentage=args.inner_percentage,
            negative_sampler=negative_sampler_cls,
            negative_sampler_kwargs=pipeline_config["negative_sampler_kwargs"],
            store_intermediate_results=args.store_intermediate_results,
            cv_lr=args.learning_rate_beta,
            cv_ent_lr=args.learning_rate_kappa,
        )
    else:
        training_kwargs = dict(
            model=model,
            triples_factory=training,
        )

    # ===== lr scheduler =====
    lr_scheduler = None
    lr_scheduler_kwargs = None
    using_LROnPlateau = False

    if args.lr_scheduler == "OCLR":
        lr_scheduler = "oneCycleLR"
        lr_scheduler_kwargs = dict(
            max_lr=args.learning_rate,
            epochs=args.epochs,
            steps_per_epoch=1,  # PyKEEN 中每个 epoch 通常只有一个 step
            anneal_strategy="cos",
            cycle_momentum=False,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000,
        )
        postpone_stopper = PostponeEarlyStopper(
            model=model,
            evaluator=evaluator_instance,
            training_triples_factory=dataset.training,
            evaluation_triples_factory=dataset.validation,
            start_epoch=int(0.3 * args.epochs),
            **pipeline_config["stopper_kwargs"],
        )
        pipeline_config["stopper"] = postpone_stopper

    elif args.lr_scheduler == "RLRP":
        using_LROnPlateau = True
        lr_scheduler = None
        lr_scheduler_kwargs = None

    # ===== register training loop =====
    pipeline_config["training_loop_kwargs"] = dict(
        lr_scheduler=lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
    )

    if is_cs_model:
        pipeline_config["training_loop"] = CategorySupplementarySLCWATrainingLoop(
            using_LROnPlateau_lr_scheduler=using_LROnPlateau,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            **training_kwargs,
        )
    else:
        pipeline_config["training_loop"] = SLCWAWithReduceLROnPlateauLRScheduler(
            using_LROnPlateau_lr_scheduler=using_LROnPlateau,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            **training_kwargs,
        )

    if args.evaluate_on_training:
        new_stopper = EarlyStopperWithTrainingResults(
            model=model,
            evaluator=evaluator_instance,
            training_triples_factory=dataset.training,
            evaluation_triples_factory=dataset.validation,
            **pipeline_config["stopper_kwargs"],
        )
        pipeline_config["stopper"] = new_stopper

    pipeline_result = pipeline(
        dataset=dataset,
        model=model,
        **pipeline_config,
    )

    # Save results and configuration
    modelpath = "./models" + date_time
    for config in pipeline_result.configuration:
        pipeline_result.configuration[config] = str(
            pipeline_result.configuration[config]
        )
    pipeline_result.configuration["loss"] = type(pipeline_config["loss"]).__name__
    pipeline_result.configuration["loss_kwargs"] = str(pipeline_config["loss"].__dict__)

    pipeline_result.configuration["num_parameters"] = model.num_parameters
    if args.framework:
        pipeline_result.configuration["model_kwargs"] = str(model.__dict__)

    pipeline_result.configuration["random_seed"] = pipeline_result.random_seed
    pipeline_result.configuration["description"] = args.description
    pipeline_result.save_to_directory(modelpath)

    with open(os.path.join(modelpath, "config.json"), "w") as f:
        json.dump(pipeline_result.configuration, f, indent=1)
