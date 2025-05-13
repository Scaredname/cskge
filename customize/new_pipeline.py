from __future__ import annotations

import ftplib
import hashlib
import json
import logging
import os
import pathlib
import pickle
import time
from collections.abc import Collection, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import pandas as pd
import scipy.stats
import torch
from class_resolver.utils import OneOrManyHintOrType, OneOrManyOptionalKwargs
from pykeen.constants import PYKEEN_CHECKPOINTS, USER_DEFINED_CODE
from pykeen.datasets import get_dataset
from pykeen.datasets.base import Dataset
from pykeen.evaluation import Evaluator, MetricResults, evaluator_resolver
from pykeen.evaluation.evaluator import normalize_flattened_metric_results
from pykeen.losses import Loss, loss_resolver
from pykeen.lr_schedulers import LRScheduler, lr_scheduler_resolver
from pykeen.models import Model, make_model_cls, model_resolver
from pykeen.nn.modules import Interaction
from pykeen.optimizers import optimizer_resolver
from pykeen.pipeline.api import (
    PipelineResult,
    _handle_dataset,
    _handle_evaluation,
    _handle_evaluator,
    _handle_model,
    _handle_random_seed,
    _handle_training,
)
from pykeen.regularizers import Regularizer, regularizer_resolver
from pykeen.sampling import NegativeSampler, negative_sampler_resolver
from pykeen.stoppers import EarlyStopper, Stopper, stopper_resolver
from pykeen.trackers import ResultTracker, resolve_result_trackers
from pykeen.trackers.base import MultiResultTracker
from pykeen.training import SLCWATrainingLoop, TrainingLoop, training_loop_resolver
from pykeen.triples import CoreTriplesFactory
from pykeen.typing import DeviceHint, Hint, HintType, MappedTriples
from pykeen.utils import (
    Result,
    ensure_ftp_directory,
    fix_dataclass_init_docs,
    get_json_bytes_io,
    get_model_io,
    load_configuration,
    normalize_path,
    random_non_negative_int,
    resolve_device,
    set_random_seed,
)
from pykeen.version import get_git_hash, get_version
from torch.optim.optimizer import Optimizer


def _handle_training_loop(
    *,
    _result_tracker: ResultTracker,
    model_instance: Model,
    training: CoreTriplesFactory,
    # 5. Optimizer
    optimizer: HintType[Optimizer] = None,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    # 5.1 Learning Rate Scheduler
    lr_scheduler: HintType[LRScheduler] = None,
    lr_scheduler_kwargs: Mapping[str, Any] | None = None,
    # 6. Training Loop
    training_loop: HintType[TrainingLoop] = None,
    training_loop_kwargs: Mapping[str, Any] | None = None,
    negative_sampler: HintType[NegativeSampler] = None,
    negative_sampler_kwargs: Mapping[str, Any] | None = None,
) -> TrainingLoop:

    optimizer_kwargs = dict(optimizer_kwargs or {})
    optimizer_instance = optimizer_resolver.make(
        optimizer,
        optimizer_kwargs,
        params=model_instance.get_grad_params(),
    )
    for key, value in optimizer_instance.defaults.items():
        optimizer_kwargs.setdefault(key, value)
    _result_tracker.log_params(
        params=dict(
            optimizer=optimizer_instance.__class__.__name__,
            optimizer_kwargs=optimizer_kwargs,
        ),
    )

    lr_scheduler_instance: LRScheduler | None
    if lr_scheduler is None:
        lr_scheduler_instance = None
    else:
        lr_scheduler_instance = lr_scheduler_resolver.make(
            lr_scheduler,
            lr_scheduler_kwargs,
            optimizer=optimizer_instance,
        )
        _result_tracker.log_params(
            params=dict(
                lr_scheduler=lr_scheduler_instance.__class__.__name__,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
            ),
        )

    training_loop_cls = training_loop_resolver.lookup(training_loop)
    if training_loop_kwargs is None:
        training_loop_kwargs = {}

    if negative_sampler is None:
        negative_sampler_cls = None
        if negative_sampler_kwargs and issubclass(training_loop_cls, SLCWATrainingLoop):
            training_loop_kwargs = dict(training_loop_kwargs)
            training_loop_kwargs.update(
                negative_sampler_kwargs=negative_sampler_kwargs,
            )
    elif not issubclass(training_loop_cls, SLCWATrainingLoop):
        raise ValueError("Can not specify negative sampler with LCWA")
    else:
        negative_sampler_cls = negative_sampler_resolver.lookup(negative_sampler)
        training_loop_kwargs = dict(training_loop_kwargs)
        training_loop_kwargs.update(
            negative_sampler=negative_sampler_cls,
            negative_sampler_kwargs=negative_sampler_kwargs,
        )
        _result_tracker.log_params(
            params=dict(
                negative_sampler=negative_sampler_cls.__name__,
                negative_sampler_kwargs=negative_sampler_kwargs,
            ),
        )

    if isinstance(training_loop, TrainingLoop):
        training_loop.result_tracker = _result_tracker
        return training_loop
    training_loop_instance = training_loop_cls(
        model=model_instance,
        triples_factory=training,
        optimizer=optimizer_instance,
        lr_scheduler=lr_scheduler_instance,
        result_tracker=_result_tracker,
        **training_loop_kwargs,
    )
    _result_tracker.log_params(
        params=dict(
            training_loop=training_loop_instance.__class__.__name__,
            training_loop_kwargs=training_loop_kwargs,
        ),
    )
    return training_loop_instance


def pipeline(  # noqa: C901
    *,
    # 1. Dataset
    dataset: None | str | Dataset | type[Dataset] = None,
    dataset_kwargs: Mapping[str, Any] | None = None,
    training: Hint[CoreTriplesFactory] = None,
    testing: Hint[CoreTriplesFactory] = None,
    validation: Hint[CoreTriplesFactory] = None,
    evaluation_entity_whitelist: Collection[str] | None = None,
    evaluation_relation_whitelist: Collection[str] | None = None,
    # 2. Model
    model: None | str | Model | type[Model] = None,
    model_kwargs: Mapping[str, Any] | None = None,
    interaction: None | str | Interaction | type[Interaction] = None,
    interaction_kwargs: Mapping[str, Any] | None = None,
    dimensions: None | int | Mapping[str, int] = None,
    # 3. Loss
    loss: HintType[Loss] = None,
    loss_kwargs: Mapping[str, Any] | None = None,
    # 4. Regularizer
    regularizer: HintType[Regularizer] = None,
    regularizer_kwargs: Mapping[str, Any] | None = None,
    # 5. Optimizer
    optimizer: HintType[Optimizer] = None,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    clear_optimizer: bool = True,
    # 5.1 Learning Rate Scheduler
    lr_scheduler: HintType[LRScheduler] = None,
    lr_scheduler_kwargs: Mapping[str, Any] | None = None,
    # 6. Training Loop
    training_loop: HintType[TrainingLoop] = None,
    training_loop_kwargs: Mapping[str, Any] | None = None,
    negative_sampler: HintType[NegativeSampler] = None,
    negative_sampler_kwargs: Mapping[str, Any] | None = None,
    # 7. Training (ronaldo style)
    epochs: int | None = None,
    training_kwargs: Mapping[str, Any] | None = None,
    stopper: HintType[Stopper] = None,
    stopper_kwargs: Mapping[str, Any] | None = None,
    # 8. Evaluation
    evaluator: HintType[Evaluator] = None,
    evaluator_kwargs: Mapping[str, Any] | None = None,
    evaluation_kwargs: Mapping[str, Any] | None = None,
    # 9. Tracking
    result_tracker: OneOrManyHintOrType[ResultTracker] = None,
    result_tracker_kwargs: OneOrManyOptionalKwargs = None,
    # Misc
    metadata: dict[str, Any] | None = None,
    device: Hint[torch.device] = None,
    random_seed: int | None = None,
    use_testing_data: bool = True,
    evaluation_fallback: bool = False,
    filter_validation_when_testing: bool = True,
    use_tqdm: bool | None = None,
) -> PipelineResult:
    """Train and evaluate a model.

    :param dataset:
        The name of the dataset (a key for the :data:`pykeen.datasets.dataset_resolver`) or the
        :class:`pykeen.datasets.Dataset` instance. Alternatively, the training triples factory (``training``), testing
        triples factory (``testing``), and validation triples factory (``validation``; optional) can be specified.
    :param dataset_kwargs:
        The keyword arguments passed to the dataset upon instantiation
    :param training:
        A triples factory with training instances or path to the training file if a a dataset was not specified
    :param testing:
        A triples factory with training instances or path to the test file if a dataset was not specified
    :param validation:
        A triples factory with validation instances or path to the validation file if a dataset was not specified
    :param evaluation_entity_whitelist:
        Optional restriction of evaluation to triples containing *only* these entities. Useful if the downstream task
        is only interested in certain entities, but the relational patterns with other entities improve the entity
        embedding quality.
    :param evaluation_relation_whitelist:
        Optional restriction of evaluation to triples containing *only* these relations. Useful if the downstream task
        is only interested in certain relation, but the relational patterns with other relations improve the entity
        embedding quality.

    :param model:
        The name of the model, subclass of :class:`pykeen.models.Model`, or an instance of
        :class:`pykeen.models.Model`. Can be given as None if the ``interaction`` keyword is used.
    :param model_kwargs:
        Keyword arguments to pass to the model class on instantiation
    :param interaction: The name of the interaction class, a subclass of :class:`pykeen.nn.modules.Interaction`,
        or an instance of :class:`pykeen.nn.modules.Interaction`. Can not be given when there is also a model.
    :param interaction_kwargs:
        Keyword arguments to pass during instantiation of the interaction class. Only use with ``interaction``.
    :param dimensions:
        Dimensions to assign to the embeddings of the interaction. Only use with ``interaction``.

    :param loss:
        The name of the loss or the loss class.
    :param loss_kwargs:
        Keyword arguments to pass to the loss on instantiation

    :param regularizer:
        The name of the regularizer or the regularizer class.
    :param regularizer_kwargs:
        Keyword arguments to pass to the regularizer on instantiation

    :param optimizer:
        The name of the optimizer or the optimizer class. Defaults to :class:`torch.optim.Adagrad`.
    :param optimizer_kwargs:
        Keyword arguments to pass to the optimizer on instantiation
    :param clear_optimizer:
        Whether to delete the optimizer instance after training. As the optimizer might have additional memory
        consumption due to e.g. moments in Adam, this is the default option. If you want to continue training, you
        should set it to False, as the optimizer's internal parameter will get lost otherwise.

    :param lr_scheduler:
        The name of the lr_scheduler or the lr_scheduler class.
        Defaults to :class:`torch.optim.lr_scheduler.ExponentialLR`.
    :param lr_scheduler_kwargs:
        Keyword arguments to pass to the lr_scheduler on instantiation

    :param training_loop:
        The name of the training loop's training approach (``'slcwa'`` or ``'lcwa'``) or the training loop class.
        Defaults to :class:`pykeen.training.SLCWATrainingLoop`.
    :param training_loop_kwargs:
        Keyword arguments to pass to the training loop on instantiation
    :param negative_sampler:
        The name of the negative sampler (``'basic'`` or ``'bernoulli'``) or the negative sampler class.
        Only allowed when training with sLCWA.
        Defaults to :class:`pykeen.sampling.BasicNegativeSampler`.
    :param negative_sampler_kwargs:
        Keyword arguments to pass to the negative sampler class on instantiation

    :param epochs:
        A shortcut for setting the ``num_epochs`` key in the ``training_kwargs`` dict.
    :param training_kwargs:
        Keyword arguments to pass to the training loop's train function on call
    :param stopper:
        What kind of stopping to use. Default to no stopping, can be set to 'early'.
    :param stopper_kwargs:
        Keyword arguments to pass to the stopper upon instantiation.

    :param evaluator:
        The name of the evaluator or an evaluator class. Defaults to :class:`pykeen.evaluation.RankBasedEvaluator`.
    :param evaluator_kwargs:
        Keyword arguments to pass to the evaluator on instantiation
    :param evaluation_kwargs:
        Keyword arguments to pass to the evaluator's evaluate function on call

    :param result_tracker: Either none (will result in a Python result tracker),
        a single tracker (as either a class, instance, or string for class name), or a list
        of trackers (as either a class, instance, or string for class name
    :param result_tracker_kwargs: Either none (will use all defaults), a single dictionary
        (will be used for all trackers), or a list of dictionaries with the same length
        as the result trackers

    :param metadata:
        A JSON dictionary to store with the experiment
    :param use_testing_data:
        If true, use the testing triples. Otherwise, use the validation triples. Defaults to true - use testing triples.
    :param device: The device or device name to run on. If none is given, the device will be looked up with
        :func:`pykeen.utils.resolve_device`.
    :param random_seed: The random seed to use. If none is specified, one will be assigned before any code
        is run for reproducibility purposes. In the returned :class:`PipelineResult` instance, it can be accessed
        through :data:`PipelineResult.random_seed`.
    :param evaluation_fallback:
        If true, in cases where the evaluation failed using the GPU it will fall back to using a smaller batch size or
        in the last instance evaluate on the CPU, if even the smallest possible batch size is too big for the GPU.
    :param filter_validation_when_testing:
        If true, during the evaluating of the test dataset, validation triples are added to the set of known positive
        triples, which are filtered out when performing filtered evaluation following the approach described by
        [bordes2013]_. This should be explicitly set to false only in the scenario that you are training a single
        model using the pipeline and evaluating with the testing set, but never using the validation set for
        optimization at all. This is a very atypical scenario, so it is left as true by default to promote
        comparability to previous publications.
    :param use_tqdm:
        Globally set the usage of tqdm progress bars. Typically more useful to set to false, since the training
        loop and evaluation have it turned on by default.

    :returns: A pipeline result package.
    """
    if training_kwargs is None:
        training_kwargs = {}
    training_kwargs = dict(training_kwargs)

    _random_seed, clear_optimizer = _handle_random_seed(
        training_kwargs=training_kwargs,
        random_seed=random_seed,
        clear_optimizer=clear_optimizer,
    )
    set_random_seed(_random_seed)

    _result_tracker = resolve_result_trackers(result_tracker, result_tracker_kwargs)

    if not metadata:
        metadata = {}
    title = metadata.get("title")

    # Start tracking
    _result_tracker.start_run(run_name=title)

    training, testing, validation = _handle_dataset(
        _result_tracker=_result_tracker,
        dataset=dataset,
        dataset_kwargs=dataset_kwargs,
        training=training,
        testing=testing,
        validation=validation,
        evaluation_entity_whitelist=evaluation_entity_whitelist,
        evaluation_relation_whitelist=evaluation_relation_whitelist,
    )

    model_instance = _handle_model(
        device=device,
        _result_tracker=_result_tracker,
        _random_seed=_random_seed,
        training=training,
        model=model,
        model_kwargs=model_kwargs,
        interaction=interaction,
        interaction_kwargs=interaction_kwargs,
        dimensions=dimensions,
        loss=loss,
        loss_kwargs=loss_kwargs,
        regularizer=regularizer,
        regularizer_kwargs=regularizer_kwargs,
    )

    training_loop_instance = _handle_training_loop(
        _result_tracker=_result_tracker,
        model_instance=model_instance,
        training=training,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        lr_scheduler=lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        training_loop=training_loop,
        training_loop_kwargs=training_loop_kwargs,
        negative_sampler=negative_sampler,
        negative_sampler_kwargs=negative_sampler_kwargs,
    )

    evaluator_instance, evaluation_kwargs = _handle_evaluator(
        _result_tracker=_result_tracker,
        evaluator=evaluator,
        evaluator_kwargs=evaluator_kwargs,
        evaluation_kwargs=evaluation_kwargs,
    )

    stopper_instance, configuration, losses, train_seconds = _handle_training(
        _result_tracker=_result_tracker,
        training=training,
        validation=validation,
        model_instance=model_instance,
        evaluator_instance=evaluator_instance,
        training_loop_instance=training_loop_instance,
        clear_optimizer=clear_optimizer,
        evaluation_kwargs=evaluation_kwargs,
        epochs=epochs,
        training_kwargs=training_kwargs,
        stopper=stopper,
        stopper_kwargs=stopper_kwargs,
        use_tqdm=use_tqdm,
    )
    metric_results, evaluate_seconds = _handle_evaluation(
        _result_tracker=_result_tracker,
        model_instance=model_instance,
        evaluator_instance=evaluator_instance,
        stopper_instance=stopper_instance,
        training=training,
        testing=testing,
        validation=validation,
        training_kwargs=training_kwargs,
        evaluation_kwargs=evaluation_kwargs,
        use_testing_data=use_testing_data,
        evaluation_fallback=evaluation_fallback,
        filter_validation_when_testing=filter_validation_when_testing,
        use_tqdm=use_tqdm,
    )
    _result_tracker.end_run()

    return PipelineResult(
        random_seed=_random_seed,
        model=model_instance,
        training=training,
        training_loop=training_loop_instance,
        losses=losses,
        stopper=stopper_instance,
        configuration=configuration,
        metric_results=metric_results,
        metadata=metadata,
        train_seconds=train_seconds,
        evaluate_seconds=evaluate_seconds,
    )
