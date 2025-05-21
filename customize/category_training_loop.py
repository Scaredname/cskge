# ==== 标准库 ====
import gc
import logging
import os
import pathlib
import time
from typing import List, Mapping, Optional, Any, Union

# ==== 第三方库 ====
import torch
from torch.nn.functional import logsigmoid
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from torch import FloatTensor

from class_resolver.contrib.torch import lr_scheduler_resolver, optimizer_resolver
from torch_max_mem.api import is_oom_error

# ==== PyKEEN ====
from pykeen.triples import CoreTriplesFactory
from pykeen.triples.instances import SLCWABatch
from pykeen.models import RGCN, Model
from pykeen.losses import Loss
from pykeen.training.slcwa import SLCWATrainingLoop
from pykeen.training.training_loop import (
    BatchType,
    _get_lr_scheduler_kwargs,
    _get_optimizer_kwargs,
    NonFiniteLossError,
    NoTrainingBatchError,
    SubBatchingNotSupportedError,
)
from pykeen.training.callbacks import (
    GradientAbsClippingTrainingCallback,
    GradientNormClippingTrainingCallback,
    LearningRateSchedulerTrainingCallback,
    MultiTrainingCallback,
    OptimizerTrainingCallback,
    StopperTrainingCallback,
    TrackerTrainingCallback,
    TrainingCallback,
    TrainingCallbackHint,
    TrainingCallbackKwargsHint,
)
from pykeen.stoppers import Stopper
from pykeen.utils import (
    format_relative_comparison,
    get_batchnorm_modules,
    get_preferred_device,
)
from torch.optim.optimizer import Optimizer

# ==== 自定义模块 ====
from customize.training_callbacks import (
    CVOptimizerTrainingCallback,
    EarlyStoppingWithLROnPlateuaCallback,
    InnerOptimizerTrainingCallback,
    OuterOptimizerTrainingCallback,
)
from .category_triple_factory import CategoryTriplesFactory


logger = logging.getLogger(__name__)


def _get_optimizer_kwargs(optimizer: Optimizer) -> Mapping[str, Any]:
    optimizer_kwargs = optimizer.state_dict()
    optimizer_kwargs = {
        key: value
        for key, value in optimizer_kwargs["param_groups"][0].items()
        if key not in ["params", "initial_lr", "max_lr", "min_lr"]
    }
    return optimizer_kwargs


class SLCWAWithReduceLROnPlateauLRScheduler(SLCWATrainingLoop):
    def __init__(self, using_LROnPlateau_lr_scheduler=True, **kwargs):
        super().__init__(**kwargs)
        self.using_LRonPlateau = using_LROnPlateau_lr_scheduler

        self.lr_scheduler_kwargs = kwargs["lr_scheduler_kwargs"]
        self.lr_scheduler_name = kwargs["lr_scheduler"]

    def _train(  # noqa: C901
        self,
        triples_factory: CoreTriplesFactory,
        num_epochs: int = 1,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        label_smoothing: float = 0.0,
        sampler: Optional[str] = None,
        continue_training: bool = False,
        only_size_probing: bool = False,
        use_tqdm: bool = True,
        use_tqdm_batch: bool = True,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
        stopper: Optional[Stopper] = None,
        sub_batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        save_checkpoints: bool = False,
        checkpoint_path: Union[None, str, pathlib.Path] = None,
        checkpoint_frequency: Optional[int] = None,
        checkpoint_on_failure_file_path: Union[None, str, pathlib.Path] = None,
        best_epoch_model_file_path: Optional[pathlib.Path] = None,
        last_best_epoch: Optional[int] = None,
        drop_last: Optional[bool] = None,
        callbacks: TrainingCallbackHint = None,
        callbacks_kwargs: TrainingCallbackKwargsHint = None,
        gradient_clipping_max_norm: Optional[float] = None,
        gradient_clipping_norm_type: Optional[float] = None,
        gradient_clipping_max_abs_value: Optional[float] = None,
        pin_memory: bool = True,
    ) -> Optional[List[float]]:
        """Train the KGE model, see docstring for :func:`TrainingLoop.train`."""
        if self.optimizer is None:
            raise ValueError("optimizer must be set before running _train()")
        # When using early stopping models have to be saved separately at the best epoch, since the training loop will
        # due to the patience continue to train after the best epoch and thus alter the model
        # -> the temporay file has to be created outside, which we assert here
        if stopper is not None and not only_size_probing and last_best_epoch is None:
            assert best_epoch_model_file_path is not None

        if isinstance(self.model, RGCN) and sampler != "schlichtkrull":
            logger.warning(
                'Using RGCN without graph-based sampling! Please select sampler="schlichtkrull" instead of %s.',
                sampler,
            )

        # Prepare all of the callbacks
        callback = MultiTrainingCallback(
            callbacks=callbacks, callbacks_kwargs=callbacks_kwargs
        )
        # Register a callback for the result tracker, if given
        if self.result_tracker is not None:
            callback.register_callback(TrackerTrainingCallback())
        # Register a callback for the early stopper, if given
        # TODO should mode be passed here?
        if stopper is not None and not self.using_LRonPlateau:
            callback.register_callback(
                StopperTrainingCallback(
                    stopper,
                    triples_factory=triples_factory,
                    last_best_epoch=last_best_epoch,
                    best_epoch_model_file_path=best_epoch_model_file_path,
                )
            )

        callback.register_training_loop(self)

        # Take the biggest possible training batch_size, if batch_size not set
        batch_size_sufficient = False
        if batch_size is None:
            if self.automatic_memory_optimization:
                # Using automatic memory optimization on CPU may result in undocumented crashes due to OS' OOM killer.
                if self.model.device.type == "cpu":
                    batch_size = 256
                    batch_size_sufficient = True
                    logger.info(
                        "Currently automatic memory optimization only supports GPUs, but you're using a CPU. "
                        "Therefore, the batch_size will be set to the default value '{batch_size}'",
                    )
                else:
                    batch_size, batch_size_sufficient = self.batch_size_search(
                        triples_factory=triples_factory
                    )
            else:
                batch_size = 256
                logger.info(
                    f"No batch_size provided. Setting batch_size to '{batch_size}'."
                )

        # This will find necessary parameters to optimize the use of the hardware at hand
        if (
            not only_size_probing
            and self.automatic_memory_optimization
            and not batch_size_sufficient
            and not continue_training
        ):
            # return the relevant parameters slice_size and batch_size
            sub_batch_size, slice_size = self.sub_batch_and_slice(
                batch_size=batch_size, sampler=sampler, triples_factory=triples_factory
            )

        if (
            sub_batch_size is None or sub_batch_size == batch_size
        ):  # by default do not split batches in sub-batches
            sub_batch_size = batch_size
        elif get_batchnorm_modules(self.model):  # if there are any, this is truthy
            raise SubBatchingNotSupportedError(self.model)

        model_contains_batch_norm = bool(get_batchnorm_modules(self.model))
        if batch_size == 1 and model_contains_batch_norm:
            raise ValueError(
                "Cannot train a model with batch_size=1 containing BatchNorm layers."
            )

        if drop_last is None:
            drop_last = model_contains_batch_norm

        # Force weight initialization if training continuation is not explicitly requested.
        if not continue_training:
            # Reset the weights
            self.model.reset_parameters_()
            # afterwards, some parameters may be on the wrong device
            self.model.to(get_preferred_device(self.model, allow_ambiguity=True))

            # Create new optimizer
            optimizer_kwargs = _get_optimizer_kwargs(self.optimizer)
            self.optimizer = self.optimizer.__class__(
                params=self.model.get_grad_params(),
                **optimizer_kwargs,
            )

            if self.lr_scheduler is not None:
                # Create a new lr scheduler and add the optimizer

                self.lr_scheduler = lr_scheduler_resolver.make(
                    self.lr_scheduler,
                    self.lr_scheduler_kwargs,
                    optimizer=self.optimizer,
                )
                # print(self.lr_scheduler.__dict__)
                # breakpoint()
        elif not self.optimizer.state:
            raise ValueError("Cannot continue_training without being trained once.")
        if stopper is not None and self.using_LRonPlateau:

            lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                factor=0.1,
                mode="max",
                patience=max(0, stopper.patience // 2 - 1),
                threshold=stopper.relative_delta,
                threshold_mode="abs",
            )

            callback.register_callback(
                EarlyStoppingWithLROnPlateuaCallback(
                    stopper,
                    [lr_plateau],
                    triples_factory=triples_factory,
                    last_best_epoch=last_best_epoch,
                    best_epoch_model_file_path=best_epoch_model_file_path,
                )
            )
        # Ensure the model is on the correct device
        self.model.to(get_preferred_device(self.model, allow_ambiguity=True))

        if num_workers is None:
            num_workers = 0

        _use_outer_tqdm = not only_size_probing and use_tqdm
        _use_inner_tqdm = _use_outer_tqdm and use_tqdm_batch

        # When size probing, we don't want progress bars
        if _use_outer_tqdm:
            # Create progress bar
            _tqdm_kwargs = dict(desc=f"Training epochs on {self.device}", unit="epoch")
            if tqdm_kwargs is not None:
                _tqdm_kwargs.update(tqdm_kwargs)
            epochs = trange(
                self._epoch + 1,
                1 + num_epochs,
                **_tqdm_kwargs,
                initial=self._epoch,
                total=num_epochs,
            )
        elif only_size_probing:
            epochs = range(1, 1 + num_epochs)
        else:
            epochs = range(self._epoch + 1, 1 + num_epochs)

        logger.debug(f"using stopper: {stopper}")

        train_data_loader = self._create_training_data_loader(
            triples_factory,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,  # always shuffle during training
            sampler=sampler,
        )
        if len(train_data_loader) == 0:
            raise NoTrainingBatchError()
        if drop_last and not only_size_probing:
            logger.info(
                "Dropping last (incomplete) batch each epoch (%s batches).",
                format_relative_comparison(part=1, total=len(train_data_loader)),
            )

        # optimizer callbacks
        pre_step_callbacks: List[TrainingCallback] = []
        if gradient_clipping_max_norm is not None:
            pre_step_callbacks.append(
                GradientNormClippingTrainingCallback(
                    max_norm=gradient_clipping_max_norm,
                    norm_type=gradient_clipping_norm_type,
                )
            )
        if gradient_clipping_max_abs_value is not None:
            pre_step_callbacks.append(
                GradientAbsClippingTrainingCallback(
                    clip_value=gradient_clipping_max_abs_value
                )
            )
        callback.register_callback(
            OptimizerTrainingCallback(
                only_size_probing=only_size_probing,
                pre_step_callbacks=pre_step_callbacks,
            )
        )
        if self.lr_scheduler is not None:
            callback.register_callback(LearningRateSchedulerTrainingCallback())

        # Save the time to track when the saved point was available
        last_checkpoint = time.time()
        best_epoch_model_checkpoint_file_path: Optional[pathlib.Path] = None

        # Training Loop
        for epoch in epochs:
            # When training with an early stopper the memory pressure changes, which may allow for errors each epoch
            try:
                # Enforce training mode
                self.model.train()

                # Batching
                # Only create a progress bar when not in size probing mode
                if _use_inner_tqdm:
                    batches = tqdm(
                        train_data_loader,
                        desc=f"Training batches on {self.device}",
                        leave=False,
                        unit="batch",
                    )
                else:
                    batches = train_data_loader

                epoch_loss = self._train_epoch(
                    batches=batches,
                    callbacks=callback,
                    sub_batch_size=sub_batch_size,
                    label_smoothing=label_smoothing,
                    slice_size=slice_size,
                    epoch=epoch,
                    only_size_probing=only_size_probing,
                )

                # When size probing we don't need the losses
                if only_size_probing:
                    return None

                # Track epoch loss
                self.losses_per_epochs.append(epoch_loss)

                # Print loss information to console
                if _use_outer_tqdm:
                    epochs.set_postfix(
                        {
                            "loss": self.losses_per_epochs[-1],
                            "prev_loss": (
                                self.losses_per_epochs[-2]
                                if epoch > 1
                                else float("nan")
                            ),
                        }
                    )

                # Save the last successful finished epoch
                self._epoch = epoch

            # When the training loop failed, a fallback checkpoint is created to resume training.
            except (MemoryError, RuntimeError) as e:
                # During automatic memory optimization only the error message is of interest
                if only_size_probing:
                    raise e

                logger.warning(
                    f"The training loop just failed during epoch {epoch} due to error {str(e)}."
                )
                if checkpoint_on_failure_file_path:
                    # When there wasn't a best epoch the checkpoint path should be None
                    if (
                        last_best_epoch is not None
                        and best_epoch_model_file_path is not None
                    ):
                        best_epoch_model_checkpoint_file_path = (
                            best_epoch_model_file_path
                        )
                    self._save_state(
                        path=checkpoint_on_failure_file_path,
                        stopper=stopper,
                        best_epoch_model_checkpoint_file_path=best_epoch_model_checkpoint_file_path,
                        triples_factory=triples_factory,
                    )
                    logger.warning(
                        "However, don't worry we got you covered. PyKEEN just saved a checkpoint when this "
                        f"happened at '{checkpoint_on_failure_file_path}'. To resume training from the checkpoint "
                        f"file just restart your code and pass this file path to the training loop or pipeline you "
                        f"used as 'checkpoint_file' argument.",
                    )
                # Delete temporary best epoch model
                if (
                    best_epoch_model_file_path is not None
                    and best_epoch_model_file_path.is_file()
                ):
                    os.remove(best_epoch_model_file_path)
                raise e

            # Includes a call to result_tracker.log_metrics
            callback.post_epoch(epoch=epoch, epoch_loss=epoch_loss)

            # If a checkpoint file is given, we check whether it is time to save a checkpoint
            if save_checkpoints and checkpoint_path is not None:
                minutes_since_last_checkpoint = (time.time() - last_checkpoint) // 60
                # MyPy overrides are because you should
                if (
                    minutes_since_last_checkpoint >= checkpoint_frequency  # type: ignore
                    or self._should_stop
                    or epoch == num_epochs
                ):
                    # When there wasn't a best epoch the checkpoint path should be None
                    if (
                        last_best_epoch is not None
                        and best_epoch_model_file_path is not None
                    ):
                        best_epoch_model_checkpoint_file_path = (
                            best_epoch_model_file_path
                        )
                    self._save_state(
                        path=checkpoint_path,
                        stopper=stopper,
                        best_epoch_model_checkpoint_file_path=best_epoch_model_checkpoint_file_path,
                        triples_factory=triples_factory,
                    )  # type: ignore
                    last_checkpoint = time.time()

            if self._should_stop:
                if (
                    last_best_epoch is not None
                    and best_epoch_model_file_path is not None
                ):
                    self._load_state(path=best_epoch_model_file_path)
                    # Delete temporary best epoch model
                    if pathlib.Path.is_file(best_epoch_model_file_path):
                        os.remove(best_epoch_model_file_path)
                return self.losses_per_epochs

        callback.post_train(losses=self.losses_per_epochs)

        # If the stopper didn't stop the training loop but derived a best epoch, the model has to be reconstructed
        # at that state
        if (
            stopper is not None
            and last_best_epoch is not None
            and best_epoch_model_file_path is not None
        ):
            self._load_state(path=best_epoch_model_file_path)
            # Delete temporary best epoch model
            if pathlib.Path.is_file(best_epoch_model_file_path):
                os.remove(best_epoch_model_file_path)

        return self.losses_per_epochs


class CategorySupplementarySLCWATrainingLoop(SLCWAWithReduceLROnPlateauLRScheduler):
    def __init__(
        self,
        inner_percentage,
        num_negs_cross_view,
        optimizer_outer,
        optimizer_outer_kwargs,
        optimizer_inner,
        optimizer_inner_kwargs,
        cv_lr=0.01,
        cv_ent_lr=0.01,
        using_LROnPlateau_lr_scheduler=True,
        store_intermediate_results=False,
        **kwargs,
    ):
        super().__init__(using_LROnPlateau_lr_scheduler, **kwargs)

        self.optimizer_outer = optimizer_resolver.make(
            optimizer_outer,
            pos_kwargs=optimizer_outer_kwargs,
            params=[self.model.ent_wrt_cat_weight],
        )

        named_params = dict(self.model.named_parameters())

        inner_params = [
            named_params["entity_representations.0._embeddings.weight"],
            named_params["relation_representations.0._embeddings.weight"],
        ]
        cv_params = [
            dict(params=named_params["project_cat_to_ent_space.0.weight"]),
            dict(params=named_params["project_cat_to_ent_space.0.bias"]),
            dict(params=named_params["category_representations.0._embeddings.weight"]),
            dict(
                params=named_params["entity_representations.0._embeddings.weight"],
                lr=cv_ent_lr,
            ),
        ]

        self.optimizer_inner = optimizer_resolver.make(
            optimizer_inner,
            pos_kwargs=optimizer_inner_kwargs,
            params=inner_params,
        )

        self.optimizer_cv = optimizer_resolver.make(
            optimizer_inner,
            pos_kwargs=dict(lr=cv_lr),
            params=cv_params,
        )
        self.num_negs_cross_view = num_negs_cross_view
        self.inner_percentage = inner_percentage

        self.store_intermediate_results = store_intermediate_results

    def _sub_batch_size_search(
        self,
        *,
        batch_size: int,
        sampler: Optional[str],
        triples_factory: CoreTriplesFactory,
    ) -> tuple[int, bool, bool]:
        """Find the allowable sub batch size for training with the current setting.

        This method checks if it is possible to train the model with the given training data and the desired batch size
        on the hardware at hand. If possible, the sub-batch size equals the batch size. Otherwise, the maximum
        permissible sub-batch size is determined.

        :param batch_size:
            The initial batch size to start with.
        :param sampler:
            The sampler (None or schlichtkrull)
        :param triples_factory:
            A triples factory

        :return:
            Tuple containing the sub-batch size to use and indicating if the search was finished, i.e. successfully
            without hardware errors, as well as if sub-batching is possible

        :raises RuntimeError:
            If a runtime error is raised during training
        """
        sub_batch_size = batch_size
        finished_search = False
        supports_sub_batching = True

        try:
            # The cache of the previous run has to be freed to allow accurate memory availability estimates
            self._free_graph_and_cache()
            logger.debug(f"Trying {batch_size=:_} for training now.")
            self._train(
                triples_factory=triples_factory,
                num_epochs=1,
                batch_size=int(1.1 * batch_size),
                sub_batch_size=int(
                    1.1 * sub_batch_size
                ),  # Here we use a larger batch size for caching allocator of Pytorch
                sampler=sampler,
                only_size_probing=True,
            )
        except RuntimeError as runtime_error:
            self._free_graph_and_cache()
            if not is_oom_error(runtime_error):
                raise runtime_error
            logger.debug(
                f"The batch_size {batch_size=:_} was too big, sub_batching is required."
            )
            sub_batch_size //= 2
        else:
            finished_search = True
            logger.debug("No sub-batching required.")

        if not finished_search:
            logger.info("Starting sub_batch_size search for training now...")
            if get_batchnorm_modules(self.model):  # if there are any, this is truthy
                logger.info("This model does not support sub-batching.")
                supports_sub_batching = False
                sub_batch_size = batch_size
            else:
                while True:
                    logger.debug(f"Trying {sub_batch_size=:_} now.")
                    try:
                        self._free_graph_and_cache()
                        self._train(
                            num_epochs=1,
                            batch_size=int(batch_size),
                            sub_batch_size=sub_batch_size,
                            sampler=sampler,
                            only_size_probing=True,
                            triples_factory=triples_factory,
                        )
                    except RuntimeError as runtime_error:
                        self._free_graph_and_cache()
                        if not is_oom_error(runtime_error):
                            raise runtime_error
                        if sub_batch_size == 1:
                            logger.info(
                                f"Even {sub_batch_size=:_} does not fit in memory with these parameters"
                            )
                            break
                        logger.debug(
                            f"The {sub_batch_size=:_} was too big, trying less now."
                        )
                        sub_batch_size //= 2
                    else:
                        finished_search = True
                        logger.info(f"Concluded search with {sub_batch_size=:_}.")
                        break

        self._free_graph_and_cache()

        return sub_batch_size, finished_search, supports_sub_batching

    def _cross_view_forward_pass(
        self,
        batch: BatchType,
        start: int,
        stop: int,
        current_batch_size: int,
        label_smoothing: float,
        slice_size: Optional[int],
        backward: bool = True,
    ) -> float:
        # forward pass
        loss = self._process_cross_view_batch_static(
            model=self.model,
            batch=batch,
            start=start,
            stop=stop,
            slice_size=slice_size,
        )

        # raise error when non-finite loss occurs (NaN, +/-inf)
        if not torch.isfinite(loss):
            raise NonFiniteLossError("Loss is non-finite.")

        # correction for loss reduction
        if self.model.loss.reduction == "mean":
            this_sub_batch_size = stop - start
            loss *= this_sub_batch_size / current_batch_size

        # backward pass
        if backward:
            loss.backward()
        current_epoch_loss = loss.item()

        self.model.post_forward_pass()
        # TODO why not call torch.cuda.empty_cache()? or call self._free_graph_and_cache()?

        return current_epoch_loss

    def _forward_pass_outer(
        self,
        batch: BatchType,
        start: int,
        stop: int,
        current_batch_size: int,
        label_smoothing: float,
        slice_size: Optional[int],
        backward: bool = True,
    ) -> float:
        # forward pass
        loss = self._process_batch_outer(
            batch=batch,
            start=start,
            stop=stop,
            label_smoothing=label_smoothing,
            slice_size=slice_size,
        )

        # raise error when non-finite loss occurs (NaN, +/-inf)
        if not torch.isfinite(loss):
            raise NonFiniteLossError("Loss is non-finite.")

        # correction for loss reduction
        if self.model.loss.reduction == "mean":
            this_sub_batch_size = stop - start
            loss *= this_sub_batch_size / current_batch_size

        # backward pass
        if backward:
            loss.backward()
        current_epoch_loss = loss.item()

        self.model.post_forward_pass()
        # TODO why not call torch.cuda.empty_cache()? or call self._free_graph_and_cache()?

        return current_epoch_loss

    def _process_batch_outer(
        self,
        batch: SLCWABatch,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> FloatTensor:  # noqa: D102
        return self._process_batch_static_outer(
            model=self.model,
            loss=self.loss,
            mode=self.mode,
            batch=batch,
            start=start,
            stop=stop,
            label_smoothing=label_smoothing,
            slice_size=slice_size,
        )

    def _create_bilevel_training_data_loader(
        self,
        triples_factory: CoreTriplesFactory,
        sampler: Optional[str],
        batch_size: int,
        drop_last: bool,
        **kwargs,
    ) -> DataLoader[SLCWABatch]:  # noqa: D102
        assert "batch_sampler" not in kwargs

        lcwa_tail_dataset = triples_factory.create_lcwa_instances(target=2)
        lcwa_head_dataset = triples_factory.create_lcwa_instances(target=0)
        return (
            DataLoader(
                dataset=triples_factory.create_slcwa_instances(
                    batch_size=batch_size,
                    shuffle=kwargs.pop("shuffle", True),
                    drop_last=drop_last,
                    negative_sampler=self.negative_sampler,
                    negative_sampler_kwargs=self.negative_sampler_kwargs,
                    sampler=sampler,
                ),
                # disable automatic batching
                batch_size=None,
                batch_sampler=None,
                **kwargs,
            ),
            DataLoader(
                dataset=lcwa_head_dataset, collate_fn=lcwa_head_dataset.get_collator()
            ),
            DataLoader(
                dataset=lcwa_tail_dataset, collate_fn=lcwa_tail_dataset.get_collator()
            ),
        )

    @staticmethod
    def _process_batch_static(
        model: Model,
        loss: Loss,
        batch: SLCWABatch,
        start: Optional[int],
        stop: Optional[int],
        label_smoothing: float = 0.0,
        mode=None,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        # Slicing is not possible in sLCWA training loops
        if slice_size is not None:
            raise AttributeError("Slicing is not possible for sLCWA training loops.")

        # split batch
        positive_batch, negative_batch, positive_filter = batch

        # send to device
        positive_batch = positive_batch[start:stop].to(device=model.device)
        negative_batch = negative_batch[start:stop]
        if positive_filter is not None:
            positive_filter = positive_filter[start:stop]
            negative_batch = negative_batch[positive_filter]
            positive_filter = positive_filter.to(model.device)

        # Make it negative batch broadcastable (required for num_negs_per_pos > 1).
        negative_score_shape = negative_batch.shape[:-1]
        negative_batch = negative_batch.view(-1, 3)

        # Ensure they reside on the device (should hold already for most simple negative samplers, e.g.
        # BasicNegativeSampler, BernoulliNegativeSampler
        negative_batch = negative_batch.to(model.device)

        # Compute negative and positive scores
        positive_scores = model.score_hrt(positive_batch)

        negative_scores = model.score_hrt(negative_batch)

        negative_scores = negative_scores.view(negative_score_shape)

        return (
            loss.process_slcwa_scores(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                label_smoothing=label_smoothing,
                batch_filter=positive_filter,
            )
            + model.collect_regularization_term()
        )

    @staticmethod
    def _process_batch_static_outer(
        model: Model,
        loss: Loss,
        batch: SLCWABatch,
        start: Optional[int],
        stop: Optional[int],
        label_smoothing: float = 0.0,
        mode=None,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        # Slicing is not possible in sLCWA training loops
        if slice_size is not None:
            raise AttributeError("Slicing is not possible for sLCWA training loops.")

        # split batch
        positive_batch, negative_batch, positive_filter = batch

        # send to device
        positive_batch = positive_batch[start:stop].to(device=model.device)
        negative_batch = negative_batch[start:stop]
        if positive_filter is not None:
            positive_filter = positive_filter[start:stop]
            negative_batch = negative_batch[positive_filter]
            positive_filter = positive_filter.to(model.device)

        # Make it negative batch broadcastable (required for num_negs_per_pos > 1).
        negative_score_shape = negative_batch.shape[:-1]
        negative_batch = negative_batch.view(-1, 3)

        # Ensure they reside on the device (should hold already for most simple negative samplers, e.g.
        # BasicNegativeSampler, BernoulliNegativeSampler
        negative_batch = negative_batch.to(model.device)

        # Compute negative and positive scores
        positive_scores = model.score_hrt_with_cat(positive_batch)
        negative_scores = model.score_hrt_with_cat(negative_batch)

        return (
            loss.process_slcwa_scores(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                label_smoothing=label_smoothing,
                batch_filter=positive_filter,
            )
            + model.collect_regularization_term()
        )

    @staticmethod
    def _process_cross_view_batch_static(
        model: Model,
        batch,
        start: Optional[int],
        stop: Optional[int],
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        # Slicing is not possible in sLCWA training loops
        if slice_size is not None:
            raise AttributeError("Slicing is not possible for sLCWA training loops.")

        # split batch
        positive_batch, negative_batch, positive_filter = batch

        # send to device
        positive_batch = positive_batch[start:stop].to(device=model.device)
        negative_batch = negative_batch[start:stop]
        if positive_filter is not None:
            positive_filter = positive_filter[start:stop]
            negative_batch = negative_batch[positive_filter]
            positive_filter = positive_filter.to(model.device)

        # Make it negative batch broadcastable (required for num_negs_per_pos > 1).
        negative_batch = negative_batch.view(-1, 2)

        # Ensure they reside on the device (should hold already for most simple negative samplers, e.g.
        # BasicNegativeSampler, BernoulliNegativeSampler
        negative_batch = negative_batch.to(model.device)

        # Compute negative and positive scores
        positive_scores = model.score_cross_view(positive_batch)
        negative_scores = model.score_cross_view(negative_batch)
        negative_scores = torch.masked_fill(
            negative_scores, mask=~torch.isfinite(negative_scores), value=0.0
        )

        positive_loss = -torch.mean(logsigmoid(positive_scores))
        negative_scores = torch.mean(logsigmoid(-negative_scores))

        neg_weights = negative_scores.detach() - negative_scores.detach() ** 2

        negative_loss = -torch.mean(negative_scores * neg_weights)
        loss = 0.5 * (positive_loss + negative_loss)

        return loss + model.collect_regularization_term()

    def _changing_trainable_parameters(self, fix_weight=True):
        if fix_weight:
            self.model.ent_wrt_cat_weight.grad = None
        else:
            self.model.entity_representations[0]._embeddings.weight.grad = None
            self.model.relation_representations[0]._embeddings.weight.grad = None
            self.model.category_representations[0]._embeddings.weight.grad = None

    def _showing_grad(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                print(name, "has grad")

    def _train_epoch(
        self,
        triple_factory,
        batches: DataLoader,
        cross_view_batches: DataLoader,
        callbacks: MultiTrainingCallback,
        sub_batch_size: Optional[int],
        label_smoothing: float,
        slice_size: Optional[int],
        epoch: int,
        only_size_probing: bool,
        backward: bool = True,
    ) -> float:
        """
        Run one epoch.

        :param batches:
            the batches to process
        :param corss_view_batches:
            the cross view batches to process
        :param callbacks:
            the callbacks to apply (wrapped into a single object)
        :param sub_batch_size:
            the sub-batch size to use
        :param label_smoothing:
            the label-smoothing to apply
        :param slice_size:
            the optional slice size
        :param epoch:
            the current epoch (only used to forward to callbacks)
        :param only_size_probing:
            whether to stop after the second batch
        :param backward:
            whether to calculate gradients via backward

        :return:
            the epoch loss
        """

        # Accumulate loss over epoch
        current_epoch_loss = 0.0

        assert self.inner_percentage > 0 and self.inner_percentage < 1

        # Flag to check when to quit the size probing
        evaluated_once = False
        batches = list(batches)
        split_point = int(self.inner_percentage * len(batches))

        # print("before")
        for batch in batches[:split_point]:
            # apply callbacks before starting with batch
            callbacks.pre_batch()

            # Get batch size of current batch (last batch may be incomplete)
            current_batch_size = self._get_batch_size(batch)
            _sub_batch_size = sub_batch_size or current_batch_size

            # accumulate gradients for whole batch
            for start in range(0, current_batch_size, _sub_batch_size):
                stop = min(start + _sub_batch_size, current_batch_size)

                # forward pass call
                batch_loss = self._forward_pass(
                    batch,
                    start,
                    stop,
                    current_batch_size,
                    label_smoothing,
                    slice_size,
                    backward=backward,
                )
                current_epoch_loss += batch_loss
                callbacks.on_batch(epoch=epoch, batch=batch, batch_loss=batch_loss)

            # self._changing_trainable_parameters(fix_weight=True)
            callbacks.pre_step()
            callbacks.post_batch(epoch=epoch, batch=batch, flag="inner")
            if only_size_probing and evaluated_once:
                break
            evaluated_once = True

        for batch in cross_view_batches:
            # apply callbacks before starting with batch
            callbacks.pre_batch()

            # Get batch size of current batch (last batch may be incomplete)
            current_batch_size = self._get_batch_size(batch)
            _sub_batch_size = sub_batch_size or current_batch_size

            # accumulate gradients for whole batch
            for start in range(0, current_batch_size, _sub_batch_size):
                stop = min(start + _sub_batch_size, current_batch_size)

                # forward pass call
                batch_loss = self._cross_view_forward_pass(
                    batch,
                    start,
                    stop,
                    current_batch_size,
                    label_smoothing,
                    slice_size,
                    backward=backward,
                )
                current_epoch_loss += batch_loss
                callbacks.on_batch(epoch=epoch, batch=batch, batch_loss=batch_loss)

            callbacks.pre_step()
            callbacks.post_batch(epoch=epoch, batch=batch, flag="cross_view")

            if only_size_probing and evaluated_once:
                break
            evaluated_once = True

        for batch in batches[split_point:]:
            # apply callbacks before starting with batch
            callbacks.pre_batch()
            # Get batch size of current batch (last batch may be incomplete)
            current_batch_size = self._get_batch_size(batch)
            _sub_batch_size = sub_batch_size or current_batch_size

            # accumulate gradients for whole batch
            for start in range(0, current_batch_size, _sub_batch_size):
                stop = min(start + _sub_batch_size, current_batch_size)

                # forward pass call
                batch_loss = self._forward_pass_outer(
                    batch,
                    start,
                    stop,
                    current_batch_size,
                    label_smoothing,
                    slice_size,
                    backward=backward,
                )
                current_epoch_loss += batch_loss
                callbacks.on_batch(epoch=epoch, batch=batch, batch_loss=batch_loss)

            callbacks.pre_step()
            callbacks.post_batch(epoch=epoch, batch=batch, flag="outer")

            # For testing purposes we're only interested in processing one batch
            if only_size_probing and evaluated_once:
                break
            evaluated_once = True

        if self.store_intermediate_results:

            with torch.no_grad():
                print("storing intermediate results................")
                distance = 0
                for cat_id in list(triple_factory.categories_to_ids.values()):
                    ent_ids = triple_factory.cat2ent[cat_id]

                    cat_id = torch.LongTensor([cat_id]).to(self.model.device)
                    ent_ids = torch.LongTensor(ent_ids).to(self.model.device)

                    ent_emb = self.model.entity_representations[0](ent_ids)
                    cat_emb = self.model.category_representations[0](cat_id)

                    distance += torch.mean(
                        torch.norm(
                            ent_emb - self.model.project_cat_to_ent_space(cat_emb),
                            dim=1,
                        )
                    )

                distance = distance / len(triple_factory.categories_to_ids)

                self.result_tracker.log_metrics(
                    dict(distance=distance.item()), step=epoch
                )

                entities_variances = torch.var(
                    self.model.entity_representations[0]._embeddings.weight
                )
                relations_variances = torch.var(
                    self.model.relation_representations[0]._embeddings.weight
                )
                categories_variances = torch.var(
                    self.model.category_representations[0]._embeddings.weight
                )
                weights_variances = torch.var(self.model.ent_wrt_cat_weight)

                linear_layer_variances = torch.var(
                    self.model.project_cat_to_ent_space[0].weight
                )
                linear_layer_bias_mean = torch.mean(
                    self.model.project_cat_to_ent_space[0].bias
                )

                linear_layer_bias_variances = torch.var(
                    self.model.project_cat_to_ent_space[0].bias
                )

                self.result_tracker.log_metrics(
                    dict(entities_variances=entities_variances.item()), step=epoch
                )
                self.result_tracker.log_metrics(
                    dict(relations_variances=relations_variances.item()), step=epoch
                )
                self.result_tracker.log_metrics(
                    dict(categories_variances=categories_variances.item()), step=epoch
                )
                self.result_tracker.log_metrics(
                    dict(weights_variances=weights_variances.item()), step=epoch
                )
                self.result_tracker.log_metrics(
                    dict(linear_layer_variances=linear_layer_variances.item()),
                    step=epoch,
                )
                self.result_tracker.log_metrics(
                    dict(linear_layer_bias_mean=linear_layer_bias_mean.item()),
                    step=epoch,
                )
                self.result_tracker.log_metrics(
                    dict(
                        linear_layer_bias_variances=linear_layer_bias_variances.item()
                    ),
                    step=epoch,
                )
        # note: this epoch loss can be slightly biased towards the last batch, if this is smaller than the rest
        #        in practice, this should have a minor effect, since typically batch_size << num_instances
        current_epoch_loss = current_epoch_loss / (
            len(cross_view_batches) + len(batches)
        )
        # TODO: is this necessary?
        del batch
        del batches
        del cross_view_batches
        gc.collect()
        self.optimizer_inner.zero_grad()
        self.optimizer_outer.zero_grad()
        self.optimizer_cv.zero_grad()
        self._free_graph_and_cache()
        # return current_epoch_loss, distance
        return current_epoch_loss

    def _create_cross_view_data_loader(
        self,
        triples_factory: CategoryTriplesFactory,
        batch_size: int,
        drop_last: bool,
        num_negs_per_pos: int,
        **kwargs,
    ) -> DataLoader[SLCWABatch]:
        return DataLoader(
            dataset=triples_factory.create_cross_view_instances(
                batch_size=batch_size,
                shuffle=kwargs.pop("shuffle", True),
                drop_last=drop_last,
                num_negs_per_pos=num_negs_per_pos,
            ),
            # disable automatic batching
            batch_size=None,
            batch_sampler=None,
            **kwargs,
        )

    def _train(  # noqa: C901
        self,
        triples_factory: CoreTriplesFactory,
        num_epochs: int = 1,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        label_smoothing: float = 0.0,
        sampler: Optional[str] = None,
        continue_training: bool = False,
        only_size_probing: bool = False,
        use_tqdm: bool = True,
        use_tqdm_batch: bool = True,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
        stopper: Optional[Stopper] = None,
        sub_batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        save_checkpoints: bool = False,
        checkpoint_path: Union[None, str, pathlib.Path] = None,
        checkpoint_frequency: Optional[int] = None,
        checkpoint_on_failure_file_path: Union[None, str, pathlib.Path] = None,
        best_epoch_model_file_path: Optional[pathlib.Path] = None,
        last_best_epoch: Optional[int] = None,
        drop_last: Optional[bool] = None,
        callbacks: TrainingCallbackHint = None,
        callbacks_kwargs: TrainingCallbackKwargsHint = None,
        gradient_clipping_max_norm: Optional[float] = None,
        gradient_clipping_norm_type: Optional[float] = None,
        gradient_clipping_max_abs_value: Optional[float] = None,
        pin_memory: bool = True,
    ) -> Optional[List[float]]:
        """Train the KGE model, see docstring for :func:`TrainingLoop.train`."""
        if self.optimizer is None:
            raise ValueError("optimizer must be set before running _train()")
        # When using early stopping models have to be saved separately at the best epoch, since the training loop will
        # due to the patience continue to train after the best epoch and thus alter the model
        # -> the temporay file has to be created outside, which we assert here
        if stopper is not None and not only_size_probing and last_best_epoch is None:
            assert best_epoch_model_file_path is not None

        if isinstance(self.model, RGCN) and sampler != "schlichtkrull":
            logger.warning(
                'Using RGCN without graph-based sampling! Please select sampler="schlichtkrull" instead of %s.',
                sampler,
            )

        # Prepare all of the callbacks
        callback = MultiTrainingCallback(
            callbacks=callbacks, callbacks_kwargs=callbacks_kwargs
        )
        # Register a callback for the result tracker, if given
        if self.result_tracker is not None:
            callback.register_callback(TrackerTrainingCallback())
        # Register a callback for the early stopper, if given
        # TODO should mode be passed here?
        if stopper is not None and not self.using_LRonPlateau:
            callback.register_callback(
                StopperTrainingCallback(
                    stopper,
                    triples_factory=triples_factory,
                    last_best_epoch=last_best_epoch,
                    best_epoch_model_file_path=best_epoch_model_file_path,
                )
            )

        callback.register_training_loop(self)

        # Take the biggest possible training batch_size, if batch_size not set
        batch_size_sufficient = False
        if batch_size is None:
            if self.automatic_memory_optimization:
                # Using automatic memory optimization on CPU may result in undocumented crashes due to OS' OOM killer.
                if self.model.device.type == "cpu":
                    batch_size = 256
                    batch_size_sufficient = True
                    logger.info(
                        "Currently automatic memory optimization only supports GPUs, but you're using a CPU. "
                        "Therefore, the batch_size will be set to the default value '{batch_size}'",
                    )
                else:
                    batch_size, batch_size_sufficient = self.batch_size_search(
                        triples_factory=triples_factory
                    )
            else:
                batch_size = 256
                logger.info(
                    f"No batch_size provided. Setting batch_size to '{batch_size}'."
                )

        # This will find necessary parameters to optimize the use of the hardware at hand
        if (
            not only_size_probing
            and self.automatic_memory_optimization
            and not batch_size_sufficient
            and not continue_training
        ):
            # return the relevant parameters slice_size and batch_size
            sub_batch_size, slice_size = self.sub_batch_and_slice(
                batch_size=batch_size, sampler=sampler, triples_factory=triples_factory
            )

        if (
            sub_batch_size is None or sub_batch_size == batch_size
        ):  # by default do not split batches in sub-batches
            sub_batch_size = batch_size
        elif get_batchnorm_modules(self.model):  # if there are any, this is truthy
            raise SubBatchingNotSupportedError(self.model)

        model_contains_batch_norm = bool(get_batchnorm_modules(self.model))
        if batch_size == 1 and model_contains_batch_norm:
            raise ValueError(
                "Cannot train a model with batch_size=1 containing BatchNorm layers."
            )

        if drop_last is None:
            drop_last = model_contains_batch_norm

        # Force weight initialization if training continuation is not explicitly requested.
        if not continue_training:
            # Reset the weights
            self.model.reset_parameters_()
            # afterwards, some parameters may be on the wrong device
            self.model.to(get_preferred_device(self.model, allow_ambiguity=True))

            # Create new optimizer
            optimizer_kwargs = _get_optimizer_kwargs(self.optimizer)
            self.optimizer = self.optimizer.__class__(
                params=self.model.get_grad_params(),
                **optimizer_kwargs,
            )

            if self.lr_scheduler is not None:
                # Create a new lr scheduler and add the optimizer
                lr_scheduler_kwargs = _get_lr_scheduler_kwargs(self.lr_scheduler)
                self.lr_scheduler = self.lr_scheduler.__class__(
                    optimizer=self.optimizer, **lr_scheduler_kwargs
                )
        elif not self.optimizer.state:
            raise ValueError("Cannot continue_training without being trained once.")
        if stopper is not None and self.using_LRonPlateau:

            inner_lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer_inner,
                factor=0.1,
                mode="max",
                patience=max(0, stopper.patience // 2 - 1),
                threshold=stopper.relative_delta,
                threshold_mode="abs",
            )

            outer_lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer_outer,
                factor=0.1,
                mode="max",
                patience=max(0, stopper.patience // 2 - 1),
                threshold=stopper.relative_delta,
                threshold_mode="abs",
            )

            cv_lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer_cv,
                factor=0.1,
                mode="max",
                patience=max(0, stopper.patience // 2 - 1),
                threshold=stopper.relative_delta,
                threshold_mode="abs",
            )

            callback.register_callback(
                EarlyStoppingWithLROnPlateuaCallback(
                    stopper,
                    [inner_lr_plateau, outer_lr_plateau, cv_lr_plateau],
                    triples_factory=triples_factory,
                    last_best_epoch=last_best_epoch,
                    best_epoch_model_file_path=best_epoch_model_file_path,
                )
            )
        # Ensure the model is on the correct device
        self.model.to(get_preferred_device(self.model, allow_ambiguity=True))

        if num_workers is None:
            num_workers = 0

        _use_outer_tqdm = not only_size_probing and use_tqdm
        _use_inner_tqdm = _use_outer_tqdm and use_tqdm_batch

        # When size probing, we don't want progress bars
        if _use_outer_tqdm:
            # Create progress bar
            _tqdm_kwargs = dict(desc=f"Training epochs on {self.device}", unit="epoch")
            if tqdm_kwargs is not None:
                _tqdm_kwargs.update(tqdm_kwargs)
            epochs = trange(
                self._epoch + 1,
                1 + num_epochs,
                **_tqdm_kwargs,
                initial=self._epoch,
                total=num_epochs,
            )
        elif only_size_probing:
            epochs = range(1, 1 + num_epochs)
        else:
            epochs = range(self._epoch + 1, 1 + num_epochs)

        logger.debug(f"using stopper: {stopper}")

        train_data_loader = self._create_training_data_loader(
            triples_factory,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,  # always shuffle during training
            sampler=sampler,
        )
        cross_view_data_loader = self._create_cross_view_data_loader(
            triples_factory,
            batch_size=batch_size,
            drop_last=drop_last,
            num_negs_per_pos=self.num_negs_cross_view,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,  # always shuffle during training
        )
        if len(train_data_loader) == 0:
            raise NoTrainingBatchError()
        if drop_last and not only_size_probing:
            logger.info(
                "Dropping last (incomplete) batch each epoch (%s batches).",
                format_relative_comparison(part=1, total=len(train_data_loader)),
            )

        # optimizer callbacks
        pre_step_callbacks: List[TrainingCallback] = []
        if gradient_clipping_max_norm is not None:
            pre_step_callbacks.append(
                GradientNormClippingTrainingCallback(
                    max_norm=gradient_clipping_max_norm,
                    norm_type=gradient_clipping_norm_type,
                )
            )
        if gradient_clipping_max_abs_value is not None:
            pre_step_callbacks.append(
                GradientAbsClippingTrainingCallback(
                    clip_value=gradient_clipping_max_abs_value
                )
            )
        callback.register_callback(
            InnerOptimizerTrainingCallback(
                only_size_probing=only_size_probing,
                pre_step_callbacks=pre_step_callbacks,
            )
        )
        callback.register_callback(
            OuterOptimizerTrainingCallback(
                only_size_probing=only_size_probing,
                pre_step_callbacks=pre_step_callbacks,
            )
        )
        callback.register_callback(
            CVOptimizerTrainingCallback(
                only_size_probing=only_size_probing,
                pre_step_callbacks=pre_step_callbacks,
            )
        )
        if self.lr_scheduler is not None:
            callback.register_callback(LearningRateSchedulerTrainingCallback())

        # Save the time to track when the saved point was available
        last_checkpoint = time.time()
        best_epoch_model_checkpoint_file_path: Optional[pathlib.Path] = None

        # Training Loop
        for epoch in epochs:
            # When training with an early stopper the memory pressure changes, which may allow for errors each epoch
            try:
                # Enforce training mode
                self.model.train()

                # Batching
                # Only create a progress bar when not in size probing mode
                if _use_inner_tqdm:
                    batches = tqdm(
                        train_data_loader,
                        desc=f"Training batches on {self.device}",
                        leave=False,
                        unit="batch",
                    )
                else:
                    batches = train_data_loader

                # epoch_loss, distance = self._train_epoch(
                epoch_loss = self._train_epoch(
                    triple_factory=triples_factory,
                    batches=batches,
                    cross_view_batches=cross_view_data_loader,
                    callbacks=callback,
                    sub_batch_size=sub_batch_size,
                    label_smoothing=label_smoothing,
                    slice_size=slice_size,
                    epoch=epoch,
                    only_size_probing=only_size_probing,
                )

                # When size probing we don't need the losses
                if only_size_probing:
                    return None

                # Track epoch loss
                self.losses_per_epochs.append(epoch_loss)

                # Print loss information to console
                if _use_outer_tqdm:
                    epochs.set_postfix(
                        {
                            "loss": self.losses_per_epochs[-1],
                            "prev_loss": (
                                self.losses_per_epochs[-2]
                                if epoch > 1
                                else float("nan")
                            ),
                        }
                    )

                # Save the last successful finished epoch
                self._epoch = epoch

            # When the training loop failed, a fallback checkpoint is created to resume training.
            except (MemoryError, RuntimeError) as e:
                # During automatic memory optimization only the error message is of interest
                if only_size_probing:
                    raise e

                logger.warning(
                    f"The training loop just failed during epoch {epoch} due to error {str(e)}."
                )
                if checkpoint_on_failure_file_path:
                    # When there wasn't a best epoch the checkpoint path should be None
                    if (
                        last_best_epoch is not None
                        and best_epoch_model_file_path is not None
                    ):
                        best_epoch_model_checkpoint_file_path = (
                            best_epoch_model_file_path
                        )
                    self._save_state(
                        path=checkpoint_on_failure_file_path,
                        stopper=stopper,
                        best_epoch_model_checkpoint_file_path=best_epoch_model_checkpoint_file_path,
                        triples_factory=triples_factory,
                    )
                    logger.warning(
                        "However, don't worry we got you covered. PyKEEN just saved a checkpoint when this "
                        f"happened at '{checkpoint_on_failure_file_path}'. To resume training from the checkpoint "
                        f"file just restart your code and pass this file path to the training loop or pipeline you "
                        f"used as 'checkpoint_file' argument.",
                    )
                # Delete temporary best epoch model
                if (
                    best_epoch_model_file_path is not None
                    and best_epoch_model_file_path.is_file()
                ):
                    os.remove(best_epoch_model_file_path)
                raise e

            # Includes a call to result_tracker.log_metrics
            callback.post_epoch(epoch=epoch, epoch_loss=epoch_loss)

            # If a checkpoint file is given, we check whether it is time to save a checkpoint
            if save_checkpoints and checkpoint_path is not None:
                minutes_since_last_checkpoint = (time.time() - last_checkpoint) // 60
                # MyPy overrides are because you should
                if (
                    minutes_since_last_checkpoint >= checkpoint_frequency  # type: ignore
                    or self._should_stop
                    or epoch == num_epochs
                ):
                    # When there wasn't a best epoch the checkpoint path should be None
                    if (
                        last_best_epoch is not None
                        and best_epoch_model_file_path is not None
                    ):
                        best_epoch_model_checkpoint_file_path = (
                            best_epoch_model_file_path
                        )
                    self._save_state(
                        path=checkpoint_path,
                        stopper=stopper,
                        best_epoch_model_checkpoint_file_path=best_epoch_model_checkpoint_file_path,
                        triples_factory=triples_factory,
                    )  # type: ignore
                    last_checkpoint = time.time()

            if self._should_stop:
                if (
                    last_best_epoch is not None
                    and best_epoch_model_file_path is not None
                ):
                    self._load_state(path=best_epoch_model_file_path)
                    # Delete temporary best epoch model
                    if pathlib.Path.is_file(best_epoch_model_file_path):
                        os.remove(best_epoch_model_file_path)
                return self.losses_per_epochs

        callback.post_train(losses=self.losses_per_epochs)

        # If the stopper didn't stop the training loop but derived a best epoch, the model has to be reconstructed
        # at that state
        if (
            stopper is not None
            and last_best_epoch is not None
            and best_epoch_model_file_path is not None
        ):
            self._load_state(path=best_epoch_model_file_path)
            # Delete temporary best epoch model
            if pathlib.Path.is_file(best_epoch_model_file_path):
                os.remove(best_epoch_model_file_path)

        return self.losses_per_epochs
