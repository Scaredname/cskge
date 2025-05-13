import logging
import pathlib
from typing import Any, List, Optional

from pykeen.stoppers import Stopper
from pykeen.training.callbacks import OptimizerTrainingCallback, StopperTrainingCallback
from pykeen.triples import CoreTriplesFactory
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger(__name__)


class EarlyStoppingWithLROnPlateuaCallback(StopperTrainingCallback):
    def __init__(
        self,
        stopper: Stopper,
        lr_scheduler: List[ReduceLROnPlateau],
        *,
        triples_factory: CoreTriplesFactory,
        last_best_epoch: Optional[int] = None,
        best_epoch_model_file_path: Optional[pathlib.Path]
    ):
        super().__init__(
            stopper,
            triples_factory=triples_factory,
            last_best_epoch=last_best_epoch,
            best_epoch_model_file_path=best_epoch_model_file_path,
        )
        assert isinstance(lr_scheduler, list)
        self.lr_scheduler = lr_scheduler

    def post_epoch(
        self, epoch: int, epoch_loss: float, **kwargs: Any
    ) -> None:  # noqa: D102
        if self.stopper.should_evaluate(epoch):
            # TODO how to pass inductive mode

            if self.stopper.should_stop(epoch):
                self.training_loop._should_stop = True
            # Since the model is also used within the stopper, its graph and cache have to be cleared
            self.model._free_graph_and_cache()
            for scheduler in self.lr_scheduler:
                scheduler.step(self.stopper.best_metric)

            # When the stopper obtained a new best epoch, this model has to be saved for reconstruction
        if (
            self.stopper.best_epoch != self.last_best_epoch
            and self.best_epoch_model_file_path is not None
        ):
            self.training_loop._save_state(
                path=self.best_epoch_model_file_path,
                triples_factory=self.triples_factory,
            )
            self.last_best_epoch = epoch


class InnerOptimizerTrainingCallback(OptimizerTrainingCallback):
    def __init__(
        self,
        only_size_probing=False,
        pre_step_callbacks=None,
    ):
        super().__init__(only_size_probing, pre_step_callbacks)

    @property
    def optimizer_inner(self) -> optim.Optimizer:  # noqa:D401
        """The optimizer, accessed via the training loop."""
        return self.training_loop.optimizer_inner

    # docstr-coverage: inherited
    def pre_batch(self, **kwargs: Any) -> None:  # noqa: D102
        # Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old instance

        # note: we want to run this step during size probing to cleanup any remaining grads
        self.optimizer_inner.zero_grad(set_to_none=True)

    # docstr-coverage: inherited
    def post_batch(self, epoch: int, batch, **kwargs: Any) -> None:

        try:
            if kwargs["flag"] == "inner":
                # pre-step callbacks
                for cb in self.pre_step_callbacks:
                    cb.pre_step(epoch=epoch, **kwargs)

                if not self.only_size_probing:
                    # update parameters according to optimizer
                    self.optimizer_inner.step()

                self.model.post_parameter_update()
        except:
            raise KeyError("flag not found in post_batch kwargs")


class OuterOptimizerTrainingCallback(OptimizerTrainingCallback):
    def __init__(
        self,
        only_size_probing=False,
        pre_step_callbacks=None,
    ):
        super().__init__(only_size_probing, pre_step_callbacks)

    @property
    def optimizer_outer(self) -> optim.Optimizer:  # noqa:D401
        """The optimizer, accessed via the training loop."""
        return self.training_loop.optimizer_outer

    # docstr-coverage: inherited
    def pre_batch(self, **kwargs: Any) -> None:  # noqa: D102
        # Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old instance

        # note: we want to run this step during size probing to cleanup any remaining grads
        self.optimizer_outer.zero_grad(set_to_none=True)

    # docstr-coverage: inherited
    def post_batch(self, epoch: int, batch, **kwargs: Any) -> None:  # noqa: D102
        try:
            if kwargs["flag"] == "outer":
                # pre-step callbacks
                for cb in self.pre_step_callbacks:
                    cb.pre_step(epoch=epoch, **kwargs)

                if not self.only_size_probing:
                    # update parameters according to optimizer
                    self.optimizer_outer.step()

                self.model.post_parameter_update()
        except:
            raise KeyError("flag not found in post_batch kwargs")


class CVOptimizerTrainingCallback(OptimizerTrainingCallback):
    def __init__(
        self,
        only_size_probing=False,
        pre_step_callbacks=None,
    ):
        super().__init__(only_size_probing, pre_step_callbacks)

    @property
    def optimizer_cv(self) -> optim.Optimizer:  # noqa:D401
        """The optimizer, accessed via the training loop."""
        return self.training_loop.optimizer_cv

    # docstr-coverage: inherited
    def pre_batch(self, **kwargs: Any) -> None:  # noqa: D102
        # Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old instance

        # note: we want to run this step during size probing to cleanup any remaining grads
        self.optimizer_cv.zero_grad(set_to_none=True)

    # docstr-coverage: inherited
    def post_batch(self, epoch: int, batch, **kwargs: Any) -> None:  # noqa: D102
        try:
            if kwargs["flag"] == "cross_view":
                # pre-step callbacks
                for cb in self.pre_step_callbacks:
                    cb.pre_step(epoch=epoch, **kwargs)

                if not self.only_size_probing:
                    # update parameters according to optimizer
                    self.optimizer_cv.step()

                self.model.post_parameter_update()
        except:
            raise KeyError("flag not found in post_batch kwargs")
