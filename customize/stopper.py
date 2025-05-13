import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, List, Mapping

import torch
from pykeen.stoppers.early_stopping import EarlyStopper

logger = logging.getLogger(__name__)


@dataclass
class PostponeEarlyStopper(EarlyStopper):

    start_epoch: int = 0

    def should_evaluate(self, epoch: int) -> bool:
        """Decide if evaluation should be done based on the current epoch and the internal frequency."""
        return epoch > self.start_epoch and epoch % self.frequency == 0


@dataclass
class EarlyStopperWithTrainingResults(EarlyStopper):

    training_results: List[float] = dataclasses.field(default_factory=list, repr=False)

    def __post_init__(self):
        super().__post_init__()

    def should_stop(self, epoch: int) -> bool:
        """Evaluate on a metric and compare to past evaluations to decide if training should stop."""
        # for mypy
        assert self.best_model_path is not None
        # Evaluate
        metric_results = self.evaluator.evaluate(
            model=self.model,
            additional_filter_triples=self.training_triples_factory.mapped_triples,
            mapped_triples=self.evaluation_triples_factory.mapped_triples,
            use_tqdm=self.use_tqdm,
            tqdm_kwargs=self.tqdm_kwargs,
            batch_size=self.evaluation_batch_size,
            slice_size=self.evaluation_slice_size,
            # Only perform time-consuming checks for the first call.
            do_time_consuming_checks=self.evaluation_batch_size is None,
        )
        # After the first evaluation pass the optimal batch and slice size is obtained and saved for re-use
        self.evaluation_batch_size = self.evaluator.batch_size
        self.evaluation_slice_size = self.evaluator.slice_size

        training_results = self.evaluator.evaluate(
            model=self.model,
            mapped_triples=self.training_triples_factory.mapped_triples,
            use_tqdm=self.use_tqdm,
            tqdm_kwargs=self.tqdm_kwargs,
            batch_size=self.evaluation_batch_size,
            slice_size=self.evaluation_slice_size,
            # Only perform time-consuming checks for the first call.
            do_time_consuming_checks=self.evaluation_batch_size is None,
        )

        if self.result_tracker is not None:
            self.result_tracker.log_metrics(
                metrics=metric_results.to_flat_dict(),
                step=epoch,
                prefix="validation",
            )
        result = metric_results.get_metric(self.metric)
        train_result = training_results.get_metric(self.metric)

        # Append to history
        self.results.append(result)
        self.training_results.append(train_result)
        for result_callback in self.result_callbacks:
            result_callback(self, result, epoch)

        self.stopped = self._stopper.report_result(metric=result, epoch=epoch)
        if self.stopped:
            logger.info(
                f"Stopping early at epoch {epoch}. The best result {self.best_metric} occurred at "
                f"epoch {self.best_epoch}.",
            )
            for stopped_callback in self.stopped_callbacks:
                stopped_callback(self, result, epoch)
            logger.info(
                f"Re-loading weights from best epoch from {self.best_model_path}"
            )
            self.model.load_state_dict(torch.load(self.best_model_path))
            if self.clean_up_checkpoint:
                self.best_model_path.unlink()
                logger.debug(
                    f"Clean up checkpoint with best weights: {self.best_model_path}"
                )
            return True

        if self._stopper.is_best:
            torch.save(self.model.state_dict(), self.best_model_path)
            logger.info(
                f"New best result at epoch {epoch}: {self.best_metric}. Saved model weights to {self.best_model_path}",
            )

        for continue_callback in self.continue_callbacks:
            continue_callback(self, result, epoch)
        return False

    def get_summary_dict(self) -> Mapping[str, Any]:
        """Get a summary dict."""
        return dict(
            frequency=self.frequency,
            patience=self.patience,
            remaining_patience=self.remaining_patience,
            relative_delta=self.relative_delta,
            metric=self.metric,
            larger_is_better=self.larger_is_better,
            results=self.results,
            training_results=self.training_results,
            stopped=self.stopped,
            best_epoch=self.best_epoch,
            best_metric=self.best_metric,
        )
