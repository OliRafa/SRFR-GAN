from pathlib import Path

import tensorflow as tf

from utils.input_data import LFW
from utils.timing import TimingLogger
from validation.validate import validate_model_on_lfw


class ValidateModelUseCase:
    def __init__(self, strategy, test_summary_writer, timing: TimingLogger, logger):
        self.strategy = strategy
        self.timing = timing
        self.test_summary_writer = test_summary_writer
        self.logger = logger
        self.CACHE_PATH = Path.cwd().joinpath("data", "temp", "lfw")
        if not self.CACHE_PATH.is_dir():
            self.CACHE_PATH.mkdir(parents=True)

    def execute(self, model, BATCH_SIZE: int, checkpoint):
        left_pairs, right_pairs, is_same_list = self._instantiate_dataset(BATCH_SIZE)

        self.timing.start(validate_model_on_lfw.__name__)
        (
            accuracy_mean,
            accuracy_std,
            validation_rate,
            validation_std,
            far,
            auc,
            eer,
        ) = validate_model_on_lfw(
            self.strategy,
            model,
            left_pairs,
            right_pairs,
            is_same_list,
        )
        elapsed_time = self.timing.end(validate_model_on_lfw.__name__, True)
        self._save_validation_data(
            checkpoint,
            accuracy_mean,
            accuracy_std,
            validation_rate,
            validation_std,
            far,
            auc,
            eer,
            elapsed_time,
        )

    def _instantiate_dataset(self, BATCH_SIZE: int):
        lfw = LFW()
        left_pairs, right_pairs, is_same_list = lfw.get_dataset()
        left_pairs = left_pairs.batch(BATCH_SIZE).cache(
            str(self.CACHE_PATH.joinpath("left"))
        )
        left_pairs = self.strategy.experimental_distribute_dataset(left_pairs)

        right_pairs = right_pairs.batch(BATCH_SIZE).cache(
            str(self.CACHE_PATH.joinpath("right"))
        )
        right_pairs = self.strategy.experimental_distribute_dataset(right_pairs)

        return left_pairs, right_pairs, is_same_list

    def _save_validation_data(
        self,
        checkpoint,
        accuracy_mean,
        accuracy_std,
        validation_rate,
        validation_std,
        far,
        auc,
        eer,
        elapsed_time,
    ):
        with self.test_summary_writer.as_default():
            tf.summary.scalar(
                "accuracy_mean",
                accuracy_mean,
                step=checkpoint.step,
            )
            tf.summary.scalar("accuracy_std", accuracy_std, step=checkpoint.step)
            tf.summary.scalar("validation_rate", validation_rate, step=checkpoint.step)
            tf.summary.scalar("validation_std", validation_std, step=checkpoint.step)
            tf.summary.scalar("far", far, step=checkpoint.step)
            tf.summary.scalar("auc", auc, step=checkpoint.step)
            tf.summary.scalar("eer", eer, step=checkpoint.step)

        self.logger.info(
            (
                f" Validation on LFW: Step {int(checkpoint.step)} -"
                f" Accuracy: {accuracy_mean:.3f} +- {accuracy_std:.3f} -"
                f" Validation Rate: {validation_rate:.3f} +-"
                f" {validation_std:.3f} @ FAR {far:.3f} -"
                f" Area Under Curve (AUC): {auc:.3f} -"
                f" Equal Error Rate (EER): {eer:.3f} -"
                f" Elapsed Time: {elapsed_time}."
            )
        )
