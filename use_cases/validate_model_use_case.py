import tensorflow as tf
from utils.timing import TimingLogger
from validation.validate import get_images, validate_model_on_lfw

AUTOTUNE = tf.data.experimental.AUTOTUNE


class ValidateModelUseCase:
    def __init__(self, strategy, summary_writer, timing: TimingLogger, logger):
        self.strategy = strategy
        self.timing = timing
        self.summary_writer = summary_writer
        self.logger = logger

    def execute(self, model, dataset, BATCH_SIZE: int, checkpoint):
        left_pairs, right_pairs, is_same_list = dataset

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
        lr_images, sr_images = get_images(self.strategy, model, left_pairs)
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
            lr_images,
            sr_images,
        )
        return accuracy_mean

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
        lr_images,
        sr_images,
    ):
        with self.summary_writer.as_default():
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
            tf.summary.image(
                f"LR Images",
                tf.concat(lr_images.values, axis=0),
                max_outputs=32,
                step=checkpoint.step,
            )
            tf.summary.image(
                f"SR Images",
                tf.concat(sr_images.values, axis=0),
                max_outputs=32,
                step=checkpoint.step,
            )

        self.logger.info(
            (
                f" Validation on LFW: Step {int(checkpoint.step.values[0])} -"
                f" Accuracy: {accuracy_mean:.3f} +- {accuracy_std:.3f} -"
                f" Validation Rate: {validation_rate:.3f} +-"
                f" {validation_std:.3f} @ FAR {far:.3f} -"
                f" Area Under Curve (AUC): {auc:.3f} -"
                f" Equal Error Rate (EER): {eer:.3f} -"
                f" Elapsed Time: {elapsed_time}."
            )
        )
