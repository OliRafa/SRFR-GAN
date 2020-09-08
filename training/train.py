"""This module contains functions used for training."""
import logging

import tensorflow as tf
from training.losses import Loss
from validation.validate import validate_model_on_lfw
from utils.timing import TimingLogger


LOGGER = logging.getLogger(__name__)


def generate_num_epochs(iterations, len_dataset, batch_size):
    LOGGER.info(
        f" Generating number of epochs for {iterations} iterations,\
 {len_dataset} dataset length and {batch_size} batch size."
    )
    train_size = tf.math.ceil(len_dataset / batch_size)
    epochs = tf.cast(tf.math.ceil(iterations / train_size), dtype=tf.int32)
    LOGGER.info(f" Number of epochs: {epochs}.")
    return epochs


@tf.function
def adjust_learning_rate(current_learning_rate: float, epoch: int = 1) -> float:
    """Adjusts learning rate based on the current value and a giving epoch.

    ### Parameters
        current_learning_rate: Current value for the learning rate.
        epoch: Epoch number.

    ### Returns
        New value for the learning rate.
    """
    if epoch % 4 == 0:
        return current_learning_rate / 10
    return current_learning_rate


class Train:
    def __init__(
        self,
        strategy,
        srfr_model,
        srfr_optimizer,
        discriminator_model,
        discriminator_optimizer,
        train_summary_writer,
        test_summary_writer,
        checkpoint,
        manager,
    ):
        self.strategy = strategy
        self.srfr_model = srfr_model
        self.srfr_optimizer = srfr_optimizer
        self.discriminator_model = discriminator_model
        self.discriminator_optimizer = discriminator_optimizer
        self.train_summary_writer = train_summary_writer
        self.test_summary_writer = test_summary_writer
        self.checkpoint = checkpoint
        self.manager = manager

        self.timing = TimingLogger()
        self.losses: Loss = None

    def train_srfr_model(
        self,
        batch_size,
        train_loss_function,
        synthetic_dataset,
        num_classes_synthetic: int,
        left_pairs,
        left_aug_pairs,
        right_pairs,
        right_aug_pairs,
        is_same_list,
        sr_weight: float = 0.1,
        scale: float = 64,
        margin: float = 0.5,
        natural_dataset=None,
        num_classes_natural: int = None,
    ) -> float:
        """Train the model using the given dataset, compute the loss_function
        and apply the optimizer.

        Parameters
        ----------
            srfr_model: The Super Resolution Face Recognition model.
            sr_discriminator_model: The Discriminator model.
            batch_size: The Batch size.
            srfr_optimizer: Optimizer for the SRFR network.
            discriminator_optimizer: Optimizer for the Discriminator network.
            train_loss_function:
            sr_weight: Weight for the SR Loss.
            scale:
            margin:
            synthetic_dataset:
            num_classes_synthetic:
            natural_dataset:
            num_classes_natural:

        Returns
        -------
            (srfr_loss, discriminator_loss) The loss value for SRFR and
            Discriminator networks.
        """
        batch_size = tf.constant(batch_size, dtype=tf.float32)
        num_classes_synthetic = tf.constant(num_classes_synthetic, dtype=tf.int32)
        sr_weight = tf.constant(sr_weight, dtype=tf.float32)
        scale = tf.constant(scale, dtype=tf.float32)
        margin = tf.constant(margin, dtype=tf.float32)
        self.losses = Loss(
            self.srfr_model,
            batch_size,
            self.train_summary_writer,
            sr_weight,
            scale,
            margin,
        )
        # if natural_dataset:
        #    return self._train_with_natural_images(
        #        batch_size,
        #        train_loss_function,
        #        synthetic_dataset,
        #        num_classes_synthetic,
        #        natural_dataset,
        #        num_classes_natural,
        #        sr_weight,
        #        scale,
        #        margin
        #    )

        return self._train_with_synthetic_images_only(
            batch_size,
            train_loss_function,
            synthetic_dataset,
            num_classes_synthetic,
            left_pairs,
            left_aug_pairs,
            right_pairs,
            right_aug_pairs,
            is_same_list,
        )

    def _train_with_synthetic_images_only(
        self,
        batch_size,
        train_loss_function,
        dataset,
        num_classes: int,
        left_pairs,
        left_aug_pairs,
        right_pairs,
        right_aug_pairs,
        is_same_list,
    ) -> float:
        srfr_losses = []
        discriminator_losses = []
        with self.strategy.scope():
            for step, (
                synthetic_images,
                groud_truth_images,
                synthetic_classes,
            ) in enumerate(dataset, start=1):
                (
                    srfr_loss,
                    discriminator_loss,
                    super_resolution_images,
                ) = self._train_step_synthetic_only(
                    synthetic_images, groud_truth_images, synthetic_classes, num_classes
                )
                srfr_losses.append(srfr_loss)
                discriminator_losses.append(discriminator_loss)

                if step % 1000 == 0:
                    self.save_model()

                self._save_metrics(
                    step,
                    srfr_loss,
                    discriminator_loss,
                    batch_size,
                    synthetic_images,
                    groud_truth_images,
                    super_resolution_images,
                )
                self.checkpoint.step.assign_add(1)

                if step % 5000 == 0:
                    self._validate_on_lfw(
                        left_pairs,
                        left_aug_pairs,
                        right_pairs,
                        right_aug_pairs,
                        is_same_list,
                        batch_size,
                    )

        return (
            train_loss_function(srfr_losses),
            train_loss_function(discriminator_losses),
        )

    def _save_metrics(
        self,
        step,
        srfr_loss,
        discriminator_loss,
        batch_size,
        synthetic_images,
        groud_truth_images,
        super_resolution_images,
    ) -> None:
        LOGGER.info(
            (
                f" SRFR Training loss (for one batch) at step {step}:"
                f" {float(srfr_loss):.3f}"
            )
        )
        LOGGER.info(
            (
                f" Discriminator loss (for one batch) at step {step}:"
                f" {float(discriminator_loss):.3f}"
            )
        )
        LOGGER.info(f" Seen so far: {step * batch_size} samples")

        step = int(self.checkpoint.step)
        with self.train_summary_writer.as_default():
            tf.summary.scalar(
                f"srfr_loss_per_step",
                float(srfr_loss),
                step=step,
            )
            tf.summary.scalar(
                f"discriminator_loss_per_step",
                float(discriminator_loss),
                step=step,
            )
            tf.summary.image(
                f"lr_images_per_step",
                tf.concat(synthetic_images.values, axis=0),
                max_outputs=10,
                step=step,
            )
            tf.summary.image(
                f"hr_images_per_step",
                tf.concat(groud_truth_images.values, axis=0),
                max_outputs=10,
                step=step,
            )
            tf.summary.image(
                f"sr_images_per_step",
                tf.concat(super_resolution_images.values, axis=0),
                max_outputs=10,
                step=step,
            )

    def save_model(self):
        save_path = self.manager.save()
        LOGGER.info(
            (
                f" Saved checkpoint for epoch {int(self.checkpoint.step)}:"
                f" {save_path}"
            )
        )

    def _validate_on_lfw(
        self, left_pairs, left_aug_pairs, right_pairs, right_aug_pairs, is_same_list
    ):
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
            self.srfr_model,
            left_pairs,
            left_aug_pairs,
            right_pairs,
            right_aug_pairs,
            is_same_list,
        )
        elapsed_time = self.timing.end(validate_model_on_lfw.__name__, True)
        with self.test_summary_writer.as_default():
            tf.summary.scalar(
                "accuracy_mean",
                accuracy_mean,
                step=int(self.checkpoint.step),
            )
            tf.summary.scalar(
                "accuracy_std", accuracy_std, step=int(self.checkpoint.step)
            )
            tf.summary.scalar(
                "validation_rate", validation_rate, step=int(self.checkpoint.step)
            )
            tf.summary.scalar(
                "validation_std", validation_std, step=int(self.checkpoint.step)
            )
            tf.summary.scalar("far", far, step=int(self.checkpoint.step))
            tf.summary.scalar("auc", auc, step=int(self.checkpoint.step))
            tf.summary.scalar("eer", eer, step=int(self.checkpoint.step))
            tf.summary.scalar(
                "testing_time", elapsed_time, step=int(self.checkpoint.step)
            )

        LOGGER.info(
            (
                f" Validation on LFW: Step {int(self.checkpoint.step)} -"
                f" Accuracy: {accuracy_mean:.3f} +- {accuracy_std:.3f} -"
                f" Validation Rate: {validation_rate:.3f} +-"
                f" {validation_std:.3f} @ FAR {far:.3f} -"
                f" Area Under Curve (AUC): {auc:.3f} -"
                f" Equal Error Rate (EER): {eer:.3f} -"
            )
        )

    #@tf.function
    def _step_function(self, low_resolution_batch, groud_truth_batch,
                       ground_truth_classes, num_classes):
        with tf.GradientTape() as srfr_tape, \
                tf.GradientTape() as discriminator_tape:
            (super_resolution_images, embeddings) = self.srfr_model(
                low_resolution_batch)
            discriminator_sr_predictions = self.discriminator_model(
                super_resolution_images)
            discriminator_gt_predictions = self.discriminator_model(
                groud_truth_batch)
            synthetic_face_recognition = (embeddings, ground_truth_classes,
                                          num_classes)
            srfr_loss = self.losses.compute_joint_loss(
                super_resolution_images,
                groud_truth_batch,
                discriminator_sr_predictions,
                discriminator_gt_predictions,
                synthetic_face_recognition,
                self.checkpoint,
            )
            discriminator_loss = self.losses.compute_discriminator_loss(
                discriminator_sr_predictions,
                discriminator_gt_predictions,
            )
            srfr_loss = srfr_loss / self.strategy.num_replicas_in_sync
            discriminator_loss = (discriminator_loss /
                                  self.strategy.num_replicas_in_sync)
            srfr_scaled_loss = self.srfr_optimizer.get_scaled_loss(srfr_loss)
            discriminator_scaled_loss = self.discriminator_optimizer.\
                get_scaled_loss(discriminator_loss)

        srfr_grads = srfr_tape.gradient(srfr_scaled_loss,
                                        self.srfr_model.trainable_weights)
        discriminator_grads = discriminator_tape.gradient(
            discriminator_scaled_loss,
            self.discriminator_model.trainable_weights,
        )
        self.srfr_optimizer.apply_gradients(
            zip(self.srfr_optimizer.get_unscaled_gradients(srfr_grads),
                self.srfr_model.trainable_weights)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(self.discriminator_optimizer.get_unscaled_gradients(
                discriminator_grads),
                self.discriminator_model.trainable_weights)
        )
        return srfr_loss, discriminator_loss, super_resolution_images

    # @tf.function
    def _train_step_synthetic_only(
        self,
        synthetic_images,
        groud_truth_images,
        synthetic_classes,
        num_classes,
    ):
        """Does a training step

        Parameters
        ----------
            model:
            images: Batch of images for training.
            classes: Batch of classes to compute the loss.
            num_classes: Total number of classes in the dataset.
            s:
            margin:

        Returns
        -------
            (srfr_loss, srfr_grads, discriminator_loss, discriminator_grads)
            The loss value and the gradients for SRFR network, as well as the
            loss value and the gradients for the Discriminative network.
        """
        srfr_loss, discriminator_loss, super_resolution_images = \
            self.strategy.experimental_run_v2(
                self._step_function,
                args=(
                    synthetic_images,
                    groud_truth_images,
                    synthetic_classes,
                    num_classes,
                ),
            )
        srfr_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                         srfr_loss, None)
        discriminator_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN,
            discriminator_loss,
            None,
        )
        return srfr_loss, discriminator_loss, super_resolution_images


# @tf.function
# def _train_step(
#        model,
#        images,
#        classes,
#        num_classes,
#        scale: float,
#        margin: float,
#    ):
#    """Does a training step
#
#    ### Parameters:
#        model:
#        images: Batch of images for training.
#        classes: Batch of classes to compute the loss.
#        num_classes: Total number of classes in the dataset.
#        s:
#        margin:
#
#    ### Returns:
#        (loss_value, grads) The loss value and the gradients to be optimized.
#    """
#    with tf.GradientTape() as tape:
#        embeddings, fc_weights = model(images)
#        loss_value = compute_arcloss(
#            embeddings,
#            classes,
#            fc_weights,
#            num_classes,
#            scale,
#            margin
#        )
#    grads = tape.gradient(loss_value, model.trainable_weights)
#    return loss_value, grads
#
# @tf.function
# def train_model(
#        model,
#        dataset,
#        num_classes,
#        batch_size,
#        optimizer,
#        train_loss_function,
#        scale: float,
#        margin: float,
#    ) -> float:
#    """Train the model using the given dataset, compute the loss_function and\
# apply the optimizer.
#
#    ### Parameters:
#        model:
#        dataset:
#        num_classes:
#        batch_size:
#        optimizer:
#        train_loss_function:
#        scale:
#        margin:
#
#    ### Returns:
#        The loss value.
#    """
#    for step, (image_batch, class_batch) in enumerate(dataset):
#        loss_value, grads = _train_step(
#            model,
#            image_batch,
#            class_batch,
#            num_classes,
#            scale,
#            margin
#        )
#        optimizer.apply_gradients(zip(grads, model.trainable_weights))
#        if step % 1000 == 0:
#            LOGGER.info(
#                (
#                    f' Training loss (for one batch) at step {step}:'
#                    f' {float(loss_value):.3f}'
#                )
#            )
#            LOGGER.info(f' Seen so far: {step * batch_size} samples')
#    return train_loss_function(loss_value)
#
#
#
#
#
# @tf.function
# def _train_step_joint_learn(
#        vgg,
#        srfr_model,
#        sr_discriminator_model,
#        natural_batch,
#        num_classes_natural,
#        synthetic_batch,
#        num_classes_synthetic,
#        sr_weight: float,
#        scale: float,
#        margin: float,
#    ):
#    """Does a training step
#
#    ### Parameters:
#        srfr_model:
#        sr_discriminator_model:
#        natural_batch:
#        num_classes_natural: Total number of classes in the natural dataset.
#        synthetic_batch:
#        num_classes_synthetic: Total number of classes in the synthetic dataset.
#        sr_weight: Weight for the SR Loss.
#        scale:
#        margin:
#
#    ### Returns:
#        (srfr_loss, srfr_grads, discriminator_loss, discriminator_grads)\
# The loss value and the gradients for SRFR network, as well as the loss value\
# and the gradients for the Discriminative network.
#    """
#    natural_images, natural_classes = natural_batch
#    synthetic_images, groud_truth_images, synthetic_classes = synthetic_batch
#    with tf.GradientTape() as srfr_tape, \
#        tf.GradientTape() as discriminator_tape:
#        (
#            synthetic_sr_images,
#            synthetic_embeddings,
#            synthetic_fc_weights,
#            natural_sr_images,
#            natural_embeddings,
#            natural_fc_weights
#        ) = srfr_model(synthetic_images, natural_images)
#        discriminator_sr_predictions = sr_discriminator_model(synthetic_sr_images)
#        discriminator_gt_predictions = sr_discriminator_model(groud_truth_images)
#        synthetic_face_recognition = (
#            synthetic_embeddings,
#            synthetic_classes,
#            synthetic_fc_weights,
#            num_classes_synthetic,
#        )
#        natural_face_recognition = (
#            natural_embeddings,
#            natural_classes,
#            natural_fc_weights,
#            num_classes_natural,
#        )
#        srfr_loss = compute_joint_loss(
#            vgg,
#            synthetic_images,
#            groud_truth_images,
#            discriminator_sr_predictions,
#            discriminator_gt_predictions,
#            synthetic_face_recognition,
#            natural_face_recognition,
#            sr_weight,
#            scale,
#            margin
#        )
#        discriminator_loss = compute_discriminator_loss(
#            discriminator_sr_predictions,
#            discriminator_gt_predictions,
#        )
#    srfr_grads = srfr_tape.gradient(srfr_loss, srfr_model.trainable_weights)
#    discriminator_grads = discriminator_tape.gradient(
#        discriminator_loss,
#        sr_discriminator_model.trainable_weights,
#    )
#    return srfr_loss, srfr_grads, discriminator_loss, discriminator_grads
#
# @tf.function
# def _train_with_natural_images(
#        vgg,
#        srfr_model,
#        discriminator_model,
#        batch_size,
#        srfr_optimizer,
#        discriminator_optimizer,
#        train_loss_function,
#        synthetic_dataset,
#        num_classes_synthetic: int,
#        natural_dataset,
#        num_classes_natural: int,
#        sr_weight: float = 0.1,
#        scale: float = 64,
#        margin: float = 0.5,
#    ) -> float:
#    for step, (natural_batch, synthetic_batch) in enumerate(
#                zip(natural_dataset, synthetic_dataset)
#        ):
#        (
#            srfr_loss,
#            srfr_grads,
#            discriminator_loss,
#            discriminator_grads,
#        ) = _train_step_joint_learn(
#            vgg,
#            srfr_model,
#            discriminator_model,
#            natural_batch,
#            num_classes_natural,
#            synthetic_batch,
#            num_classes_synthetic,
#            sr_weight,
#            scale,
#            margin,
#        )
#        srfr_optimizer.apply_gradients(
#            zip(srfr_grads, srfr_model.trainable_weights)
#        )
#        discriminator_optimizer.apply_gradients(
#            zip(discriminator_grads, discriminator_model.trainable_weights)
#        )
#        if step % 1000 == 0:
#            LOGGER.info(
#                (
#                    f' SRFR Training loss (for one batch) at step {step}:'
#                    f' {float(srfr_loss):.3f}'
#                )
#            )
#            LOGGER.info(
#                (
#                    f' Discriminator loss (for one batch) at step {step}:'
#                    f' {float(discriminator_loss):.3f}'
#                )
#            )
#            LOGGER.info(f' Seen so far: {step * batch_size} samples')
#    return (
#        train_loss_function(srfr_loss),
#        train_loss_function(discriminator_loss),
#    )
#