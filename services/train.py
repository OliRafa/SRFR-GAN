"""This module contains functions used for training."""
import logging

import tensorflow as tf

from services.losses import Loss

LOGGER = logging.getLogger(__name__)


class Train:
    def __init__(
        self,
        strategy,
        srfr_model,
        srfr_optimizer,
        discriminator_model,
        discriminator_optimizer,
        train_summary_writer,
        checkpoint,
        manager,
        loss,
    ):
        self.strategy = strategy
        self.srfr_model = srfr_model
        self.srfr_optimizer = srfr_optimizer
        self.discriminator_model = discriminator_model
        self.discriminator_optimizer = discriminator_optimizer
        self.train_summary_writer = train_summary_writer
        self.checkpoint = checkpoint
        self.manager = manager

        self.losses: Loss = loss

    def train_with_synthetic_images_only(
        self,
        batch_size,
        train_dataset,
    ) -> float:
        for (
            synthetic_images,
            groud_truth_images,
            synthetic_classes,
        ) in train_dataset:
            (
                srfr_loss,
                discriminator_loss,
                super_resolution_images,
            ) = self._train_step_synthetic_only(
                synthetic_images,
                groud_truth_images,
                synthetic_classes,
                self.checkpoint.step,
            )
            if int(self.checkpoint.step) % 1000 == 0:
                self.save_model()

            self._save_metrics(
                self.checkpoint.step,
                srfr_loss,
                discriminator_loss,
                batch_size,
                synthetic_images,
                groud_truth_images,
                super_resolution_images,
            )
            self.checkpoint.step.assign_add(1)

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
        step = int(self.checkpoint.step)
        batch_size = int(batch_size)

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

        with self.train_summary_writer.as_default():
            tf.summary.scalar(
                f"SRFR Loss",
                float(srfr_loss),
                step=step,
            )
            tf.summary.scalar(
                f"Discriminator Loss",
                float(discriminator_loss),
                step=step,
            )
            tf.summary.image(
                f"LR Images",
                tf.concat(synthetic_images.values, axis=0),
                max_outputs=10,
                step=step,
            )
            tf.summary.image(
                f"HR Images",
                tf.concat(groud_truth_images.values, axis=0),
                max_outputs=10,
                step=step,
            )
            tf.summary.image(
                f"SR Images",
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

    @tf.function
    def _step_function(
        self, low_resolution_batch, groud_truth_batch, ground_truth_classes, step
    ):
        with tf.GradientTape() as srfr_tape, tf.GradientTape() as discriminator_tape:
            (super_resolution_images, embeddings, predictions) = self.srfr_model(
                low_resolution_batch
            )
            discriminator_sr_predictions = self.discriminator_model(
                super_resolution_images
            )
            discriminator_gt_predictions = self.discriminator_model(groud_truth_batch)
            synthetic_face_recognition = (
                embeddings,
                predictions,
                ground_truth_classes,
            )
            srfr_loss = self.losses.compute_joint_loss(
                super_resolution_images,
                groud_truth_batch,
                discriminator_sr_predictions,
                discriminator_gt_predictions,
                synthetic_face_recognition,
                step,
            )
            discriminator_loss = self.losses.compute_discriminator_loss(
                discriminator_sr_predictions,
                discriminator_gt_predictions,
            )
            divided_srfr_loss = srfr_loss / self.strategy.num_replicas_in_sync
            divided_discriminator_loss = (
                discriminator_loss / self.strategy.num_replicas_in_sync
            )
            srfr_scaled_loss = self.srfr_optimizer.get_scaled_loss(divided_srfr_loss)
            discriminator_scaled_loss = self.discriminator_optimizer.get_scaled_loss(
                divided_discriminator_loss
            )

        srfr_grads = srfr_tape.gradient(
            srfr_scaled_loss, self.srfr_model.trainable_weights
        )
        discriminator_grads = discriminator_tape.gradient(
            discriminator_scaled_loss,
            self.discriminator_model.trainable_weights,
        )
        self.srfr_optimizer.apply_gradients(
            zip(
                self.srfr_optimizer.get_unscaled_gradients(srfr_grads),
                self.srfr_model.trainable_weights,
            )
        )
        self.discriminator_optimizer.apply_gradients(
            zip(
                self.discriminator_optimizer.get_unscaled_gradients(
                    discriminator_grads
                ),
                self.discriminator_model.trainable_weights,
            )
        )
        return srfr_loss, discriminator_loss, super_resolution_images

    @tf.function
    def _train_step_synthetic_only(
        self,
        synthetic_images,
        groud_truth_images,
        synthetic_classes,
        step,
    ):
        """Does a training step

        Parameters
        ----------
            model:
            images: Batch of images for training.
            classes: Batch of classes to compute the loss.
            num_classes: Total number of classes in the dataset.

        Returns
        -------
            (srfr_loss, srfr_grads, discriminator_loss, discriminator_grads)
            The loss value and the gradients for SRFR network, as well as the
            loss value and the gradients for the Discriminative network.
        """
        (srfr_loss, discriminator_loss, super_resolution_images) = self.strategy.run(
            self._step_function,
            args=(
                synthetic_images,
                groud_truth_images,
                synthetic_classes,
                step,
            ),
        )

        new_srfr_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, srfr_loss, None
        )
        new_discriminator_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            discriminator_loss,
            None,
        )

        return new_srfr_loss, new_discriminator_loss, super_resolution_images

    def test_model(self, dataset, num_classes) -> None:
        self.losses.reset_accuracy_metric()
        for (
            synthetic_images,
            groud_truth_images,
            synthetic_classes,
        ) in dataset:
            self._call_test(synthetic_images, synthetic_classes, num_classes)

        return self.losses.get_accuracy_results() * 100

    @tf.function
    def _call_test(self, synthetic_images, synthetic_classes, num_classes):
        self.strategy.run(
            self._call_accuracy_calc,
            args=(synthetic_images, synthetic_classes, num_classes),
        )

    def _call_accuracy_calc(
        self, synthetic_images, synthetic_classes, num_classes
    ) -> None:
        (super_resolution_images, embeddings, predictions) = self.srfr_model(
            synthetic_images, training=False
        )
        # predictions = tf.argmax(predictions, axis=1, output_type=tf.int32)
        synthetic_classes = tf.one_hot(synthetic_classes, num_classes)
        self.losses.calculate_accuracy(predictions, synthetic_classes)
