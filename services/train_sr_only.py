import tensorflow as tf
from utils.common import denormalize_tensor, tensor_to_uint8

from services.train import Train


class TrainSrOnly(Train):
    def __init__(
        self,
        strategy,
        srfr_model,
        srfr_optimizer,
        discriminator_model,
        discriminator_optimizer,
        summary_writer,
        checkpoint,
        manager,
        loss,
    ):
        super().__init__(
            strategy,
            srfr_model,
            srfr_optimizer,
            discriminator_model,
            discriminator_optimizer,
            summary_writer,
            checkpoint,
            manager,
            loss,
        )

    def train_with_synthetic_images_only(
        self,
        batch_size,
        train_dataset,
    ) -> float:
        for (
            low_resolution_images,
            groud_truth_images,
            _,
        ) in train_dataset:
            (
                srfr_loss,
                discriminator_loss,
                super_resolution_images,
            ) = self._train_step_synthetic_only(
                low_resolution_images,
                groud_truth_images,
                self.checkpoint.step,
            )
            if int(self.checkpoint.step) % 1000 == 0:
                self.save_model()

            self._save_metrics(
                self.checkpoint.step,
                srfr_loss,
                discriminator_loss,
                batch_size,
                low_resolution_images,
                groud_truth_images,
                super_resolution_images,
            )
            self.checkpoint.step.assign_add(1)

    @tf.function
    def _train_step_synthetic_only(
        self,
        synthetic_images,
        groud_truth_images,
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

    @tf.function
    def _step_function(self, low_resolution_batch, groud_truth_batch, step):
        with tf.GradientTape() as srfr_tape, tf.GradientTape() as discriminator_tape:
            super_resolution_images = self.srfr_model(low_resolution_batch)
            discriminator_sr_predictions = self.discriminator_model(
                super_resolution_images
            )
            discriminator_gt_predictions = self.discriminator_model(groud_truth_batch)

            srfr_loss = self.losses.compute_generator_loss(
                super_resolution_images,
                groud_truth_batch,
                discriminator_sr_predictions,
                discriminator_gt_predictions,
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

    def test_model(self, dataset) -> None:
        self.losses.reset_accuracy_metric()
        for (
            synthetic_images,
            groud_truth_images,
            _,
        ) in dataset:
            self._call_test(synthetic_images, groud_truth_images)

        return self.losses.get_accuracy_results() * 100

    @tf.function
    def _call_test(self, synthetic_images, groud_truth_images):
        self.strategy.run(
            self._call_accuracy_calc,
            args=(synthetic_images, groud_truth_images),
        )

    def _call_accuracy_calc(self, synthetic_images, groud_truth_images) -> None:
        super_resolution_images = self.srfr_model(synthetic_images, training=False)

        super_resolution_images = tensor_to_uint8(
            denormalize_tensor(super_resolution_images)
        )
        groud_truth_images = tensor_to_uint8(denormalize_tensor(groud_truth_images))

        psnr_values = self.losses.calculate_psnr(
            super_resolution_images, groud_truth_images
        )
        self.losses.calculate_mean_accuracy(psnr_values)
