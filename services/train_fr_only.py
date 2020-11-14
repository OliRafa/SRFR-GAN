import tensorflow as tf

from services.train import Train


class TrainFrOnly(Train):
    def __init__(
        self,
        strategy,
        srfr_model,
        srfr_optimizer,
        summary_writer,
        checkpoint,
        manager,
        loss,
        logger,
    ):
        super().__init__(
            strategy=strategy,
            srfr_model=srfr_model,
            srfr_optimizer=srfr_optimizer,
            discriminator_model=None,
            discriminator_optimizer=None,
            train_summary_writer=summary_writer,
            checkpoint=checkpoint,
            manager=manager,
            loss=loss,
        )
        self.logger = logger

    def train(
        self,
        batch_size,
        train_dataset,
    ) -> float:
        for (
            _,
            groud_truth_images,
            groud_truth_classes,
        ) in train_dataset:
            srfr_loss = self._train_step(
                groud_truth_images,
                groud_truth_classes,
            )
            if int(self.checkpoint.step) % 1000 == 0:
                self.save_model()

            self._save_metrics(
                self.checkpoint.step,
                srfr_loss,
                batch_size,
            )
            self.checkpoint.step.assign_add(1)

    @tf.function
    def _train_step(
        self,
        synthetic_images,
        groud_truth_classes,
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
        srfr_loss = self.strategy.run(
            self._step_function,
            args=(
                synthetic_images,
                groud_truth_classes,
            ),
        )

        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, srfr_loss, None)

    @tf.function
    def _step_function(self, low_resolution_batch, ground_truth_classes):
        with tf.GradientTape() as srfr_tape:
            embeddings = self.srfr_model(low_resolution_batch)

            srfr_loss = self.losses.compute_arcloss(
                embeddings,
                ground_truth_classes,
            )
            divided_srfr_loss = srfr_loss / self.strategy.num_replicas_in_sync
            srfr_scaled_loss = self.srfr_optimizer.get_scaled_loss(divided_srfr_loss)

        srfr_grads = srfr_tape.gradient(
            srfr_scaled_loss, self.srfr_model.trainable_weights
        )
        self.srfr_optimizer.apply_gradients(
            zip(
                self.srfr_optimizer.get_unscaled_gradients(srfr_grads),
                self.srfr_model.trainable_weights,
            )
        )
        return srfr_loss

    def _save_metrics(
        self,
        step,
        fr_loss,
        batch_size,
    ) -> None:
        step = int(self.checkpoint.step)
        batch_size = int(batch_size)

        self.logger.info(
            (
                f" SRFR Training loss (for one batch) at step {step}:"
                f" {float(fr_loss):.3f}"
            )
        )
        self.logger.info(f" Seen so far: {step * batch_size} samples")

        with self.train_summary_writer.as_default():
            tf.summary.scalar(
                f"ArcLoss",
                float(fr_loss),
                step=step,
            )
