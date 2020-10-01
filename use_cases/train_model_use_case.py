from pathlib import Path

import tensorflow as tf
from services.losses import Loss
from services.train import Train
from utils.timing import TimingLogger


class TrainModelUseCase:
    def __init__(
        self,
        strategy,
        loss: Loss,
        summary_writer,
        timing: TimingLogger,
        logger,
        checkpoint,
        checkpoint_manager,
    ):
        self.strategy = strategy
        self.loss = loss
        self.timing = timing
        self.summary_writer = summary_writer
        self.logger = logger
        self.checkpoint = checkpoint
        self.checkpoint_manager = checkpoint_manager

        self.CACHE_PATH = Path.cwd().joinpath("data", "temp", "train_dataset")
        if not self.CACHE_PATH.is_dir():
            self.CACHE_PATH.mkdir(parents=True)

    def execute(
        self,
        sr_model,
        sr_optimizer,
        discriminator_model,
        discriminator_optimizer,
        dataset,
        num_classes: int,
        batch_size: int,
        epochs: int,
    ):
        batch_size, num_classes = self._instantiate_values_as_tensors(
            batch_size, num_classes
        )
        self._try_restore_checkpoint()
        self.timing.start("TrainModelUseCase")

        train = Train(
            self.strategy,
            sr_model,
            sr_optimizer,
            discriminator_model,
            discriminator_optimizer,
            self.summary_writer,
            self.checkpoint,
            self.checkpoint_manager,
            self.loss,
        )

        for epoch in range(int(self.checkpoint.epoch), epochs + 1):
            self.logger.info(f" Start of epoch {epoch}")

            train.train_with_synthetic_images_only(batch_size, dataset)

            elapsed_time = self.timing.end("TrainModelUseCase", True)
            with self.summary_writer.as_default():
                tf.summary.scalar("Training Time Per Epoch", elapsed_time, step=epoch)

            self.checkpoint.epoch.assign_add(1)

        train.save_model()

    def _instantiate_values_as_tensors(self, batch_size: int, num_classes: int):
        with self.strategy.scope():
            batch_size = tf.constant(batch_size, dtype=tf.float32, name="batch_size")
            num_classes = tf.constant(num_classes, dtype=tf.int32, name="num_classes")

        return batch_size, num_classes

    def _try_restore_checkpoint(self):
        with self.strategy.scope():
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            if self.checkpoint_manager.latest_checkpoint:
                ...
                # self.logger.info(
                #    f" Restored from {self.checkpoint_manager.latest_checkpoint}"
                # )
            else:
                self.logger.info(" Initializing from scratch.")
