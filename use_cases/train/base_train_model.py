from datetime import datetime
from pathlib import Path

import tensorflow as tf
from utils.input_data import parseConfigsFile
from utils.timing import TimingLogger


class BaseTrainModelUseCase:
    def __init__(
        self,
        strategy,
        timing: TimingLogger,
        logger,
        batch_size: int,
        dataset_len: int,
        summary_writer=None,
    ):
        self.strategy = strategy
        self.timing = timing
        self.logger = logger
        self.summary_writer = summary_writer

        self.checkpoint = None
        self.checkpoint_manager = None

        self.train_settings = self._get_training_settings()
        self.BATCH_SIZE = batch_size
        self.EPOCHS = self._generate_num_epochs(
            self.train_settings["iterations"],
            dataset_len,
        )

    def _generate_num_epochs(self, iterations, len_dataset):
        self.logger.info(
            f" Generating number of epochs for {iterations} iterations,\
        {len_dataset} dataset length and {self.BATCH_SIZE} batch size."
        )
        train_size = tf.math.ceil(len_dataset / self.BATCH_SIZE)
        epochs = tf.cast(tf.math.ceil(iterations / train_size), dtype=tf.int32)
        self.logger.info(f" Number of epochs: {epochs}.")
        return epochs

    @staticmethod
    def _create_checkpoint_and_manager(
        srfr_model, discriminator_model, srfr_optimizer, discriminator_optimizer
    ):
        checkpoint = tf.train.Checkpoint(
            epoch=tf.Variable(1, dtype=tf.int64),
            step=tf.Variable(1, dtype=tf.int64),
            srfr_model=srfr_model,
            discriminator_model=discriminator_model,
            srfr_optimizer=srfr_optimizer,
            discriminator_optimizer=discriminator_optimizer,
        )

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        manager = tf.train.CheckpointManager(
            checkpoint,
            directory=str(
                Path.cwd().joinpath("output", "training_checkpoints", current_time)
            ),
            max_to_keep=None,
        )
        return checkpoint, manager

    def _instantiate_values_as_tensors(self, batch_size: int):
        with self.strategy.scope():
            batch_size = tf.constant(batch_size, dtype=tf.float32, name="batch_size")

        return batch_size

    def _try_restore_checkpoint(self):
        with self.strategy.scope():
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            if self.checkpoint_manager.latest_checkpoint:
                self.logger.info(
                    f" Restored from {self.checkpoint_manager.latest_checkpoint}"
                )
            else:
                self.logger.info(" Initializing from scratch.")

    @staticmethod
    def _get_training_settings():
        return parseConfigsFile(["train"])
