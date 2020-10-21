from datetime import datetime
from pathlib import Path
from typing import Dict

import tensorflow as tf
from services.train import Train, generate_num_epochs
from tensorboard.plugins.hparams import api as hp
from utils.input_data import parseConfigsFile
from utils.timing import TimingLogger


class TrainModelUseCase:
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
        self.checkpoint = None
        self.checkpoint_manager = None

        self.network_settings = {}
        self.train_settings = self._get_training_settings()
        self.preprocess_settings = {}
        self.BATCH_SIZE = batch_size
        self.EPOCHS = generate_num_epochs(
            self.train_settings["iterations"],
            dataset_len,
            self.BATCH_SIZE,
        )

        self.summary_writer = summary_writer

    def execute(
        self,
        srfr_model,
        discriminator_model,
        srfr_optimizer,
        discriminator_optimizer,
        synthetic_train,
        synthetic_test,
        loss,
        hparams: Dict,
    ):
        self.checkpoint, self.checkpoint_manager = self._create_checkpoint_and_manager(
            srfr_model, discriminator_model, srfr_optimizer, discriminator_optimizer
        )

        self.BATCH_SIZE = self._instantiate_values_as_tensors(self.BATCH_SIZE)
        self._try_restore_checkpoint()
        self.timing.start("TrainModelUseCase")

        train = Train(
            self.strategy,
            srfr_model,
            srfr_optimizer,
            discriminator_model,
            discriminator_optimizer,
            self.summary_writer,
            self.checkpoint,
            self.checkpoint_manager,
            loss,
        )

        with self.summary_writer.as_default():
            hp.hparams(hparams)

        self.logger.info(" -------- Starting Training --------")

        initial_epoch = int(self.checkpoint.epoch)
        for epoch in range(initial_epoch, self.EPOCHS + 1):
            self.logger.info(f" Start of epoch {epoch}")

            train.train_with_synthetic_images_only(self.BATCH_SIZE, synthetic_train)
            accuracy = train.test_model(synthetic_test)

            with self.summary_writer.as_default():
                tf.summary.scalar("Accuracy", accuracy, step=int(self.checkpoint.epoch))

            _ = self.timing.end("TrainModelUseCase", True)

            self.checkpoint.epoch.assign_add(1)

        train.save_model()
        return -accuracy.numpy().tolist()

    @staticmethod
    def _create_checkpoint_and_manager(
        srfr_model, discriminator_model, srfr_optimizer, discriminator_optimizer
    ):
        checkpoint = tf.train.Checkpoint(
            epoch=tf.Variable(1),
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
                Path.cwd().joinpath("data", "training_checkpoints", current_time)
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
