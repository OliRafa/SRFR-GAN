from datetime import datetime
from pathlib import Path
from typing import Dict

import tensorflow as tf
from services.train_fr_only import TrainFrOnly
from tensorboard.plugins.hparams import api as hp
from use_cases.train.base_train_model import BaseTrainModelUseCase
from use_cases.validate_model_use_case import ValidateModelUseCase
from utils.timing import TimingLogger


class TrainModelFrOnlyUseCase(BaseTrainModelUseCase):
    def __init__(
        self,
        strategy,
        timing: TimingLogger,
        logger,
        batch_size: int,
        dataset_len: int,
        summary_writer=None,
    ):
        super().__init__(
            strategy, timing, logger, batch_size, dataset_len, summary_writer
        )
        self.validate_model_use_case = ValidateModelUseCase(
            strategy, summary_writer, timing, logger
        )

    def execute(
        self,
        srfr_model,
        srfr_optimizer,
        train_dataset,
        validation_dataset,
        loss,
        hparams: Dict,
    ):
        self.validate_model_use_case.summary_writer = self.summary_writer
        self.checkpoint, self.checkpoint_manager = self._create_checkpoint_and_manager(
            srfr_model, srfr_optimizer
        )

        self.BATCH_SIZE = self._instantiate_values_as_tensors(self.BATCH_SIZE)
        self.timing.start("TrainModelFrOnlyUseCase")

        train = TrainFrOnly(
            self.strategy,
            srfr_model,
            srfr_optimizer,
            self.summary_writer,
            self.checkpoint,
            self.checkpoint_manager,
            loss,
            self.logger,
        )

        self.logger.info(" -------- Starting Training --------")

        for epoch in range(1, self.EPOCHS + 1):
            self.logger.info(f" Start of epoch {epoch}")

            train.train(self.BATCH_SIZE, train_dataset)
            accuracy = self.validate_model_use_case.execute(
                srfr_model, validation_dataset, self.BATCH_SIZE, self.checkpoint
            )

            with self.summary_writer.as_default():
                hp.hparams(hparams)
                tf.summary.scalar("accuracy", accuracy, step=int(self.checkpoint.epoch))

            _ = self.timing.end("TrainModelFrOnlyUseCase", True)

            self.checkpoint.epoch.assign_add(1)

        train.save_model()
        return accuracy.tolist()

    @staticmethod
    def _create_checkpoint_and_manager(srfr_model, srfr_optimizer):
        checkpoint = tf.train.Checkpoint(
            epoch=tf.Variable(1, dtype=tf.int64),
            step=tf.Variable(1, dtype=tf.int64),
            srfr_model=srfr_model,
            srfr_optimizer=srfr_optimizer,
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
