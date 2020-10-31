from typing import Dict

import tensorflow as tf
from services.train_sr_only import TrainSrOnly
from tensorboard.plugins.hparams import api as hp
from use_cases.train.base_train_model import BaseTrainModelUseCase
from utils.timing import TimingLogger


class TrainModelSrOnlyUseCase(BaseTrainModelUseCase):
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
        self.timing.start("TrainModelSrOnlyUseCase")

        train = TrainSrOnly(
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

        self.logger.info(" -------- Starting Training --------")

        for epoch in range(1, self.EPOCHS + 1):
            self.logger.info(f" Start of epoch {epoch}")

            train.train_with_synthetic_images_only(self.BATCH_SIZE, synthetic_train)
            accuracy = train.test_model(synthetic_test)

            with self.summary_writer.as_default():
                hp.hparams(hparams)
                tf.summary.scalar("accuracy", accuracy, step=int(self.checkpoint.epoch))

            _ = self.timing.end("TrainModelSrOnlyUseCase", True)

            self.checkpoint.epoch.assign_add(1)

        train.save_model()
        return accuracy.numpy().tolist()
