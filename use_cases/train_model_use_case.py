from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import tensorflow as tf
from models.discriminator import DiscriminatorNetwork
from models.srfr import SRFR
from services.losses import Loss
from services.train import Train, generate_num_epochs
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_addons.optimizers import NovoGrad
from utils.input_data import VggFace2, parseConfigsFile
from utils.timing import TimingLogger

AUTOTUNE = tf.data.experimental.AUTOTUNE


class TrainModelUseCase:
    def __init__(
        self,
        strategy,
        timing: TimingLogger,
        logger,
    ):
        self.strategy = strategy
        self.timing = timing
        self.logger = logger
        self.checkpoint = None
        self.checkpoint_manager = None

        self.network_settings = {}
        self.train_settings = {}
        self.preprocess_settings = {}
        self.EPOCHS = 0

        self._get_configs()
        self.BATCH_SIZE = (
            self.train_settings["batch_size"] * self.strategy.num_replicas_in_sync
        )
        self.summary_writer, self.summary_test = self._create_summary_writer()

        self.loss = Loss(
            self._instantiate_metrics(),
            self.BATCH_SIZE,
            self.summary_writer,
            self.train_settings["perceptual_weight"],
            self.train_settings["generator_weight"],
            self.train_settings["l1_weight"],
            self.train_settings["face_recognition_weight"],
            self.train_settings["super_resolution_weight"],
        )

        self.CACHE_PATH = Path.cwd().joinpath("data", "temp")
        if not self.CACHE_PATH.is_dir():
            self.CACHE_PATH.mkdir(parents=True)

    def execute(
        self,
    ):
        (
            synthetic_train,
            synthetic_test,
            synthetic_dataset_len,
            synthetic_num_classes,
        ) = self._get_datasets()

        self.EPOCHS = generate_num_epochs(
            self.train_settings["iterations"],
            synthetic_dataset_len,
            self.BATCH_SIZE,
        )

        learning_rate = self._instantiate_learning_rate(
            self.train_settings["learning_rate"],
            self.train_settings["learning_rate_decay_steps"],
        )
        (
            srfr_model,
            discriminator_model,
            srfr_optimizer,
            discriminator_optimizer,
        ) = self._instantiate_models(synthetic_num_classes, learning_rate)

        self.checkpoint, self.checkpoint_manager = self._create_checkpoint_and_manager(
            srfr_model, discriminator_model, srfr_optimizer, discriminator_optimizer
        )

        self.BATCH_SIZE, synthetic_num_classes = self._instantiate_values_as_tensors(
            self.BATCH_SIZE, synthetic_num_classes
        )
        self._try_restore_checkpoint()
        self.timing.start("TrainModelUseCase")

        train = Train(
            self.strategy,
            srfr_model,
            srfr_optimizer,
            discriminator_model,
            discriminator_optimizer,
            self.summary_writer,
            self.summary_test,
            self.checkpoint,
            self.checkpoint_manager,
            self.loss,
        )

        self.logger.info(" -------- Starting Training --------")

        for epoch in range(int(self.checkpoint.epoch), self.EPOCHS + 1):
            self.logger.info(f" Start of epoch {epoch}")

            train.train_with_synthetic_images_only(self.BATCH_SIZE, synthetic_train)
            train.test_model(synthetic_test, self.checkpoint.epoch)

            _ = self.timing.end("TrainModelUseCase", True)

            self.checkpoint.epoch.assign_add(1)

        train.save_model()

    def _get_configs(self) -> None:
        (
            self.network_settings,
            self.train_settings,
            self.preprocess_settings,
        ) = parseConfigsFile(["network", "train", "preprocess"])

    def _get_datasets(self):
        self.logger.info(" -------- Importing Datasets --------")

        vgg_dataset = VggFace2(mode="both")
        synthetic_train, synthetic_test = vgg_dataset.get_dataset()
        vgg_dataset._dataset = synthetic_train
        synthetic_train = vgg_dataset.augment_dataset()
        synthetic_train = vgg_dataset.normalize_dataset()

        synthetic_train = synthetic_train.cache(str(self.CACHE_PATH.joinpath("train")))
        # str(temp_folder))
        # synthetic_dataset_len = vgg_dataset.get_dataset_size()
        synthetic_dataset_len = 100_000
        synthetic_num_classes = vgg_dataset.get_number_of_classes()
        synthetic_train = (
            synthetic_train.shuffle(buffer_size=2_048)
            .repeat()
            .batch(self.BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )

        synthetic_train = self.strategy.experimental_distribute_dataset(synthetic_train)

        vgg_dataset._dataset = synthetic_test
        synthetic_test = vgg_dataset.normalize_dataset()
        synthetic_test = synthetic_test.cache(str(self.CACHE_PATH.joinpath("test")))
        synthetic_test = (
            synthetic_test.shuffle(buffer_size=2_048)
            .repeat()
            .batch(self.BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )
        synthetic_test = self.strategy.experimental_distribute_dataset(synthetic_test)

        return (
            synthetic_train,
            synthetic_test,
            synthetic_dataset_len,
            synthetic_num_classes,
        )

    @staticmethod
    def _instantiate_learning_rate(
        learning_rate: float, learning_rate_decay_steps: int
    ):
        return tf.keras.experimental.CosineDecay(
            learning_rate, learning_rate_decay_steps
        )

    def _instantiate_models(
        self,
        synthetic_num_classes,
        learning_rate_scheduler,
    ):
        self.logger.info(" -------- Creating Models and Optimizers --------")

        with self.strategy.scope():
            srfr_model = SRFR(
                num_filters=self.network_settings["num_filters"],
                depth=50,
                categories=self.network_settings["embedding_size"],
                num_gc=self.network_settings["gc"],
                num_blocks=self.network_settings["num_blocks"],
                residual_scailing=self.network_settings["residual_scailing"],
                training=True,
                input_shape=self.preprocess_settings["image_shape_low_resolution"],
                num_classes_syn=synthetic_num_classes,
            )
            discriminator_model = DiscriminatorNetwork()

            srfr_optimizer = NovoGrad(
                learning_rate=learning_rate_scheduler,
                beta_1=self.train_settings["beta_1"],
                beta_2=self.train_settings["beta_2"],
                weight_decay=self.train_settings["weight_decay"],
                name="novograd_srfr",
            )
            srfr_optimizer = mixed_precision.LossScaleOptimizer(
                srfr_optimizer,
                loss_scale="dynamic",
            )
            discriminator_optimizer = NovoGrad(
                learning_rate=learning_rate_scheduler,
                beta_1=self.train_settings["beta_1"],
                beta_2=self.train_settings["beta_2"],
                weight_decay=self.train_settings["weight_decay"],
                name="novograd_discriminator",
            )
            discriminator_optimizer = mixed_precision.LossScaleOptimizer(
                discriminator_optimizer, loss_scale="dynamic"
            )

        return (
            srfr_model,
            discriminator_model,
            srfr_optimizer,
            discriminator_optimizer,
        )

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
        manager = tf.train.CheckpointManager(
            checkpoint,
            directory=str(Path.cwd().joinpath("data", "training_checkpoints")),
            max_to_keep=None,
        )
        return checkpoint, manager

    def _create_summary_writer(self):
        with self.strategy.scope():
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            train = tf.summary.create_file_writer(
                str(
                    Path.cwd().joinpath(
                        "data",
                        "logs",
                        "gradient_tape",
                        current_time,
                        "train",
                    )
                ),
            )
            test = tf.summary.create_file_writer(
                str(
                    Path.cwd().joinpath(
                        "data",
                        "logs",
                        "gradient_tape",
                        current_time,
                        "test",
                    )
                ),
            )

            return train, test

    def _instantiate_values_as_tensors(self, batch_size: int, num_classes: int):
        with self.strategy.scope():
            batch_size = tf.constant(batch_size, dtype=tf.float32, name="batch_size")
            num_classes = tf.constant(num_classes, dtype=tf.int32, name="num_classes")

        return batch_size, num_classes

    def _try_restore_checkpoint(self):
        with self.strategy.scope():
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            if self.checkpoint_manager.latest_checkpoint:
                self.logger.info(
                    f" Restored from {self.checkpoint_manager.latest_checkpoint}"
                )
            else:
                self.logger.info(" Initializing from scratch.")

    def _instantiate_metrics(self):
        with self.strategy.scope():
            return tf.keras.metrics.Accuracy(name="test_accuracy")
