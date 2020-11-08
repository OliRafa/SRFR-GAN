"""Main training module for the Joint Learning Super Resolution Face\
 Recognition.
"""
import logging
from functools import partial
from pathlib import Path
from typing import Dict

import tensorflow as tf
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from tensorboard.plugins.hparams import api as hp

from utils.timing import TimingLogger

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("set_memory_growth ok!")
    except RuntimeError as e:
        print("set_memory_growth failed!")
        print(str(e))


from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_addons.optimizers import AdamW

from base_training import BaseTraining
from models.discriminator import DiscriminatorNetwork
from models.srfr_fr_only import SrfrFrOnly
from services.losses import Loss
from use_cases.train.train_model_fr_only import TrainModelFrOnlyUseCase
from utils.input_data import LFW, parseConfigsFile

AUTOTUNE = tf.data.experimental.AUTOTUNE


class TrainingFrOnly(BaseTraining):
    def __init__(self):
        logging.basicConfig(
            filename="train_logs.txt",
            level=logging.DEBUG,
        )
        logger = logging.getLogger(__name__)

        strategy = tf.distribute.MirroredStrategy()
        timing = TimingLogger()
        super().__init__(logger, strategy, timing)

    def train(self):
        """Main training function."""
        self.timing.start()

        dimensions = self._create_dimensions()
        hyperparameters = self._create_hyprparameters_domain()
        with tf.summary.create_file_writer(
            str(Path.cwd().joinpath("output", "logs", "hparam_tuning"))
        ).as_default():
            hp.hparams_config(
                hparams=hyperparameters,
                metrics=[hp.Metric("accuracy", display_name="Accuracy")],
            )

        (
            network_settings,
            train_settings,
            preprocess_settings,
        ) = parseConfigsFile(["network", "train", "preprocess"])

        BATCH_SIZE = train_settings["batch_size"] * self.strategy.num_replicas_in_sync

        (
            synthetic_train,
            synthetic_test,
            synthetic_dataset_len,
            synthetic_num_classes,
        ) = self._get_datasets(BATCH_SIZE)

        validation_dataset = self._get_validation_dataset(BATCH_SIZE)

        srfr_model = self._instantiate_models(
            synthetic_num_classes, network_settings, preprocess_settings, train_settings
        )

        train_model_sr_only_use_case = TrainModelFrOnlyUseCase(
            self.strategy,
            TimingLogger(),
            self.logger,
            BATCH_SIZE,
            synthetic_dataset_len,
        )

        _training = partial(
            self._fitness_function,
            train_model_use_case=train_model_sr_only_use_case,
            srfr_model=srfr_model,
            batch_size=BATCH_SIZE,
            synthetic_train=synthetic_train,
            validation_dataset=validation_dataset,
            num_classes=synthetic_num_classes,
            train_settings=train_settings,
            hparams=hyperparameters,
        )
        _train = use_named_args(dimensions=dimensions)(_training)

        initial_parameters = [0.0002, 0.9, 10_000, 0.0005]

        search_result = gp_minimize(
            func=_train,
            dimensions=dimensions,
            acq_func="EI",
            n_calls=20,
            x0=initial_parameters,
        )

        self.logger.info(f"Best hyperparameters: {search_result.x}")

    def _fitness_function(
        self,
        learning_rate,
        beta_1,
        learning_rate_decay_steps,
        weight_decay,
        train_model_use_case: TrainModelFrOnlyUseCase,
        srfr_model,
        batch_size,
        synthetic_train,
        validation_dataset,
        num_classes,
        train_settings,
        hparams,
    ):
        (
            HP_LEARNING_RATE,
            HP_BETA_1,
            HP_LEARNING_RATE_DECAY_STEPS,
            HP_WEIGHT_DECAY,
        ) = hparams
        tensorboard_params = {
            HP_LEARNING_RATE: learning_rate,
            HP_BETA_1: beta_1,
            HP_LEARNING_RATE_DECAY_STEPS: learning_rate_decay_steps,
            HP_WEIGHT_DECAY: weight_decay,
        }

        learning_rate = self._instantiate_learning_rate(
            learning_rate, learning_rate_decay_steps
        )
        srfr_optimizer = self._instantiate_optimizers(
            learning_rate, beta_1, weight_decay, train_settings
        )

        summary_writer = self._create_summary_writer()

        loss = Loss(
            None,
            batch_size,
            summary_writer,
            scale=train_settings["scale"],
            margin=train_settings["angular_margin"],
            num_classes=num_classes,
        )

        train_model_use_case.summary_writer = summary_writer

        return train_model_use_case.execute(
            srfr_model,
            srfr_optimizer,
            synthetic_train,
            validation_dataset,
            loss,
            tensorboard_params,
        )

    def _get_validation_dataset(self, BATCH_SIZE: int):
        lfw = LFW()
        left_pairs, right_pairs, is_same_list = lfw.get_dataset()
        left_pairs = left_pairs.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
        left_pairs = self.strategy.experimental_distribute_dataset(left_pairs)

        right_pairs = right_pairs.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
        right_pairs = self.strategy.experimental_distribute_dataset(right_pairs)

        return left_pairs, right_pairs, is_same_list

    def _instantiate_models(
        self,
        num_classes: int,
        network_settings: Dict,
        preprocess_settings: Dict,
        train_settings: Dict,
    ):
        self.logger.info(" -------- Creating Models --------")

        with self.strategy.scope():
            return SrfrFrOnly(
                depth=50,
                categories=network_settings["embedding_size"],
                num_classes=num_classes,
                scale=train_settings["scale"],
                training=True,
                input_shape=preprocess_settings["image_shape_low_resolution"],
            )

    def _instantiate_optimizers(
        self, learning_rate, beta_1, weight_decay, train_settings
    ):
        self.logger.info(" -------- Creating Optimizers --------")

        with self.strategy.scope():
            srfr_optimizer = AdamW(
                learning_rate=learning_rate,
                beta_1=beta_1,
                beta_2=train_settings["beta_2"],
                weight_decay=weight_decay,
                name="adam_srfr",
            )
            return mixed_precision.LossScaleOptimizer(
                srfr_optimizer,
                loss_scale="dynamic",
            )

    @staticmethod
    def _create_dimensions():
        return [
            Real(low=1.0e-4, high=7.0e-3, prior="log-uniform", name="learning_rate"),
            Real(low=0.5, high=0.9, prior="log-uniform", name="beta_1"),
            Integer(low=1_000, high=10_000, name="learning_rate_decay_steps"),
            Real(low=1.0e-4, high=1.0e-3, prior="log-uniform", name="weight_decay"),
        ]

    @staticmethod
    def _create_hyprparameters_domain():
        HP_LEARNING_RATE = hp.HParam(
            "learning_rate",
            hp.RealInterval(1.0e-4, 7.0e-3),
        )
        HP_BETA_1 = hp.HParam("beta_1", hp.RealInterval(0.5, 0.9))
        HP_LEARNING_RATE_DECAY_STEPS = hp.HParam(
            "learning_rate_decay_steps", hp.Discrete(range(1_000, 10_000 + 1))
        )
        HP_WEIGHT_DECAY = hp.HParam("weight_decay", hp.RealInterval(1.0e-4, 1.0e-3))

        return [
            HP_LEARNING_RATE,
            HP_BETA_1,
            HP_LEARNING_RATE_DECAY_STEPS,
            HP_WEIGHT_DECAY,
        ]


if __name__ == "__main__":
    TrainingFrOnly().train()
