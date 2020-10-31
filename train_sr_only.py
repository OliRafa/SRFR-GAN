"""Main training module for the Joint Learning Super Resolution Face\
 Recognition.
"""
import logging
from functools import partial
from pathlib import Path

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


from base_training import BaseTraining
from models.discriminator import DiscriminatorNetwork
from models.srfr_sr_only import SrfrSrOnly
from services.losses import Loss
from use_cases.train.train_model_sr_only import TrainModelSrOnlyUseCase
from utils.input_data import parseConfigsFile


class TrainingSrOnly(BaseTraining):
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

        srfr_model, discriminator_model = self._instantiate_models(
            synthetic_num_classes, network_settings, preprocess_settings
        )

        train_model_sr_only_use_case = TrainModelSrOnlyUseCase(
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
            discriminator_model=discriminator_model,
            batch_size=BATCH_SIZE,
            synthetic_train=synthetic_train,
            synthetic_test=synthetic_test,
            num_classes=synthetic_num_classes,
            train_settings=train_settings,
            hparams=hyperparameters,
        )
        _train = use_named_args(dimensions=dimensions)(_training)

        initial_parameters = [0.0002, 2.0e3, 0.9, 1.0, 0.005, 0.01]

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
        learning_rate_decay_steps,
        beta_1,
        perceptual_weight,
        generator_weight,
        l1_weight,
        train_model_use_case: TrainModelSrOnlyUseCase,
        srfr_model,
        discriminator_model,
        batch_size,
        synthetic_train,
        synthetic_test,
        num_classes,
        train_settings,
        hparams,
    ):
        (
            HP_LEARNING_RATE,
            HP_LEARNING_RATE_DECAY_STEPS,
            HP_BETA_1,
            HP_PERCEPTUAL_WEIGHT,
            HP_GENERATOR_WEIGHT,
            HP_L1_WEIGHT,
        ) = hparams
        tensorboard_params = {
            HP_LEARNING_RATE: learning_rate,
            HP_LEARNING_RATE_DECAY_STEPS: learning_rate_decay_steps,
            HP_BETA_1: beta_1,
            HP_PERCEPTUAL_WEIGHT: perceptual_weight,
            HP_GENERATOR_WEIGHT: generator_weight,
            HP_L1_WEIGHT: l1_weight,
        }
        learning_rate = self._instantiate_learning_rate(
            learning_rate, learning_rate_decay_steps
        )
        srfr_optimizer, discriminator_optimizer = self._instantiate_optimizers(
            learning_rate, beta_1, train_settings
        )

        summary_writer = self._create_summary_writer()
        metrics = self._instantiate_metrics()

        loss = Loss(
            metrics,
            batch_size,
            summary_writer,
            perceptual_weight,
            generator_weight,
            l1_weight,
        )

        train_model_use_case.summary_writer = summary_writer

        return train_model_use_case.execute(
            srfr_model,
            discriminator_model,
            srfr_optimizer,
            discriminator_optimizer,
            synthetic_train,
            synthetic_test,
            loss,
            tensorboard_params,
        )

    def _instantiate_metrics(self):
        with self.strategy.scope():
            return tf.keras.metrics.Mean(name="mean_psnr")

    def _instantiate_models(
        self,
        synthetic_num_classes,
        network_settings,
        preprocess_settings,
    ):
        self.logger.info(" -------- Creating Models --------")

        with self.strategy.scope():
            srfr_model = SrfrSrOnly(
                num_filters=network_settings["num_filters"],
                num_gc=network_settings["gc"],
                num_blocks=network_settings["num_blocks"],
                residual_scailing=network_settings["residual_scailing"],
                training=True,
                input_shape=preprocess_settings["image_shape_low_resolution"],
            )
            discriminator_model = DiscriminatorNetwork()

        return srfr_model, discriminator_model

    @staticmethod
    def _create_dimensions():
        return [
            Real(low=1.0e-3, high=7.0e-3, prior="log-uniform", name="learning_rate"),
            Integer(low=1_000, high=6_000, name="learning_rate_decay_steps"),
            Real(low=0.5, high=0.9, prior="log-uniform", name="beta_1"),
            Real(low=1.0e-3, high=1.0, prior="log-uniform", name="perceptual_weight"),
            Real(low=1.0e-3, high=7.0e-2, prior="log-uniform", name="generator_weight"),
            Real(low=1.0e-2, high=9.0e-2, prior="log-uniform", name="l1_weight"),
        ]

    @staticmethod
    def _create_hyprparameters_domain():
        HP_LEARNING_RATE = hp.HParam(
            "learning_rate",
            hp.RealInterval(1.0e-3, 7.0e-3),
        )
        HP_LEARNING_RATE_DECAY_STEPS = hp.HParam(
            "learning_rate_decay_steps",
            hp.Discrete(range(1_000, 6_000 + 1)),
        )
        HP_BETA_1 = hp.HParam("beta_1", hp.RealInterval(0.5, 0.9))
        HP_PERCEPTUAL_WEIGHT = hp.HParam(
            "perceptual_weight", hp.RealInterval(1.0e-3, 1.0)
        )
        HP_GENERATOR_WEIGHT = hp.HParam(
            "generator_weight", hp.RealInterval(1.0e-3, 7.0e-2)
        )
        HP_L1_WEIGHT = hp.HParam("l1_weight", hp.RealInterval(1.0e-2, 9.0e-2))

        return [
            HP_LEARNING_RATE,
            HP_LEARNING_RATE_DECAY_STEPS,
            HP_BETA_1,
            HP_PERCEPTUAL_WEIGHT,
            HP_GENERATOR_WEIGHT,
            HP_L1_WEIGHT,
        ]


if __name__ == "__main__":
    TrainingSrOnly().train()
