"""Main training module for the Joint Learning Super Resolution Face\
 Recognition.
"""
import logging
from datetime import datetime
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


from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_addons.optimizers import NovoGrad

from models.discriminator import DiscriminatorNetwork
from models.srfr import SRFR
from repositories.casia import CasiaWebface
from services.losses import Loss
from use_cases.train_model_use_case import TrainModelUseCase
from utils.input_data import parseConfigsFile

# Importar Natural DS.
AUTOTUNE = tf.data.experimental.AUTOTUNE

logging.basicConfig(
    filename="train_logs.txt",
    level=logging.DEBUG,
)
LOGGER = logging.getLogger(__name__)
CACHE_PATH = Path.cwd().joinpath("data", "temp")
if not CACHE_PATH.is_dir():
    CACHE_PATH.mkdir(parents=True)


def main():
    """Main training function."""
    timing = TimingLogger()
    timing.start()
    strategy = tf.distribute.MirroredStrategy()
    # strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    dimensions = _create_dimensions()
    hyperparameters = _create_hyprparameters_domain()
    with tf.summary.create_file_writer(
        str(Path.cwd().joinpath("data", "logs", "hparam_tuning"))
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

    BATCH_SIZE = train_settings["batch_size"] * strategy.num_replicas_in_sync

    (
        synthetic_train,
        synthetic_test,
        synthetic_dataset_len,
        synthetic_num_classes,
    ) = _get_datasets(BATCH_SIZE, strategy)

    srfr_model, discriminator_model = _instantiate_models(
        strategy, synthetic_num_classes, network_settings, preprocess_settings
    )

    train_model_use_case = TrainModelUseCase(
        strategy,
        TimingLogger(),
        LOGGER,
        BATCH_SIZE,
        synthetic_dataset_len,
    )

    _training = partial(
        _instantiate_training,
        strategy=strategy,
        train_model_use_case=train_model_use_case,
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

    search_result = gp_minimize(
        func=_train, dimensions=dimensions, acq_func="EI", n_calls=20
    )

    LOGGER.info(f"Best hyperparameters: {search_result.x}")


def _instantiate_training(
    learning_rate,
    learning_rate_decay_steps,
    beta_1,
    face_recognition_weight,
    super_resolution_weight,
    perceptual_weight,
    generator_weight,
    l1_weight,
    strategy,
    train_model_use_case: TrainModelUseCase,
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
        HP_FR_WEIGHT,
        HP_SR_WEIGHT,
        HP_PERCEPTUAL_WEIGHT,
        HP_GENERATOR_WEIGHT,
        HP_L1_WEIGHT,
    ) = hparams
    tensorboard_params = {
        HP_LEARNING_RATE: learning_rate,
        HP_LEARNING_RATE_DECAY_STEPS: learning_rate_decay_steps,
        HP_BETA_1: beta_1,
        HP_FR_WEIGHT: face_recognition_weight,
        HP_SR_WEIGHT: super_resolution_weight,
        HP_PERCEPTUAL_WEIGHT: perceptual_weight,
        HP_GENERATOR_WEIGHT: generator_weight,
        HP_L1_WEIGHT: l1_weight,
    }
    learning_rate = _instantiate_learning_rate(learning_rate, learning_rate_decay_steps)
    srfr_optimizer, discriminator_optimizer = _instantiate_optimizers(
        strategy, learning_rate, beta_1, train_settings
    )

    summary_writer = _create_summary_writer(strategy)
    metrics = _instantiate_metrics(strategy)

    loss = Loss(
        metrics,
        batch_size,
        summary_writer,
        perceptual_weight,
        generator_weight,
        l1_weight,
        face_recognition_weight,
        super_resolution_weight,
    )

    train_model_use_case.summary_writer = summary_writer

    return train_model_use_case.execute(
        srfr_model,
        discriminator_model,
        srfr_optimizer,
        discriminator_optimizer,
        synthetic_train,
        synthetic_test,
        num_classes,
        loss,
        tensorboard_params,
    )


def _create_summary_writer(strategy):
    with strategy.scope():
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        return tf.summary.create_file_writer(
            str(Path.cwd().joinpath("data", "logs", "hparam_tuning", current_time))
        )


def _instantiate_metrics(strategy):
    with strategy.scope():
        return tf.keras.metrics.CategoricalAccuracy(name="test_crossentropy")


def _get_datasets(batch_size, strategy):
    LOGGER.info(" -------- Importing Datasets --------")

    casia_dataset = CasiaWebface()
    synthetic_train = casia_dataset.get_train_dataset()
    synthetic_train = casia_dataset.augment_dataset(synthetic_train)
    synthetic_train = casia_dataset.normalize_dataset(synthetic_train)

    synthetic_train = synthetic_train.cache(str(CACHE_PATH.joinpath("train")))
    synthetic_dataset_len = casia_dataset.get_dataset_size(synthetic_train)
    synthetic_train = (
        synthetic_train.shuffle(buffer_size=2_048)
        .batch(batch_size, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )
    synthetic_train = strategy.experimental_distribute_dataset(synthetic_train)

    synthetic_test = casia_dataset.get_test_dataset()
    synthetic_test = casia_dataset.normalize_dataset(synthetic_test)
    synthetic_test = synthetic_test.cache(str(CACHE_PATH.joinpath("test")))
    synthetic_test = (
        synthetic_test.shuffle(buffer_size=2_048)
        .batch(batch_size, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )
    synthetic_test = strategy.experimental_distribute_dataset(synthetic_test)

    synthetic_num_classes = casia_dataset.get_number_of_classes()

    return (
        synthetic_train,
        synthetic_test,
        synthetic_dataset_len,
        synthetic_num_classes,
    )


def _instantiate_learning_rate(learning_rate: float, learning_rate_decay_steps: int):
    return tf.keras.experimental.CosineDecay(learning_rate, learning_rate_decay_steps)


def _instantiate_models(
    strategy,
    synthetic_num_classes,
    network_settings,
    preprocess_settings,
):
    LOGGER.info(" -------- Creating Models --------")

    with strategy.scope():
        srfr_model = SRFR(
            num_filters=network_settings["num_filters"],
            depth=50,
            categories=network_settings["embedding_size"],
            num_gc=network_settings["gc"],
            num_blocks=network_settings["num_blocks"],
            residual_scailing=network_settings["residual_scailing"],
            training=True,
            input_shape=preprocess_settings["image_shape_low_resolution"],
            num_classes_syn=synthetic_num_classes,
        )
        discriminator_model = DiscriminatorNetwork()

    return srfr_model, discriminator_model


def _instantiate_optimizers(strategy, learning_rate, beta_1, train_settings):
    LOGGER.info(" -------- Creating Optimizers --------")

    with strategy.scope():
        srfr_optimizer = NovoGrad(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=train_settings["beta_2"],
            weight_decay=train_settings["weight_decay"],
            name="novograd_srfr",
        )
        srfr_optimizer = mixed_precision.LossScaleOptimizer(
            srfr_optimizer,
            loss_scale="dynamic",
        )
        discriminator_optimizer = NovoGrad(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=train_settings["beta_2"],
            weight_decay=train_settings["weight_decay"],
            name="novograd_discriminator",
        )
        discriminator_optimizer = mixed_precision.LossScaleOptimizer(
            discriminator_optimizer, loss_scale="dynamic"
        )

    return (
        srfr_optimizer,
        discriminator_optimizer,
    )


def _create_dimensions():
    return [
        Real(low=1.0e-3, high=7.0e-3, prior="log-uniform", name="learning_rate"),
        Integer(low=1_000, high=6_000, name="learning_rate_decay_steps"),
        Real(low=0.5, high=0.9, prior="log-uniform", name="beta_1"),
        Real(low=0.1, high=1.0, prior="log-uniform", name="face_recognition_weight"),
        Real(low=0.1, high=1.0, prior="log-uniform", name="super_resolution_weight"),
        Real(low=1.0e-3, high=1.0e-2, prior="log-uniform", name="perceptual_weight"),
        Real(low=1.0e-2, high=7.0e-2, prior="log-uniform", name="generator_weight"),
        Real(low=1.0e-2, high=9.0e-2, prior="log-uniform", name="l1_weight"),
    ]


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
    HP_FR_WEIGHT = hp.HParam("face_recognition_weight", hp.RealInterval(0.1, 1.0))
    HP_SR_WEIGHT = hp.HParam("super_resolution_weight", hp.RealInterval(0.1, 1.0))
    HP_PERCEPTUAL_WEIGHT = hp.HParam(
        "perceptual_weight", hp.RealInterval(1.0e-3, 1.0e-2)
    )
    HP_GENERATOR_WEIGHT = hp.HParam("generator_weight", hp.RealInterval(1.0e-2, 7.0e-2))
    HP_L1_WEIGHT = hp.HParam("l1_weight", hp.RealInterval(1.0e-2, 9.0e-2))

    return [
        HP_LEARNING_RATE,
        HP_LEARNING_RATE_DECAY_STEPS,
        HP_BETA_1,
        HP_FR_WEIGHT,
        HP_SR_WEIGHT,
        HP_PERCEPTUAL_WEIGHT,
        HP_GENERATOR_WEIGHT,
        HP_L1_WEIGHT,
    ]


if __name__ == "__main__":
    main()
