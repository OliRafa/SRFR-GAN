"""Main training module for the Joint Learning Super Resolution Face\
 Recognition.
"""
import tensorflow as tf  # isort:skip

gpus = tf.config.experimental.list_physical_devices("GPU")  # isort:skip
if gpus:  # isort:skip
    try:  # isort:skip
        for gpu in gpus:  # isort:skip
            tf.config.experimental.set_memory_growth(gpu, True)  # isort:skip
        print("set_memory_growth ok!")  # isort:skip
    except RuntimeError as e:  # isort:skip
        print("set_memory_growth failed!")  # isort:skip
        print(str(e))  # isort:skip

import datetime
import logging
from pathlib import Path

from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_addons.optimizers import NovoGrad

from models.discriminator import DiscriminatorNetwork
from models.srfr import SRFR
from services.losses import Loss
from services.train import Train, generate_num_epochs
from use_cases.train_model_use_case import TrainModelUseCase
from utils.input_data import VggFace2, parseConfigsFile
from utils.timing import TimingLogger

# Importar Natural DS.

logging.basicConfig(
    filename="train_logs.txt",
    level=logging.DEBUG,
)
LOGGER = logging.getLogger(__name__)
AUTOTUNE = tf.data.experimental.AUTOTUNE


def main():
    """Main training function."""
    timing = TimingLogger()
    timing.start()
    network_settings, train_settings, preprocess_settings = parseConfigsFile(
        ["network", "train", "preprocess"]
    )

    strategy = tf.distribute.MirroredStrategy()
    # strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    BATCH_SIZE = train_settings["batch_size"] * strategy.num_replicas_in_sync
    # BATCH_SIZE = train_settings["batch_size"] * 2
    temp_folder = Path.cwd().joinpath(
        "data", "temp", "train_dataset", "synthetic_dataset"
    )

    LOGGER.info(" -------- Importing Datasets --------")

    vgg_dataset = VggFace2(mode="concatenated")
    synthetic_dataset = vgg_dataset.get_dataset()
    synthetic_dataset = vgg_dataset.augment_dataset()
    synthetic_dataset = vgg_dataset.normalize_dataset()
    synthetic_dataset = synthetic_dataset.cache(str(temp_folder))
    # synthetic_dataset_len = vgg_dataset.get_dataset_size()
    synthetic_dataset_len = 20_000
    synthetic_num_classes = vgg_dataset.get_number_of_classes()
    synthetic_dataset = (
        synthetic_dataset.shuffle(buffer_size=2_048).repeat().batch(BATCH_SIZE)
    )

    # Using `distribute_dataset` to distribute the batches across the GPUs
    synthetic_dataset = strategy.experimental_distribute_dataset(synthetic_dataset)

    LOGGER.info(" -------- Creating Models and Optimizers --------")

    EPOCHS = generate_num_epochs(
        train_settings["iterations"],
        synthetic_dataset_len,
        BATCH_SIZE,
    )

    (
        srfr_model,
        discriminator_model,
        srfr_optimizer,
        discriminator_optimizer,
    ) = _instantiate_models(
        strategy,
        network_settings,
        train_settings,
        preprocess_settings,
        synthetic_num_classes,
    )

    checkpoint, manager = _create_checkpoint_and_manager(
        srfr_model, discriminator_model, srfr_optimizer, discriminator_optimizer
    )

    train_summary_writer = _create_summary_writer(strategy)

    loss = Loss(
        BATCH_SIZE,
        train_summary_writer,
        train_settings["super_resolution_weight"],
        train_settings["scale"],
        train_settings["angular_margin"],
    )
    train_model_use_case = TrainModelUseCase(
        strategy,
        loss,
        train_summary_writer,
        TimingLogger(),
        LOGGER,
        checkpoint,
        manager,
    )

    LOGGER.info(" -------- Starting Training --------")

    train_model_use_case.execute(
        srfr_model,
        srfr_optimizer,
        discriminator_model,
        discriminator_optimizer,
        synthetic_dataset,
        synthetic_num_classes,
        BATCH_SIZE,
        EPOCHS,
    )


def _instantiate_models(
    strategy,
    network_settings,
    train_settings,
    preprocess_settings,
    synthetic_num_classes,
):
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

        srfr_optimizer = NovoGrad(
            learning_rate=train_settings["learning_rate"],
            beta_1=train_settings["momentum"],
            beta_2=train_settings["beta_2"],
            weight_decay=train_settings["weight_decay"],
            name="novograd_srfr",
        )
        srfr_optimizer = mixed_precision.LossScaleOptimizer(
            srfr_optimizer,
            loss_scale="dynamic",
        )
        discriminator_optimizer = NovoGrad(
            learning_rate=train_settings["learning_rate"],
            beta_1=train_settings["momentum"],
            beta_2=train_settings["beta_2"],
            weight_decay=train_settings["weight_decay"],
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


def _create_summary_writer(strategy):
    with strategy.scope():
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return tf.summary.create_file_writer(
            str(
                Path.cwd().joinpath(
                    "data", "logs", "test", "gradient_tape", current_time, "train"
                )
            ),
        )


if __name__ == "__main__":
    main()
