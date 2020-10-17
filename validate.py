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
        print(str(e))  # isort:skipf

import datetime
import logging
from pathlib import Path

from models.srfr import SRFR
from use_cases.validate_model_use_case import ValidateModelUseCase
from utils.input_data import VggFace2, parseConfigsFile
from utils.timing import TimingLogger

logging.basicConfig(
    filename="train_logs.txt",
    level=logging.DEBUG,
)
LOGGER = logging.getLogger(__name__)
AUTOTUNE = tf.data.experimental.AUTOTUNE


def main():
    """Main valiation function."""
    timing = TimingLogger()
    timing.start()
    network_settings, train_settings, preprocess_settings = parseConfigsFile(
        ["network", "train", "preprocess"]
    )

    strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE = train_settings["batch_size"] * strategy.num_replicas_in_sync

    LOGGER.info(" -------- Importing Datasets --------")

    vgg_dataset = VggFace2(mode="concatenated")
    synthetic_num_classes = vgg_dataset.get_number_of_classes()
    # synthetic_num_classes = 8529

    LOGGER.info(" -------- Creating Models and Optimizers --------")

    srfr_model = _instantiate_models(
        strategy, network_settings, preprocess_settings, synthetic_num_classes
    )

    checkpoint, manager = _create_checkpoint_and_manager(srfr_model)

    test_summary_writer = _create_summary_writer()

    LOGGER.info(" -------- Starting Validation --------")
    with strategy.scope():
        validate_model_use_case = ValidateModelUseCase(
            strategy, test_summary_writer, TimingLogger(), LOGGER
        )

        for model_checkpoint in manager.checkpoints:
            checkpoint.restore(model_checkpoint)
            LOGGER.info(f" Restored from {model_checkpoint}")

            validate_model_use_case.execute(srfr_model, BATCH_SIZE, checkpoint)


def _instantiate_models(
    strategy, network_settings, preprocess_settings, synthetic_num_classes
):
    with strategy.scope():
        return SRFR(
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


def _create_checkpoint_and_manager(srfr_model):
    checkpoint = tf.train.Checkpoint(
        epoch=tf.Variable(1),
        step=tf.Variable(1, dtype=tf.int64),
        srfr_model=srfr_model,
    )
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=str(Path.cwd().joinpath("data", "training_checkpoints")),
        max_to_keep=5,
    )
    return checkpoint, manager


def _create_summary_writer():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.summary.create_file_writer(
        str(Path.cwd().joinpath("data", "logs", "gradient_tape", current_time, "test")),
    )


if __name__ == "__main__":
    main()
