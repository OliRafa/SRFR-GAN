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

import logging

from use_cases.train_model_use_case import TrainModelUseCase
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
    strategy = tf.distribute.MirroredStrategy()
    # strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    train_model_use_case = TrainModelUseCase(
        strategy,
        TimingLogger(),
        LOGGER,
    )

    train_model_use_case.execute()


if __name__ == "__main__":
    main()
