import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from utils.input_data import parseConfigsFile

from repositories.repository import BaseRepository

AUTOTUNE = tf.data.experimental.AUTOTUNE


class LFW(BaseRepository):
    DATASET_PATH_LR = Path.cwd().joinpath("data", "datasets", "LFW_LR")
    DATASET_PATH_HR = Path.cwd().joinpath("data", "datasets", "LFW_HR")

    def __init__(self, resolution: str = "lr"):
        super().__init__()
        if resolution == "lr":
            self.DATASET_PATH = self.DATASET_PATH_LR
            self.image_shape = parseConfigsFile(["preprocess"])[
                "image_shape_low_resolution"
            ]

        else:
            self.DATASET_PATH = self.DATASET_PATH_HR
            self.image_shape = parseConfigsFile(["preprocess"])[
                "image_shape_high_resolution"
            ]

        self.image_shape = self.image_shape[:2]

        self._number_of_classes = 5750
        self._logger = super().get_logger()

    def get_dataset(self):
        self._logger.info(" Loading LFW_LR in test mode.")

        left_pairs = self._get_dataset_pair(
            self.DATASET_PATH.joinpath("images", "left", "*")
        )
        right_pairs = self._get_dataset_pair(
            self.DATASET_PATH.joinpath("images", "right", "*")
        )

        with self.DATASET_PATH.joinpath("is_same_list.json").open("r") as obj:
            is_same_list = json.load(obj)

        is_same_list = np.array(is_same_list, dtype=np.int16).astype(np.bool)

        return left_pairs, right_pairs, is_same_list

    def _get_dataset_pair(self, path: Path):
        dataset = tf.data.Dataset.list_files(str(path), shuffle=False)
        dataset = dataset.map(self._decode_image_from_path, num_parallel_calls=AUTOTUNE)
        dataset = self.augment_dataset(dataset)
        return dataset.map(
            lambda image, augmented_image: (
                self.normalize_image(image),
                self.normalize_image(augmented_image),
            ),
            num_parallel_calls=AUTOTUNE,
        )

    def _decode_image_from_path(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        return tf.image.resize(image, self.image_shape)

    def get_number_of_classes(self) -> int:
        return self._number_of_classes

    def get_dataset_size(self, dataset) -> int:
        return super()._get_dataset_size(dataset)

    def get_dataset_shape(self):
        return "iis"

    def augment_dataset(self, dataset):
        self._logger.info(" Augmenting LFW_LR dataset.")
        augmented_dataset = dataset.map(
            super()._augment_image_i,
            num_parallel_calls=AUTOTUNE,
        )
        return dataset.zip((dataset, augmented_dataset))
