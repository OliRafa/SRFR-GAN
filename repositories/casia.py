from pathlib import Path

import tensorflow as tf

from repositories.repository import BaseRepository

AUTOTUNE = tf.data.experimental.AUTOTUNE


class CasiaWebface(BaseRepository):
    DATASET_PATH = Path.cwd().joinpath("data", "datasets", "CASIA_LR_TFRecords")

    def __init__(
        self,
        BASE_CACHE_PATH: Path,
        remove_overlaps: bool = True,
    ):
        super().__init__()
        self._remove_overlaps = remove_overlaps

        self._dataset_shape = "iic"
        self._serialized_features = {
            "class_id": tf.io.FixedLenFeature([], tf.string),
            "image_low_resolution": tf.io.FixedLenFeature([], tf.string),
            "image_high_resolution": tf.io.FixedLenFeature([], tf.string),
        }

        self._logger = super().get_logger()
        self._class_pairs = super()._get_class_pairs("CASIA", "concatenated")
        super().set_class_pairs(self._class_pairs)

        self._dataset = self._initialize_dataset()
        self._dataset_size = 446_883

    def get_train_dataset(self):
        self._logger.info(f" Loading CASIA-Webface in train mode.")

        return self._dataset.take(int(0.9 * self._dataset_size))

    def get_train_dataset_len(self) -> int:
        if self._remove_overlaps:
            return 804_384
        return 0

    def get_test_dataset_len(self) -> int:
        if self._remove_overlaps:
            return 44_672
        return 0

    def get_test_dataset(self):
        self._logger.info(f" Loading CASIA-Webface in test mode.")

        return self._dataset.skip(int(0.9 * self._dataset_size))

    def _initialize_dataset(self):
        self._logger.info(f" Loading CASIA-Webface in concatenated mode.")

        dataset = super().load_dataset_multiple_shards(
            "CASIA",
            self.DATASET_PATH,
            self._decoding_function,
            self._remove_overlaps,
        )

        return self._convert_tfrecords(dataset)

    def _convert_tfrecords(self, dataset):
        return dataset.map(
            super()._convert_class_ids,
            num_parallel_calls=AUTOTUNE,
        )

    def _get_number_of_classes(self):
        classes = set()
        for _, _, class_id in self._dataset:
            classes.add(class_id.numpy())

        num_classes = len(classes)
        print(num_classes)
        return num_classes

    def get_full_dataset(self):
        return self._dataset

    def get_full_dataset_len(self) -> int:
        return self._dataset_size

    def get_number_of_classes(self) -> int:
        return 10_574

    def get_dataset_size(self, dataset):
        return super()._get_dataset_size(dataset)

    def get_dataset_shape(self):
        return self._dataset_shape

    @tf.function
    def _decoding_function(self, serialized_example):
        deserialized_example = tf.io.parse_single_example(
            serialized_example,
            self._serialized_features,
        )
        image_lr = super()._decode_raw_image(
            deserialized_example["image_low_resolution"]
        )
        # image_lr.set_shape(image_shape)
        image_lr = tf.reshape(
            image_lr,
            tf.stack(
                [*super().get_preprocess_settings()["image_shape_low_resolution"]]
            ),
        )
        image_hr = super()._decode_raw_image(
            deserialized_example["image_high_resolution"]
        )
        image_hr = tf.reshape(
            image_hr,
            tf.stack(
                [*super().get_preprocess_settings()["image_shape_high_resolution"]]
            ),
        )

        class_id = super()._decode_string(deserialized_example["class_id"])
        self._dataset_shape = "iic"
        return image_lr, image_hr, class_id

    def augment_dataset(self, dataset):
        self._logger.info(" Augmenting CASIA-Webface dataset.")
        return super().augment_dataset(dataset, self.get_dataset_shape())

    def normalize_dataset(self, dataset):
        self._logger.info(" Normalizing CASIA-Webface dataset.")
        return dataset.map(
            lambda image_lr, image_hr, class_id: (
                self.normalize_image(image_lr),
                self.normalize_image(image_hr),
                class_id,
            ),
            num_parallel_calls=AUTOTUNE,
        )
