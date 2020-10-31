import logging
import os
from functools import partial
from pathlib import Path
from typing import List, Union

import tensorflow as tf
from utils.input_data import parseConfigsFile

AUTOTUNE = tf.data.experimental.AUTOTUNE


class BaseRepository:
    """"""

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._preprocess_settigs = parseConfigsFile(["preprocess"])

    @staticmethod
    @tf.function
    def _convert_image(image):
        return tf.image.decode_jpeg(image, channels=3)

    @staticmethod
    @tf.function
    def normalize_image(image):
        """Normalizes a given image in format [0, 255] to (-1, 1) by subtracting
        127.5 and then dividing by 128.

        ### Parameters
            image: an image to be normalized.

        ### Returns
            the normalized image.
        """
        image = tf.subtract(tf.dtypes.cast(image, dtype=tf.float32), 127.5)
        return tf.divide(image, 128)

    @staticmethod
    @tf.function
    def _augment_image_iics(image_lr, image_hr, class_id, sample_id):
        """Flips the given image and adds '-augmented' at the end of the sample
        for data augmentation.

        ### Parameters
            image: image to be flipped.
            class_id: corresponding class for the image, passed through the function
            to be correctly concatenated with the original dataset.
            sample: sample name. Can be None in case the dataset has no sample
            attribute.

        ### Returns
            If the dataset has sample attribute, returns (augmented_image, class_id,
            augmented_sample): the flipped image, it's
            class and it's sample name.
            If the dataset has no sample attribute, returns (augmented_image,
            class_id) instead.
        """

        image_lr = tf.image.flip_left_right(image_lr)
        image_hr = tf.image.flip_left_right(image_hr)
        sample_id = tf.strings.join((sample_id, "augmented"), separator="-")

        return image_lr, image_hr, class_id, sample_id

    @staticmethod
    @tf.function
    def _augment_image_iic(image_lr, image_hr, class_id):
        """Flips the given image and adds '-augmented' at the end of the sample
        for data augmentation.

        ### Parameters
            image: image to be flipped.
            class_id: corresponding class for the image, passed through the function
            to be correctly concatenated with the original dataset.
            sample: sample name. Can be None in case the dataset has no sample
            attribute.

        ### Returns
            If the dataset has sample attribute, returns (augmented_image, class_id,
            augmented_sample): the flipped image, it's
            class and it's sample name.
            If the dataset has no sample attribute, returns (augmented_image,
            class_id) instead.
        """

        image_lr = tf.image.flip_left_right(image_lr)
        image_hr = tf.image.flip_left_right(image_hr)

        return image_lr, image_hr, class_id

    @staticmethod
    @tf.function
    def _augment_image_ics(image, class_id, sample_id):
        """Flips the given image and adds '-augmented' at the end of the sample
        for data augmentation.

        ### Parameters
            image: image to be flipped.
            class_id: corresponding class for the image, passed through the function
            to be correctly concatenated with the original dataset.
            sample: sample name. Can be None in case the dataset has no sample
            attribute.

        ### Returns
            If the dataset has sample attribute, returns (augmented_image, class_id,
            augmented_sample): the flipped image, it's
            class and it's sample name.
            If the dataset has no sample attribute, returns (augmented_image,
            class_id) instead.
        """

        image = tf.image.flip_left_right(image)
        sample_id = tf.strings.join((sample_id, "augmented"), separator="-")

        return image, class_id, sample_id

    @staticmethod
    @tf.function
    def _augment_image_ic(image, class_id):
        """Flips the given image and adds '-augmented' at the end of the sample
        for data augmentation.

        ### Parameters
            image: image to be flipped.
            class_id: corresponding class for the image, passed through the function
            to be correctly concatenated with the original dataset.
            sample: sample name. Can be None in case the dataset has no sample
            attribute.

        ### Returns
            If the dataset has sample attribute, returns (augmented_image, class_id,
            augmented_sample): the flipped image, it's
            class and it's sample name.
            If the dataset has no sample attribute, returns (augmented_image,
            class_id) instead.
        """
        return tf.image.flip_left_right(image), class_id

    @staticmethod
    @tf.function
    def _augment_image_i(image):
        """Flips the given image and adds '-augmented' at the end of the sample
        for data augmentation.

        ### Parameters
            image: image to be flipped.
            class_id: corresponding class for the image, passed through the function
            to be correctly concatenated with the original dataset.
            sample: sample name. Can be None in case the dataset has no sample
            attribute.

        ### Returns
            If the dataset has sample attribute, returns (augmented_image, class_id,
            augmented_sample): the flipped image, it's
            class and it's sample name.
            If the dataset has no sample attribute, returns (augmented_image,
            class_id) instead.
        """
        return tf.image.flip_left_right(image)

    def augment_dataset(self, dataset, dataset_shape):
        if dataset_shape == "iics":
            augmented_dataset = dataset.map(
                self._augment_image_iics,
                num_parallel_calls=AUTOTUNE,
            )
        elif dataset_shape == "iic":
            augmented_dataset = dataset.map(
                self._augment_image_iic,
                num_parallel_calls=AUTOTUNE,
            )
        elif dataset_shape == "ics":
            augmented_dataset = dataset.map(
                self._augment_image_ics,
                num_parallel_calls=AUTOTUNE,
            )
        else:
            augmented_dataset = dataset.map(
                self._augment_image_ic,
                num_parallel_calls=AUTOTUNE,
            )

        return dataset.concatenate(augmented_dataset)

    @staticmethod
    def split_path(file_path):
        parts = tf.strings.split(file_path, os.path.sep)

        class_id = parts.numpy()[-2].decode("utf-8")
        sample_id = parts.numpy()[-1].decode("utf-8").split(".")[0]

        return class_id, sample_id

    def _populate_list(self, paths_list: List[str]):
        class_list = []
        for class_id, new_class_id in zip(paths_list, range(len(paths_list))):
            if class_id not in class_list:
                class_list.append([Path(class_id).parts[-1], new_class_id])

        return class_list, new_class_id

    def _get_class_pairs(self, dataset_name: str, file_name: str):
        self._logger.info(f" Getting class pairs.")

        path = Path.cwd().joinpath(
            "data", "class_pairs", f"{dataset_name}", f"{file_name}.txt"
        )
        if not path.is_file():
            self._logger.warning(f" File not found for {dataset_name}.")
            return None

        return tf.lookup.StaticHashTable(
            tf.lookup.TextFileInitializer(
                str(path), tf.string, 0, tf.int32, 1, delimiter=","
            ),
            -1,
        )

    @staticmethod
    @tf.function
    def _decode_raw_image(image):
        return tf.io.decode_png(image)

    @staticmethod
    @tf.function
    def _decode_image_shape(height, width, depth):
        height = tf.cast(height, tf.int32)
        width = tf.cast(width, tf.int32)
        depth = tf.cast(depth, tf.int8)

        return [height, width, depth]

    @staticmethod
    @tf.function
    def _decode_string(raw_bytes):
        return tf.cast(raw_bytes, tf.string)

    def set_class_pairs(self, class_pairs):
        self._class_pairs = class_pairs

    @tf.function
    def _convert_class_ids_with_sample_id(self, *args):
        if self._sample_ids:
            class_id = args[-2]
        else:
            class_id = args[-1]
        class_id = self._class_pairs.lookup(class_id)

        args = list(args)
        if self._sample_ids:
            args[-2] = class_id
        else:
            args[-1] = class_id

        return args

    @tf.function
    def _convert_class_ids(self, *args):
        class_id = args[-1]
        class_id = self._class_pairs.lookup(class_id)

        args = list(args)
        args[-1] = class_id

        return args

    @tf.function
    def _filter_overlaps(self, *args):
        if getattr(self, "_sample_ids", False):
            class_id = args[-2]
        else:
            class_id = args[-1]

        # When class_id is in self._overlaps, returns False so that ds.filter()
        # will filter this id out.
        return tf.cond(
            tf.math.reduce_any(tf.math.equal(class_id, self._overlaps)),
            lambda: False,
            lambda: True,
        )

    def _load_from_tfrecords(
        self,
        dataset_paths: Union[str, List[str]],
        decoding_function,
    ):
        self._logger.info(f" Loading from {dataset_paths}.")
        dataset = tf.data.TFRecordDataset(dataset_paths)
        dataset = dataset.map(
            decoding_function,
            num_parallel_calls=AUTOTUNE,
        )
        if self._overlaps:
            dataset = dataset.filter(self._filter_overlaps)

        return dataset

    def _get_overlapping_identities(self, dataset_name: str) -> tuple:
        """Loads overlapping identities files for a given dataset.

        ### Parameters
            dataset_name: Name of the dataset.

        ### Returns
            Tuple with the identities to be cleaned.
        """
        overlapping = set()
        path = Path.cwd().joinpath("data", "overlapping_identities", dataset_name)
        for file_path in path.glob("*"):
            with file_path.open("r") as _file:
                for line in _file.readlines():
                    overlapping.add(line.replace("\n", ""))

        return tuple(overlapping)

    def _get_dataset_size(
        self,
        dataset,
    ) -> int:
        """Gets the size of a given dataset.

        ### Parameters
            file_path: File path for the dataset.

        ### Returns
            Number of samples.
        """
        # Sum number of samples in the dataset
        return sum(1 for _ in dataset)

    def load_dataset_multiple_shards(
        self,
        dataset_name: str,
        dataset_paths: List[str],
        decoding_function,
        remove_overlaps: bool = False,
    ):
        _load_tfrecords = partial(
            self._load_from_tfrecords, decoding_function=decoding_function
        )
        self._overlaps = (
            self._get_overlapping_identities(dataset_name)
            if remove_overlaps
            else remove_overlaps
        )

        paths = tf.io.matching_files(f"{str(dataset_paths)}/*")
        paths = tf.random.shuffle(paths)
        paths = tf.data.Dataset.from_tensor_slices(paths)
        paths = paths.interleave(
            _load_tfrecords, deterministic=False, num_parallel_calls=AUTOTUNE
        )
        return paths.shuffle(buffer_size=9000)

    def load_dataset(
        self,
        dataset_name: str,
        dataset_paths: Union[str, List[str]],
        decoding_function,
        mode: str,
        remove_overlaps: bool = False,
        sample_ids: bool = False,
    ):
        """Loads the dataset from disk, returning a TF Tensor with shape\
     (image, class_id, sample).

        Viable datasets: (
            'VGGFace2',
            'VGGFace2_LR',
            'LFW',
            'LFW_LR',
        )

        ### Arguments
            dataset_name: Dataset to be loaded.
            mode: One of 'train', 'test', 'both', 'concatenated'. If 'both', it\
     will return train and test datasets separately, unlike 'concatenated' that\
     will return both in a single concatenated dataset.
            remove_overlap: If True, overlapping identities between the given\
     dataset and others (Validation Datasets) will be removed.
            sample_ids: If True, return a Tensor containing the sample_id for each\
     sample in the dataset. Necessary in case of loading a test dataset that will\
     be augmented.

        ### Returns
            If mode='both', returns two tuples, one for train and one for test, with\
     (train_dataset, num_train_classes, dataset_length) - dataset Tensor, number of\
     classes, and dataset length - in the shape of (train, test).
            Otherwise, returns only one tuple (train_dataset, num_train_classes,\
     dataset_length) - dataset Tensor, number of classes and dataset length.
        """
        self._sample_ids = sample_ids
        self._overlaps = (
            self._get_overlapping_identities(dataset_name)
            if remove_overlaps
            else remove_overlaps
        )

        if mode == "both":
            train_dataset = self._load_from_tfrecords(
                dataset_paths[0],
                decoding_function,
            )
            test_dataset = self._load_from_tfrecords(
                dataset_paths[1],
                decoding_function,
            )
            return train_dataset, test_dataset

        return self._load_from_tfrecords(
            dataset_paths,
            decoding_function,
        )

    def get_preprocess_settings(self):
        return self._preprocess_settigs

    def get_logger(self):
        return self._logger
