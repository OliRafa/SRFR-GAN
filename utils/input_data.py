import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm

# Checar se o filter de overlaps no loading do dataset esta funcionando corretamente
# Terminar checagem do get_lfw

AUTOTUNE = tf.data.experimental.AUTOTUNE


class InputData:
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

    # def _export_file(self, class_list, dataset_type):
    #    try:
    #        path = Path(os.path.join(
    #            os.getcwd(),
    #            'utils',
    #            'class_pairs',
    #            f'{self._dataset.name}'
    #        ))
    #        if not path.is_dir():
    #            path.mkdir(parents=True)
    #
    #        path = path.joinpath(f'{dataset_type}.txt')
    #        with path.open('a') as file_:
    #            for class_id, new_class_id in class_list:
    #                file_.write(f'{class_id},{new_class_id}\n')
    #    except IOError as ex:
    #        self._logger.warning(f" Class pairs for {self._dataset.name} couldn't be saved.\
    #            Exception: {ex}")

    # def _generate_class_pairs(
    #        self,
    #        dataset_path: str,
    #        concatenate: bool = False,
    #    ) -> None:
    #    self._logger.info(f' Generating class pairs for {self._dataset.name}.')
    #    dataset_path = os.path.join(dataset_path, 'Images')
    #    dataset_paths = list(
    #        os.path.join(dataset_path, folder) \
    #            for folder in next(os.walk(dataset_path))[1]
    #    )
    #    first = True
    #    for dataset in dataset_paths:
    #        paths_list = glob(os.path.join(dataset, '*'))
    #        class_list, count = self._populate_list(paths_list)
    #        self._export_file(class_list, Path(dataset).parts[-1])
    #        if concatenate:
    #            if first:
    #                self._export_file(class_list, 'concatenated')
    #                adding_parameter = count
    #                first = False
    #            else:
    #                old_classes, new_classes = zip(*class_list)
    #                new_classes = np.array(new_classes, dtype=np.int64)
    #                #new_classes += np.asarray(count, dtype=np.int64)
    #                new_classes += adding_parameter
    #                self._export_file(
    #                    list(zip(old_classes, new_classes)),
    #                    'concatenated',
    #                )

    def _get_class_pairs(self, dataset_name: str, file_name: str):
        self._logger.info(f" Getting class pairs.")

        path = (
            Path()
            .cwd()
            .joinpath("utils", "class_pairs", f"{dataset_name}", f"{file_name}.txt")
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
    def _convert_class_ids(self, *args):
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
    def _filter_overlaps(self, *args):
        if self._sample_ids:
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
        path = Path().cwd().joinpath("utils", "overlapping_identities", dataset_name)
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


class LFW(InputData):
    def __init__(self):
        super().__init__()
        self._number_of_classes = 5750
        self._serialized_features = {
            "class_id": tf.io.FixedLenFeature([], tf.string),
            "sample_id": tf.io.FixedLenFeature([], tf.string),
            "image_low_resolution": tf.io.FixedLenFeature([], tf.string),
        }
        self._dataset_settings = parseConfigsFile(["dataset"])["lfw_lr"]
        self.image_shape = parseConfigsFile(["preprocess"])[
            "image_shape_low_resolution"
        ]
        self._logger = super().get_logger()
        self._dataset = None

    # def get_dataset(self):
    #    self._logger.info(" Loading LFW_LR in test mode.")
    #    self._dataset = super().load_dataset(
    #        "LFW_LR",
    #        self._dataset_settings["path"],
    #        self._decoding_function,
    #        "train",
    #    )
    #    self._dataset = self.augment_dataset(self._dataset)
    #    self._dataset = self._dataset.map(
    #        lambda image, augmented_image, class_id, sample: (
    #            self.normalize_image(image),
    #            self.normalize_image(augmented_image),
    #            sample,
    #        ),
    #        num_parallel_calls=AUTOTUNE,
    #    )
    #    pairs = self.load_lfw_pairs()
    #    # with tf.device("/GPU:0"):
    #    (
    #        left_pairs,
    #        left_aug_pairs,
    #        right_pairs,
    #        right_aug_pairs,
    #        is_same_list,
    #    ) = self._generate_dataset(self._dataset, pairs)
    #    # with tf.device("/CPU:0"):
    #    left_pairs = tf.data.Dataset.from_tensor_slices(left_pairs)
    #    left_aug_pairs = tf.data.Dataset.from_tensor_slices(left_aug_pairs)
    #    right_pairs = tf.data.Dataset.from_tensor_slices(right_pairs)
    #    right_aug_pairs = tf.data.Dataset.from_tensor_slices(right_aug_pairs)
    #
    #    self._dataset = (
    #        left_pairs,
    #        left_aug_pairs,
    #        right_pairs,
    #        right_aug_pairs,
    #        is_same_list,
    #    )
    #    return self._dataset
    def get_dataset(self):
        self._logger.info(" Loading LFW_LR in test mode.")
        path = Path.cwd().joinpath("data", "datasets", "LFW")
        left_pairs = tf.data.Dataset.list_files(
            str(path.joinpath("images", "left", "*")), shuffle=False
        )
        left_pairs = left_pairs.map(
            self._decode_image_from_path, num_parallel_calls=AUTOTUNE
        )
        left_pairs = self.augment_dataset(left_pairs)
        left_pairs = left_pairs.map(
            lambda image, augmented_image: (
                self.normalize_image(image),
                self.normalize_image(augmented_image),
            ),
            num_parallel_calls=AUTOTUNE,
        )
        right_pairs = tf.data.Dataset.list_files(
            str(path.joinpath("images", "right", "*")), shuffle=False
        )
        right_pairs = right_pairs.map(
            self._decode_image_from_path, num_parallel_calls=AUTOTUNE
        )
        right_pairs = self.augment_dataset(right_pairs)
        right_pairs = right_pairs.map(
            lambda image, augmented_image: (
                self.normalize_image(image),
                self.normalize_image(augmented_image),
            ),
            num_parallel_calls=AUTOTUNE,
        )

        with path.joinpath("is_same_list.json").open("r") as obj:
            is_same_list = json.load(obj)

        is_same_list = np.array(is_same_list, dtype=np.int16).astype(np.bool)

        return left_pairs, right_pairs, is_same_list

    def _decode_image_from_path(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        return tf.image.resize(image, self.image_shape[:2])

    def get_number_of_classes(self):
        return self._number_of_classes

    def get_dataset_size(self):
        return super()._get_dataset_size(self._dataset)

    def get_dataset_shape(self):
        return "iis"

    @tf.function
    def _decoding_function(self, serialized_example):
        deserialized_example = tf.io.parse_single_example(
            serialized_example,
            self._serialized_features,
        )
        image_lr = super()._decode_raw_image(
            deserialized_example["image_low_resolution"]
        )
        image_lr = tf.reshape(
            image_lr,
            tf.stack(
                [*super().get_preprocess_settings()["image_shape_low_resolution"]]
            ),
        )
        class_id = super()._decode_string(deserialized_example["class_id"])
        sample_id = self._decode_string(deserialized_example["sample_id"])
        return image_lr, class_id, sample_id

    def augment_dataset(self, dataset):
        self._logger.info(" Augmenting LFW_LR dataset.")
        augmented_dataset = dataset.map(
            super()._augment_image_i,
            num_parallel_calls=AUTOTUNE,
        )
        dataset_zip = dataset.zip((dataset, augmented_dataset))
        # Reordering dataset to match the format
        # (image, augmented_image, class_id, sample_id)
        # return dataset_zip.map(
        #    lambda input_ds, augmented_ds: (
        #        input_ds[0],
        #        augmented_ds,
        #        input_ds[1],
        #        input_ds[2],
        #    ),
        #    num_parallel_calls=AUTOTUNE,
        # )
        return dataset_zip

    def _generate_dataset(self, dataset, pairs):
        is_same_list = []
        left_pairs: tf.data.Dataset = []
        left_aug_pairs: tf.data.Dataset = []
        right_pairs: tf.data.Dataset = []
        right_aug_pairs: tf.data.Dataset = []
        # i = 0
        for id_01, id_02, is_same in tqdm(pairs):
            left, left_augmented = next(
                iter(
                    dataset.filter(lambda x, y, z: tf.equal(z, id_01)).map(
                        lambda image, augmented_image, label: (image, augmented_image)
                    )
                )
            )
            right, right_augmented = next(
                iter(
                    dataset.filter(lambda x, y, z: tf.equal(z, id_02)).map(
                        lambda image, augmented_image, label: (image, augmented_image)
                    )
                )
            )
            left_pairs.append(left)
            left_aug_pairs.append(left_augmented)
            right_pairs.append(right)
            right_aug_pairs.append(right_augmented)
            is_same_list.append(is_same)

            # if i == 305:
            #    break
            # i += 1

        return (
            left_pairs,
            left_aug_pairs,
            right_pairs,
            right_aug_pairs,
            np.array(is_same_list, dtype=np.int16).astype(np.bool),
        )

    def load_lfw_pairs(self):
        """Loads the Labeled Faces in the Wild pairs from file.
        Output array has only the sample_id, not giving the full path or class_id.

        ### Returns
            Numpy Array of the pairs, with shape [id_01, id_2, is_same].
        """
        pairs = []
        path = Path().cwd().joinpath("validation", "pairs_label.txt")
        with path.open("r") as pairs_file:
            for line in pairs_file.readlines()[1:]:
                pair = line.strip().split()
                _, id_01 = InputData.split_path(pair[0])
                _, id_02 = InputData.split_path(pair[1])
                pairs.append([id_01, id_02, int(pair[2])])
        return np.array(pairs)


class VggFace2(InputData):
    def __init__(
        self,
        mode: str,
        remove_overlaps: bool = True,
        sample_ids: bool = False,
    ):
        super().__init__()
        self._mode = mode
        self._remove_overlaps = remove_overlaps
        self._sample_ids = sample_ids
        if self._sample_ids:
            self._dataset_shape = "iics"
        else:
            self._dataset_shape = "iic"

        if self._remove_overlaps:
            self._number_of_train_classes = 8069
            self._number_of_test_classes = 460
        else:
            self._number_of_train_classes = 8631
            self._number_of_test_classes = 500

        self._serialized_features = {
            "class_id": tf.io.FixedLenFeature([], tf.string),
            "sample_id": tf.io.FixedLenFeature([], tf.string),
            "image_low_resolution": tf.io.FixedLenFeature([], tf.string),
            "image_high_resolution": tf.io.FixedLenFeature([], tf.string),
        }
        self._dataset_settings = parseConfigsFile(["dataset"])["vggface2_lr"]
        self._dataset_paths = [
            self._dataset_settings["train_path"],
            self._dataset_settings["test_path"],
        ]
        self._logger = super().get_logger()
        self._class_pairs = super()._get_class_pairs("VGGFace2_LR", self._mode)
        super().set_class_pairs(self._class_pairs)
        self._dataset = None

    def get_dataset(self):
        self._logger.info(f" Loading VGGFace2_LR in {self._mode} mode.")
        self._dataset = super().load_dataset(
            "VGGFace2_LR",
            self._dataset_paths,
            self._decoding_function,
            self._mode,
            remove_overlaps=self._remove_overlaps,
            sample_ids=self._sample_ids,
        )
        self._dataset = self._dataset.map(
            super()._convert_class_ids,
            num_parallel_calls=AUTOTUNE,
        )
        return self._dataset

    def _get_number_of_classes(self):
        classes = set()
        for _, _, class_id in self._dataset:
            classes.add(class_id.numpy())

        num_classes = len(classes)
        print(num_classes)
        return num_classes

    def get_number_of_classes(self) -> Union[int, Tuple[int]]:
        #return 272
        return 27
        # if self._mode == "train":
        #    return self._number_of_train_classes
        # if self._mode == "test":
        #    return self._number_of_test_classes
        # if self._mode == "concatenated":
        #    return self._number_of_train_classes + self._number_of_test_classes
        # return self._number_of_train_classes, self._number_of_test_classes

    def get_dataset_size(self):
        return super()._get_dataset_size(self._dataset)

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
        if self._sample_ids:
            self._dataset_shape = "iics"
            sample_id = self._decode_string(deserialized_example["sample_id"])
            return image_lr, image_hr, class_id, sample_id

        self._dataset_shape = "iic"
        return image_lr, image_hr, class_id

    def augment_dataset(self):
        self._logger.info(" Augmenting VggFace2_LR dataset.")
        return super().augment_dataset(self._dataset, self.get_dataset_shape())

    def normalize_dataset(self):
        self._logger.info(" Normalizing VggFace2_LR dataset.")
        return self._dataset.map(
            lambda image_lr, image_hr, class_id: (
                self.normalize_image(image_lr),
                self.normalize_image(image_hr),
                class_id,
            ),
            num_parallel_calls=AUTOTUNE,
        )


def parseConfigsFile(
    settings_list: List[str] = None,
) -> Union[Dict, List[Dict]]:
    """Loads settings from config YAML file.

    ### Parameters
        settings_list: List of selected settings to return.

    ### Returns
        If settings_list were provided, returns the selected settings,\
 otherwise returns the full file settings.
    """
    path = Path().cwd()
    path = path.joinpath("config.yaml")
    with path.open("r") as yaml_filestream:
        settings = yaml.safe_load(yaml_filestream)
    if settings_list:
        returns = []
        for setting in settings_list:
            returns.append(settings[setting])
        return returns[0] if len(returns) == 1 else returns

    return settings


def load_json(path):
    with Path(path).open("r") as json_file:
        return json.load(json_file)


def load_resnet_config() -> List[Dict]:
    path = Path().cwd().joinpath("models", "resnet_config.txt")
    configs = load_json(path)
    return configs["network_config"], configs["layer_config"]
