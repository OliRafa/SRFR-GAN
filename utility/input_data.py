import os
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import tensorflow as tf
from collections import namedtuple
from glob import glob
from functools import partial

from glob import iglob

# Checar se o filter de overlaps no loading do dataset esta funcionando corretamente

AUTOTUNE = tf.data.experimental.AUTOTUNE

@tf.function
def _convert_image(image):
    return tf.image.decode_jpeg(image, channels=3)

@tf.function
def normalize_images(image):
    """Normalizes a given image in format [0, 255] to (-1, 1) by subtracting
    127.5 and then dividing by 128.

    ### Parameters
        image: an image to be normalized.

    ### Returns
        the normalized image.
    """
    image = tf.subtract(tf.dtypes.cast(image, dtype=tf.float32), 127.5)
    image = tf.divide(image, 128)

    return image

@tf.function
def _augment_image(image, class_id, sample=None):
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
    augmented_image = tf.image.flip_left_right(image)

    if sample:
        augmented_sample = tf.strings.join((sample, 'augmented'), separator='-')
        return augmented_image, class_id, augmented_sample

    return augmented_image, class_id

@tf.function
def augment_dataset(dataset):
    augmented_dataset = dataset.map(_augment_image, num_parallel_calls=AUTOTUNE)

    return dataset.concatenate(augmented_dataset)

@tf.function
def augment_dataset_without_concatenation(dataset):
    augmented_dataset = dataset.map(_augment_image, num_parallel_calls=AUTOTUNE)
    augmented_dataset = augmented_dataset.map(lambda x: x[0])

    dataset_zip = dataset.zip((dataset, augmented_dataset))
    # Reordering dataset to match the format
    # (image, augmented_image, class_id, sample_id)
    return dataset_zip.map(lambda x, y: (x[0], y[0], x[1], x[2]))

@tf.function
def split_path(file_path):
    parts = tf.strings.split(file_path, os.path.sep)

    class_id = parts.numpy()[-2].decode('utf-8')
    sample_id = parts.numpy()[-1].decode('utf-8').split('.')[0]

    return class_id, sample_id

@tf.function
def _read_image_and_labels(file_path: str, sample_ids: bool):
    image = tf.io.read_file(file_path)
    image = _convert_image(image)
    class_id, sample = split_path(file_path)

    return image, class_id, sample if sample_ids else image, class_id

def _populate_list(paths_list: List[str]):
    class_list = []
    count = 0
    for class_id in paths_list:
        if class_id not in class_list:
            class_list.append([Path(class_id).parts[-1], count])
            count += 1
    return class_list, count - 1

def _export_file(class_list, dataset_name, dataset_type):
    with open(os.path.join('class_pairs',
                           f'{dataset_name}_{dataset_type}.txt'), 'a') as file_:
        for class_id, new_class_id in class_list:
            file_.write(f'{class_id},{new_class_id}\n')

def _generate_class_pairs(
        dataset_path: str,
        dataset_name: str,
        concatenate: bool = False,
    ) -> None:
    dataset_path = os.path.join(dataset_path, 'Images')
    dataset_paths = list(
        os.path.join(dataset_path, folder) \
            for folder in next(os.walk(dataset_path))[1]
    )
    first = True
    for dataset in dataset_paths:
        paths_list = glob(os.path.join(dataset, '*'))
        class_list, count = _populate_list(paths_list)
        _export_file(class_list, dataset_name, Path(dataset).parts[-1])
        if concatenate:
            if first:
                _export_file(class_list, dataset_name, 'concatenated')
                first = False
            else:
                old_classes, new_classes = zip(*class_list)
                new_classes = np.array(new_classes, dtype=np.int64)
                #new_classes += np.asarray(count, dtype=np.int64)
                new_classes += count
                _export_file(
                    list(zip(old_classes, new_classes)),
                    dataset_name,
                    'concatenated',
                )

def _get_class_pairs(dataset_name: str) -> Any[Dict[str, int], None]:
    try:
        pairs = {}
        with open(f'class_pairs/{dataset_name}.txt', 'r') as class_pairs:
            for line in class_pairs.readlines():
                old_class, new_class = line.split(',')
                pairs[old_class] = new_class
        return pairs
    except OSError:
        return None

@tf.function
def _decode_raw_image(image):
    return tf.io.decode_raw(image, tf.uint8)

@tf.function
def _decode_image_shape(height, width, depth):
    height = tf.cast(height, tf.int32)
    width = tf.cast(width, tf.int32)
    depth = tf.cast(depth, tf.int8)

    return [height, width, depth]

@tf.function
def _decode_string(raw_bytes):
    return tf.cast(raw_bytes, tf.string)

@tf.function
def _decode_lfw_lr(
        serialized_example,
        sample_ids: bool,
    ):
    features = {
        'height_lr': tf.io.FixedLenFeature([], tf.int64),
        'width_lr': tf.io.FixedLenFeature([], tf.int64),
        'depth_lr': tf.io.FixedLenFeature([], tf.int64),
        'height_hr': tf.io.FixedLenFeature([], tf.int64),
        'width_hr': tf.io.FixedLenFeature([], tf.int64),
        'depth_hr': tf.io.FixedLenFeature([], tf.int64),
        'class_id': tf.io.FixedLenFeature([], tf.string),
        'sample_id': tf.io.FixedLenFeature([], tf.string),
        'image_low_resolution': tf.io.FixedLenFeature([], tf.string),
        'image_high_resolution': tf.io.FixedLenFeature([], tf.string),
    }

    deserialized_example = tf.io.parse_single_example(
        serialized_example,
        features,
    )
    image_shape = _decode_image_shape(
        deserialized_example['height_lr'],
        deserialized_example['width_lr'],
        deserialized_example['depth_lr'],
    )
    image_lr = _decode_raw_image(deserialized_example['image_low_resolution'])
    image_lr.set_shape(image_shape)
    image_shape = _decode_image_shape(
        deserialized_example['height_hr'],
        deserialized_example['width_hr'],
        deserialized_example['depth_hr'],
    )
    image_hr = _decode_raw_image(deserialized_example['image_high_resolution'])
    image_hr.set_shape(image_shape)

    class_id = _decode_string(deserialized_example['class_id'])
    if sample_ids:
        sample_id = _decode_string(deserialized_example['sample_id'])
        return image_lr, image_hr, class_id, sample_id

    return image_lr, image_hr, class_id

@tf.function
def _convert_class_ids(class_id, class_pairs: Dict[str, int]):
    for old_class, new_class in class_pairs.items():
        if tf.math.equal(class_id, old_class):
            class_id = tf.convert_to_tensor(new_class, tf.string)
    return class_id

@tf.function
def _load_from_tfrecords(
        dataset_paths: List[str],
        dataset_name: str,
        decode_function: Callable,
        overlaps: tuple = False,
    ):
    dataset = tf.data.TFRecordDataset(dataset_paths)
    dataset = dataset.map(decode_function, num_parallel_calls=AUTOTUNE)
    if overlaps:
        dataset = dataset.filter(lambda x: x not in overlaps)

    pairs = _get_class_pairs(dataset_name)
    if not pairs:
        _generate_class_pairs(Path(dataset_paths).parents[2], dataset_name)
        pairs = _get_class_pairs(dataset_name)

    _convert_classes = partial(
        _convert_class_ids,
        class_pairs=pairs,
    )
    return dataset.map(_convert_classes, num_parallel_calls=AUTOTUNE)

@tf.function
def _load_dataset_from_path(
        dataset_path: str,
        overlaps: tuple = False,
        sample_ids: bool = False,
    ):
    dataset = tf.data.Dataset.list_files(dataset_path)
    if overlaps:
        dataset = dataset.filter(lambda x: x not in overlaps)

    _read_images = partial(
        _read_image_and_labels,
        sample_ids=sample_ids
    )
    dataset = dataset.map(_read_images, num_parallel_calls=AUTOTUNE)

    return dataset

def _get_overlapping_identities(dataset_name: str) -> tuple:
    """Loads overlapping identities files for a given dataset.

    ### Parameters
        dataset_name: Name of the dataset.

    ### Returns
        Tuple with the identities to be cleaned.
    """
    overlapping = ()
    path = os.path.join(
        os.getcwd(), 'overlapping_identities',
        dataset_name,
        '*.txt',
    )
    for file_ in glob(path):
        with open(file_, 'r') as f:
            overlapping += tuple(f.readlines())

    return overlapping

def _get_number_of_classes(
        file_path: str,
        overlaps: tuple = False
    ) -> int:
    """Gets the number of classes for a given dataset.

    ### Parameters
        file_path: File path for the dataset.
        overlaps: List of the overlapping identities.

    ### Returns
        Number of classes.
    """
    file_path = file_path[:-1]
    file_path = iglob(file_path)
    if overlaps:
        file_path = filter(lambda x: x not in overlaps, file_path)

    # Sum number of classes in the file_path generator
    return sum(1 for _ in file_path)

def _get_dataset_size(
        file_path: str,
        overlaps: tuple = False
    ) -> int:
    """Gets the size of a given dataset.

    ### Parameters
        file_path: File path for the dataset.
        overlaps: List of the overlapping identities.

    ### Returns
        Number of samples.
    """
    file_path = iglob(file_path)
    if overlaps:
        file_path = filter(lambda x: x not in overlaps, file_path)

    # Sum number of samples in the file_path generator
    return sum(1 for _ in file_path)

def load_dataset(
        dataset_name: str,
        test: bool = True,
        concatenate: bool = False,
        remove_overlaps: bool = True,
        sample_ids: bool = False
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
        test: If true, returns train and test datasets, otherwise only returns\
 the training one.
        concatenate: If True, train and test datasets will be concatenated in a\
 single dataset
        remove_overlap: If True, overlapping identities between the given\
 dataset and others (Validation Datasets) will be removed.
        sample_ids: If True, return a Tensor containing the sample_id for each\
 sample in the dataset. Necessary in case of loading a test dataset that will\
 be augmented.

    ### Returns
        If test=True, returns two tuples, one for Train and one for Test, with\
 (train_dataset, num_train_classes, dataset_length) - dataset Tensor, number of\
 classes, and dataset length - in the shape of (train, test).
        If test=False or concatenate=True, returns only one tuple\
 (train_dataset, num_train_classes, dataset_length) Tensor, number of classes\
 and dataset length.
    """
    Dataset = namedtuple(
        'Dataset',
        (
            'name',
            'train',
            'test',
            'decode_function',
        )
    )
    dataset_paths = (
        Dataset(
            name='VGGFace2',
            train='/mnt/hdd_raid/datasets/VGGFace2/train/*/*',
            test='/mnt/hdd_raid/datasets/VGGFace2/test/*/*',
        ),
        Dataset(
            name='VGGFace2_LR',
            train='/mnt/hdd_raid/datasets/VGGFace2/train/*/*',
            test='/mnt/hdd_raid/datasets/VGGFace2/test/*/*',
            decode_function=_decode_lfw_lr,
        ),
        Dataset(
            name='LFW',
            train='/mnt/hdd_raid/datasets/VGGFace2/train/*/*',
            test='/mnt/hdd_raid/datasets/VGGFace2/test/*/*',
        ),
        Dataset(
            name='LFW_LR',
            train='/mnt/hdd_raid/datasets/VGGFace2/train/*/*',
            test='/mnt/hdd_raid/datasets/VGGFace2/test/*/*',
            features = {
                'height': tf.io.FixedLenFeature([], tf.int64),
                'width': tf.io.FixedLenFeature([], tf.int64),
                'depth': tf.io.FixedLenFeature([], tf.int64),
                'class_id': tf.io.FixedLenFeature([], tf.string),
                'sample': tf.io.FixedLenFeature([], tf.string),
                'image_low_resolution': tf.io.FixedLenFeature([], tf.string),
                'image_high_resolution': tf.io.FixedLenFeature([], tf.string),
            },
        ),
    )

    

    dataset = next(filter(lambda x: x.name == dataset_name, dataset_paths))
    if remove_overlaps:
        remove_overlaps = _get_overlapping_identities(dataset.name)

    decode_function = partial(
        dataset.features,
        sample_ids=sample_ids,
    )
    #_load_dataset = partial(
    #    _load_dataset_from_path,
    #    sample_ids=sample_ids
    #)
    _load_dataset = partial(
        _load_from_tfrecords,
        dataset_name=dataset.name,
        decode_function=decode_function,
        overlaps=remove_overlaps,
    )

    if test:
        train = (
            _load_dataset(dataset.train),
            _get_number_of_classes(dataset.train, remove_overlaps),
            _get_dataset_size(dataset.train, remove_overlaps),
        )
        test = (
            _load_dataset(dataset.test),
            _get_number_of_classes(dataset.test, remove_overlaps),
            _get_dataset_size(dataset.test, remove_overlaps),
        )
        return train, test

    if concatenate:
        dataset = _load_dataset(dataset.train)
        return (
            dataset.concatenate(_load_dataset(dataset.test)),
            (_get_number_of_classes(dataset.train, remove_overlaps) +
             _get_number_of_classes(dataset.test, remove_overlaps)),
            (_get_dataset_size(dataset.train, remove_overlaps) +
             _get_dataset_size(dataset.test, remove_overlaps))
            )

    return (
        _load_dataset(dataset.train),
        _get_number_of_classes(dataset.train, remove_overlaps),
        _get_dataset_size(dataset.train, remove_overlaps)
    )

@tf.function
def load_lfw():
    """Loads the Labeled Faces in the Wild dataset, ready for validation tasks.

    ### Returns
        The LFW dataset.
    """
    dataset, _, _ = load_dataset('LFW', remove_overlaps=False, sample_ids=True)
    dataset = dataset.map(augment_dataset_without_concatenation)
    return dataset.map(
        lambda image, augmented_image, class_id, sample: (
            normalize_images(image),
            normalize_images(augmented_image),
            class_id,
            sample,
        )
    )

#if __name__ == "__main__":
    ###x = load_datasets()
    #for a, b, c in x:
    #    print(a)
    #    print(b)
    #    print(c)
    #load_dataset('VGGFace2')
    #print(_get_overlapping_identities('VGGFace2'))
