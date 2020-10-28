import os
import sys

sys.path.append(os.path.abspath("."))

import logging
from pathlib import Path

import cv2
import tensorflow as tf
from tqdm import tqdm
from utils.input_data import InputData, parseConfigsFile
from utils.timing import TimingLogger

logging.basicConfig(filename="casia_to_tfrecords.txt", level=logging.INFO)
LOGGER = logging.getLogger(__name__)

timing = TimingLogger()
timing.start()

LOGGER.info("--- Setting Functions ---")

SHAPE = tuple(parseConfigsFile(["preprocess"])["image_shape_low_resolution"][:2])

DATASET_NAME = "CASIA_Webface"
BASE_DATA_DIR = Path("/workspace/data/datasets/CASIA_LR")
BASE_OUTPUT_PATH = Path("/workspace/data/datasets/CASIA_LR_TFRecords")
if not BASE_OUTPUT_PATH.is_dir():
    BASE_OUTPUT_PATH.mkdir(parents=True)


def _reduce_resolution(high_resolution_image):
    low_resolution_image = cv2.cvtColor(
        cv2.resize(high_resolution_image, SHAPE, interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB,
    )
    high_resolution_image = cv2.cvtColor(high_resolution_image, cv2.COLOR_BGR2RGB)
    return (
        tf.image.encode_png(low_resolution_image),
        tf.image.encode_png(high_resolution_image),
    )


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    try:
        value = value.encode("utf-8")
    except Exception:
        pass
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string_low_resolution, image_string_high_resolution, _class_id):
    feature = {
        "class_id": _bytes_feature(_class_id),
        "image_low_resolution": _bytes_feature(image_string_low_resolution),
        "image_high_resolution": _bytes_feature(image_string_high_resolution),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def preprocess_image(image_path):
    class_id, _ = InputData.split_path(str(image_path))
    high_resolution_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    low_resolution, high_resolution_image = _reduce_resolution(high_resolution_image)
    return image_example(low_resolution, high_resolution_image, class_id)


timing.start("train")

data_dir = list(BASE_DATA_DIR.glob("*/*.jpg"))

_NUM_IMAGES = len(data_dir)

index = 0
n_images_shard = 8000
n_shards = int(_NUM_IMAGES / n_images_shard) + (1 if _NUM_IMAGES % 800 != 0 else 0)

for shard in tqdm(range(n_shards)):
    tfrecords_shard_path = BASE_OUTPUT_PATH.joinpath(
        f"{DATASET_NAME}_{shard:03d}-of-{(n_shards - 1):03d}.tfrecords"
    )
    end = index + n_images_shard if _NUM_IMAGES > (index + n_images_shard) else -1
    images_shard_list = data_dir[index:end]

    with tf.io.TFRecordWriter(str(tfrecords_shard_path)) as writer:
        for image in images_shard_list:
            tf_example = preprocess_image(image)
            writer.write(tf_example.SerializeToString())

    index = end

timing.end("train")
