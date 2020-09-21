import sys, os  # isort:skip

sys.path.append(os.path.abspath("."))  # isort:skip

import logging
from pathlib import Path

import cv2
import tensorflow as tf

from utils.input_data import InputData, parseConfigsFile
from utils.timing import TimingLogger

logging.basicConfig(filename="vgg_to_tfrecords.txt", level=logging.INFO)
LOGGER = logging.getLogger(__name__)

timing = TimingLogger()
timing.start()

LOGGER.info("--- Setting Functions ---")

SHAPE = tuple(parseConfigsFile(["preprocess"])["image_shape_low_resolution"][:2])

BASE_DATA_DIR = Path("/datasets/VGGFace2_LR/Images")
BASE_OUTPUT_PATH = Path("/workspace/datasets/VGGFace2")


def _reduce_resolution(high_resolution_image):
    low_resolution_image = cv2.cvtColor(
        cv2.resize(high_resolution_image, SHAPE, interpolation=cv2.INTER_CUBIC),
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


def image_example(
    image_string_low_resolution, image_string_high_resolution, _class_id, _sample_id
):
    feature = {
        "class_id": _bytes_feature(_class_id),
        "sample_id": _bytes_feature(_sample_id),
        "image_low_resolution": _bytes_feature(image_string_low_resolution),
        "image_high_resolution": _bytes_feature(image_string_high_resolution),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def preprocess_image(image_path):
    class_id, sample_id = InputData.split_path(str(image_path))
    high_resolution_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    low_resolution, high_resolution_image = _reduce_resolution(high_resolution_image)
    return image_example(low_resolution, high_resolution_image, class_id, sample_id)


_NUM_IMAGES = 5000

timing.start("test")

data_dir = list(BASE_DATA_DIR.joinpath("test").glob("*/*.jpg"))[:_NUM_IMAGES]
partial = 1
output_path = str(BASE_OUTPUT_PATH.joinpath("Test_Low_Resolution_5k.tfrecords"))
with tf.io.TFRecordWriter(output_path) as writer:
    for image in data_dir:
        LOGGER.info(f" Test Image {partial}/{_NUM_IMAGES}")
        tf_example = preprocess_image(image)
        writer.write(tf_example.SerializeToString())
        partial += 1

timing.end("test")

timing.start("train")

data_dir = list(BASE_DATA_DIR.joinpath("train").glob("*/*.jpg"))[:_NUM_IMAGES]
partial = 1
output_path = str(BASE_OUTPUT_PATH.joinpath("Train_Low_Resolution_5k.tfrecords"))
with tf.io.TFRecordWriter(output_path) as writer:
    for image in data_dir:
        LOGGER.info(f" Train Image {partial}/{_NUM_IMAGES}")
        tf_example = preprocess_image(image)
        writer.write(tf_example.SerializeToString())
        partial += 1

timing.end("train")
