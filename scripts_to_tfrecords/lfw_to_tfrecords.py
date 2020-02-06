import os
import sys
import logging
import pathlib
import tensorflow as tf
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utility.timing import TimingLogger; TimingLogger().start()
from utility.input_data import split_path

logging.basicConfig(
    filename='lfw_to_tfrecords.txt',
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)

LOGGER.info('--- Setting Functions ---')

def _reduce_resolution(high_resolution_image):
    low_resolution_image = tf.image.resize(high_resolution_image, (28, 28), method='bicubic')
    low_resolution_image = tf.image.convert_image_dtype(low_resolution_image, tf.uint8)
    shape = (28, 28, 3)
    return tf.image.encode_png(low_resolution_image), shape, tf.image.encode_png(high_resolution_image)

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    try:
        value = value.encode('utf-8')
    except Exception:
        pass
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string_low_resolution, image_shape, _class_id, _sample_id):
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'class_id': _bytes_feature(_class_id),
        'sample_id': _bytes_feature(_sample_id),
        'image_low_resolution': _bytes_feature(image_string_low_resolution),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def preprocess_image(image_path):
    class_id, sample_id = split_path(str(image_path))
    high_resolution_image = tf.io.read_file(str(image_path))
    high_resolution_image = tf.image.decode_jpeg(high_resolution_image)
    low_resolution, image_shape_lr, _ = _reduce_resolution(high_resolution_image)
    return image_example(
        low_resolution,
        image_shape_lr,
        class_id,
        sample_id
    )

data_dir = pathlib.Path('/mnt/hdd_raid/datasets/LFW/lfw-deepfunneled/lfw-deepfunneled')
data_dir = list(data_dir.glob('*/*.jpg'))
partial = 1
total = len(data_dir)
PATH = '/mnt/hdd_raid/datasets/LFW/Raw_Low_Resolution.tfrecords'
LOGGER.info(' Started Recording')

with tf.io.TFRecordWriter(PATH) as writer:
    for image in data_dir:
        LOGGER.info(f' Image {partial}/{total}')
        tf_example = preprocess_image(image)
        writer.write(tf_example.SerializeToString())
        partial += 1
