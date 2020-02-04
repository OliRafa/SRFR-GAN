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
    return tf.image.resize(high_resolution_image, (28, 28), method='bicubic')

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

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

data_dir = pathlib.Path('/mnt/hdd_raid/datasets/LFW/lfw-deepfunneled/lfw-deepfunneled')
partial = 1
total = len(list(data_dir.glob('*/*.jpg')))

LOGGER.info(' Started Recording')

with tf.io.TFRecordWriter('/mnt/hdd_raid/datasets/TFRecords/LFW/Raw_Low_Resolution.tfrecords') as writer:
    for image in list(data_dir.glob('*/*.jpg')):
        LOGGER.info(f' Image {partial}/{total}')
        class_id, sample_id = split_path(str(image))
        img = tf.io.read_file(str(image))
        img = tf.image.decode_jpeg(img)
        low_resolution = _reduce_resolution(img)
        img_shape = tf.shape(low_resolution).numpy()
        low_resolution = tf.io.encode_jpeg(low_resolution)
        tf_example = image_example(low_resolution, img_shape, class_id, sample_id)
        writer.write(tf_example.SerializeToString())
        partial += 1
