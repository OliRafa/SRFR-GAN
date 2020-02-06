import os
import sys
import pathlib
import logging
import tensorflow as tf
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utility.timing import TimingLogger
from utility.input_data import split_path

logging.basicConfig(
    filename='vgg_to_tfrecords.txt',
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)

timing = TimingLogger()
timing.start()

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

def image_example(image_string_low_resolution, image_string_high_resolution, image_shape_hr, image_shape_lr, _class_id, _sample_id):
    feature = {
        'height_lr': _int64_feature(image_shape_lr[0]),
        'width_lr': _int64_feature(image_shape_lr[1]),
        'depth_lr': _int64_feature(image_shape_lr[2]),
        'height_hr': _int64_feature(image_shape_hr[0]),
        'width_hr': _int64_feature(image_shape_hr[1]),
        'depth_hr': _int64_feature(image_shape_hr[2]),
        'class_id': _bytes_feature(_class_id),
        'sample_id': _bytes_feature(_sample_id),
        'image_low_resolution': _bytes_feature(image_string_low_resolution),
        'image_high_resolution': _bytes_feature(image_string_high_resolution),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def preprocess_image(image_path):
    class_id, sample_id = split_path(str(image_path))
    high_resolution_image = tf.io.read_file(str(image_path))
    high_resolution_image = tf.image.decode_jpeg(high_resolution_image)
    image_shape_hr = tf.shape(high_resolution_image).numpy()
    low_resolution, image_shape_lr, high_resolution_image = _reduce_resolution(high_resolution_image)
    return image_example(
        low_resolution,
        high_resolution_image,
        image_shape_hr,
        image_shape_lr,
        class_id,
        sample_id
    )

timing.start('test')
data_dir = pathlib.Path('/mnt/hdd_raid/datasets/VGGFace2_Aligned/test')
data_dir = list(data_dir.glob('*/*.jpg'))
partial = 1
total = len(data_dir)
PATH = '/mnt/hdd_raid/datasets/VGGFace2_Aligned/Test_Low_Resolution_Raw.tfrecords'
with tf.io.TFRecordWriter(PATH) as writer:
    for image in data_dir:
        LOGGER.info(f' Test Image {partial}/{total}')
        tf_example = preprocess_image(image)
        writer.write(tf_example.SerializeToString())
        partial += 1
timing.end('test')

timing.start('train')
data_dir = pathlib.Path('/mnt/hdd_raid/datasets/VGGFace2_Aligned/train')
data_dir = list(data_dir.glob('*/*.jpg'))
partial = 1
total = len(data_dir)
PATH = '/mnt/hdd_raid/datasets/VGGFace2_Aligned/Train_Low_Resolution_Raw.tfrecords'
with tf.io.TFRecordWriter(PATH) as writer:
    for image in data_dir:
        LOGGER.info(f' Train Image {partial}/{total}')
        tf_example = preprocess_image(image)
        writer.write(tf_example.SerializeToString())
        partial += 1
timing.end('train')