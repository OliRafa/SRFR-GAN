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
    return tf.image.encode_png(low_resolution_image)

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string_low_resolution, image_string_high_resolution, image_shape, _class_id, _sample_id):
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'class_id': _bytes_feature(_class_id),
        'sample': _bytes_feature(_sample_id),
        'image_low_resolution': _bytes_feature(image_string_low_resolution),
        'image_high_resolution': _bytes_feature(image_string_high_resolution),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

timing.start('test')
data_dir = pathlib.Path('/mnt/hdd_raid/datasets/VGGFace2_Aligned/test')
partial = 1
total = len(list(data_dir.glob('*/*.jpg')))

with tf.io.TFRecordWriter(
    '/mnt/hdd_raid/datasets/TFRecords/VGGFace2/Test_Low_Resolution_Raw.tfrecords') as writer:
    for image in list(data_dir.glob('*/*.jpg')):
        LOGGER.info(f' Test Image {partial}/{total}')
        class_id, sample_id = split_path(str(image))
        img = tf.io.read_file(str(image))
        high_resolution = tf.image.decode_jpeg(img)
        low_resolution = _reduce_resolution(high_resolution)
        img_shape = tf.shape(high_resolution).numpy()
        #img_array = tf.keras.preprocessing.image.img_to_array(img, dtype=int)
        #img_bytes = tf.io.serialize_tensor(img_array)
        tf_example = image_example(low_resolution, img, img_shape, class_id, sample_id)
        writer.write(tf_example.SerializeToString())
        partial += 1
timing.end('test')

timing.start('train')
data_dir = pathlib.Path('/mnt/hdd_raid/datasets/VGGFace2_Aligned/train')
partial = 1
total = len(list(data_dir.glob('*/*.jpg')))

with tf.io.TFRecordWriter('/mnt/hdd_raid/datasets/TFRecords/VGGFace2/Train_Low_Resolution_Raw.tfrecords') as writer:
    for image in list(data_dir.glob('*/*.jpg')):
        LOGGER.info(f' Train Image {partial}/{total}')
        class_id, sample_id = split_path(str(image))
        img = tf.io.read_file(str(image))
        high_resolution = tf.image.decode_jpeg(img)
        low_resolution = _reduce_resolution(high_resolution)
        img_shape = tf.shape(high_resolution).numpy()
        tf_example = image_example(low_resolution, img, img_shape, class_id, sample_id)
        writer.write(tf_example.SerializeToString())
        partial += 1
timing.end('train')