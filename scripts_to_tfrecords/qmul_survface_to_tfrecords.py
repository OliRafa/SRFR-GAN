import os
import sys
import pathlib
import tensorflow as tf
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils.timing

tf.debugging.set_log_device_placement(True)

print('--- Setting Functions ---')

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string, image_shape, label):

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'class_id': _bytes_feature(label[0].encode('UTF-8')),
        'sample': _bytes_feature(label[1].encode('UTF-8')),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

print('--- Loading Train Dataset ---')

data_dir = pathlib.Path(
    '/mnt/hdd_raid/datasets/QMUL-SurvFace/Challenge_Train_Validation_Set/training_set'
)

partial = 1
total = len(list(data_dir.glob('*/*.jpg')))

with tf.io.TFRecordWriter(
    '/mnt/hdd_raid/datasets/TFRecords/QMUL-SurvFace/Train_Raw.tfrecords'
    ) as writer:
    for image in list(data_dir.glob('*/*.jpg')):
        print('Test image: {}/{}'.format(partial, total))
        label = image.parts[-2], image.parts[-1]
        img = tf.io.read_file(str(image))
        img_shape = tf.shape(tf.image.decode_jpeg(img)).numpy()
        tf_example = image_example(img, img_shape, label)
        writer.write(tf_example.SerializeToString())
        partial += 1
