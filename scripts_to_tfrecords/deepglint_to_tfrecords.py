import os
import sys
import pathlib
import tensorflow as tf
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils.timing

tf.debugging.set_log_device_placement(True)

metadata = pd.read_csv(
    '/mnt/hdd_raid/datasets/DeepGlint/celebrity_lmk',
    sep=' ',
    names=['sample',
           'class_id',
           'left_eye_x',
           'left_eye_y',
           'right_eye_x',
           'right_eye_y',
           'nose_tip_x',
           'nose_tip_y',
           'mouth_left_x',
           'mouth_left_y',
           'mouth_right_x',
           'mouth_right_y']
)

print('--- Loading Metadata ---')

def update_keys(key): 
    key = key.strip('celebrity/')
    index = key.find('/')
    return key[index+1:]

def new_keys(key): 
    key = key.strip('celebrity/')
    index = key.find('/')
    return 'm' + key[:index]

metadata['class_name'] = metadata['sample'].map(new_keys)
metadata['sample'] = metadata['sample'].map(update_keys)

print('--- Setting Functions ---')

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string, image_shape, label, metadata):
    metadata = metadata.loc[metadata['class_name'] == label[0]]
    metadata = metadata.loc[metadata['sample'] == label[1]].values

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'class_id': _int64_feature(metadata[0, 1]),
        'sample': _bytes_feature(label[1].encode('UTF-8')),
        'left_eye_x': _float_feature(metadata[0, 2]),
        'left_eye_y': _float_feature(metadata[0, 3]),
        'right_eye_x': _float_feature(metadata[0, 4]),
        'right_eye_y': _float_feature(metadata[0, 5]),
        'nose_tip_x': _float_feature(metadata[0, 6]),
        'nose_tip_y': _float_feature(metadata[0, 7]),
        'mouth_left_x': _float_feature(metadata[0, 8]),
        'mouth_left_y': _float_feature(metadata[0, 9]),
        'mouth_right_x': _float_feature(metadata[0, 10]),
        'mouth_right_y': _float_feature(metadata[0, 11]),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

print('--- Loading Dataset ---')

data_dir = pathlib.Path('/mnt/hdd_raid/datasets/DeepGlint/celebrity')

partial = 1
total = len(list(data_dir.glob('*/*.jpg')))

with tf.io.TFRecordWriter(
    '/mnt/hdd_raid/datasets/TFRecords/DeepGlint/Asian_Raw.tfrecords'
    ) as writer:
    for image in list(data_dir.glob('*/*.jpg')):
        print('Test image: {}/{}'.format(partial, total))
        label = image.parts[-2], image.parts[-1]
        img = tf.io.read_file(str(image))
        img_shape = tf.shape(tf.image.decode_jpeg(img)).numpy()
        tf_example = image_example(img, img_shape, label, metadata)
        writer.write(tf_example.SerializeToString())
        partial += 1
