import os
import sys
import pathlib
import tensorflow as tf
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utility.timing

metadata = pd.read_csv('/mnt/hdd_raid/datasets/VGGFace2/identity_meta.csv')

print('--- Loading Metadata ---')

def update_keys(key): return str.strip(str(key))

metadata = metadata.rename(lambda x: update_keys(x), axis='columns')
metadata.Gender = metadata.Gender.str.strip()
metadata.Name = metadata.Name.str.strip()
metadata.Name = metadata.Name.str.strip('"')

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
    metadata = metadata.loc[metadata['Class_ID'] == label[0]].values

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'class_id': _bytes_feature(label[0].encode('UTF-8')),
        'sample': _bytes_feature(label[1].encode('UTF-8')),
        'name': _bytes_feature(metadata[0,1].encode('UTF-8')),
        'gender': _bytes_feature(metadata[0,-1].encode('UTF-8')),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

print('--- Loading Test Dataset ---')

data_dir = pathlib.Path('/mnt/hdd_raid/datasets/VGGFace2/test')

partial = 1
total = len(list(data_dir.glob('*/*.jpg')))

with tf.io.TFRecordWriter(
    '/mnt/hdd_raid/datasets/TFRecords/VGGFace2/Test_Raw.tfrecords') as writer:
    for image in list(data_dir.glob('*/*.jpg')):
        print('Test image: {}/{}'.format(partial, total))
        label = image.parts[-2], image.parts[-1]
        img = tf.io.read_file(str(image))
        img_shape = tf.shape(tf.image.decode_jpeg(img)).numpy()
        #img_array = tf.keras.preprocessing.image.img_to_array(img, dtype=int)
        #img_bytes = tf.io.serialize_tensor(img_array)
        tf_example = image_example(img, img_shape, label, metadata)
        writer.write(tf_example.SerializeToString())
        partial += 1

print('--- Loading Train Dataset ---')

data_dir = pathlib.Path('/mnt/hdd_raid/datasets/VGGFace2/train')

partial = 1
total = len(list(data_dir.glob('*/*.jpg')))

with tf.io.TFRecordWriter('/mnt/hdd_raid/datasets/TFRecords/VGGFace2/Train_Raw.tfrecords') as writer:
    for image in list(data_dir.glob('*/*.jpg')):
        print('Train image: {}/{}'.format(partial, total))
        label = image.parts[-2], image.parts[-1]
        img = tf.io.read_file(str(image))
        img_shape = tf.shape(tf.image.decode_jpeg(img)).numpy()
        tf_example = image_example(img, img_shape, label, metadata)
        writer.write(tf_example.SerializeToString())
        partial += 1
