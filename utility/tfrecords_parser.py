import tensorflow as tf

def read_vggface2(serialized_dataset):
    """Parser for the VGGFace2 dataset, saved in TFRecods format.

    Arguments:
        serialized_dataset: dataset from tf.data.TFRecordDataset

    Output:
        MapDataset with shape: `(image_raw, class_id, sample, name, gender)`
    """

    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'class_id': tf.io.FixedLenFeature([], tf.string),
        'sample': tf.io.FixedLenFeature([], tf.string),
        'name': tf.io.FixedLenFeature([], tf.string),
        'gender': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(
        serialized_dataset,
        image_feature_description
    )
    return (
        example['image_raw'],
        example['class_id'],
        example['sample'],
        example['name'],
        example['gender']
    )

def read_lfw(serialized_dataset):
    """Parser for the Labeled Faces in the Wild dataset, saved in TFRecods format.

    Arguments:
        serialized_dataset: dataset from tf.data.TFRecordDataset

    Output:
        MapDataset with shape: `(image_raw, sample, name)`
    """

    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'sample': tf.io.FixedLenFeature([], tf.string),
        'name': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(
        serialized_dataset,
        image_feature_description
    )
    return (
        example['image_raw'],
        example['sample'],
        example['name']
    )

def read_casia(serialized_dataset):
    """Parser for the CASIA-Webface dataset, saved in TFRecods format.

    Arguments:
        serialized_dataset: dataset from tf.data.TFRecordDataset

    Output:
        MapDataset with shape: `(image_raw, sample, name)`
    """

    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'sample': tf.io.FixedLenFeature([], tf.string),
        'name': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(
        serialized_dataset,
        image_feature_description
    )
    return (
        example['image_raw'],
        example['sample'],
        example['name']
    )

def read_ms_celeb(serialized_dataset):
    """Parser for the MS-Celeb-1M DeepGlint dataset, saved in TFRecods format.

    Arguments:
        serialized_dataset: dataset from tf.data.TFRecordDataset

    Output:
        MapDataset with shape: `(
            image_raw,
            sample,
            class_id,
            left_eye_x, 
            left_eye_y,
            right_eye_x,
            right_eye_y,
            nose_tip_x,
            nose_tip_y,
            mouth_left_x,
            mouth_left_y,
            mouth_right_x,
            mouth_right_y
       )`
    """

    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'sample': tf.io.FixedLenFeature([], tf.string),
        'class_id': tf.io.FixedLenFeature([], tf.int64),
        'left_eye_x': tf.io.FixedLenFeature([], tf.float32),
        'left_eye_y': tf.io.FixedLenFeature([], tf.float32),
        'right_eye_x': tf.io.FixedLenFeature([], tf.float32),
        'right_eye_y': tf.io.FixedLenFeature([], tf.float32),
        'nose_tip_x': tf.io.FixedLenFeature([], tf.float32),
        'nose_tip_y': tf.io.FixedLenFeature([], tf.float32),
        'mouth_left_x': tf.io.FixedLenFeature([], tf.float32),
        'mouth_left_y': tf.io.FixedLenFeature([], tf.float32),
        'mouth_right_x': tf.io.FixedLenFeature([], tf.float32),
        'mouth_right_y': tf.io.FixedLenFeature([], tf.float32),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(
        serialized_dataset,
        image_feature_description
    )
    return (
        example['image_raw'],
        example['sample'],
        example['class_id'],
        example['left_eye_x'],
        example['left_eye_y'],
        example['right_eye_x'],
        example['right_eye_y'],
        example['nose_tip_x'],
        example['nose_tip_y'],
        example['mouth_left_x'],
        example['mouth_left_y'],
        example['mouth_right_x'],
        example['mouth_right_y']
    )

def read_tinyface(serialized_dataset):
    """Parser for the TinyFace dataset, saved in TFRecods format.

    Arguments:
        serialized_dataset: dataset from tf.data.TFRecordDataset

    Output:
        MapDataset with shape: `(image_raw, class_id, sample)`
    """

    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'class_id': tf.io.FixedLenFeature([], tf.string),
        'sample': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(
        serialized_dataset,
        image_feature_description
    )
    return (
        example['image_raw'],
        example['class_id'],
        example['sample']
    )

def read_qmul_survface(serialized_dataset):
    """Parser for the QMUL-SurvFace dataset, saved in TFRecods format.

    Arguments:
        serialized_dataset: dataset from tf.data.TFRecordDataset

    Output:
        MapDataset with shape: `(image_raw, class_id, sample)`
    """

    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'class_id': tf.io.FixedLenFeature([], tf.string),
        'sample': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(
        serialized_dataset,
        image_feature_description
    )
    return (
        example['image_raw'],
        example['class_id'],
        example['sample']
    )
