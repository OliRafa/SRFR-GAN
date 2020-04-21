from functools import wraps

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils.losses_utils import reduce_weighted_loss
from tensorflow.keras.losses import (
    BinaryCrossentropy,
    CategoricalCrossentropy,
    MAE,
    MeanAbsoluteError,
    MSE,
    MeanSquaredError,
)
from utils.input_data import parseConfigsFile


_BATCH_SIZE_PER_SAMPLE = parseConfigsFile(['train'])['batch_size']


def distributed_sum_over_batch_size(batch_size: int = _BATCH_SIZE_PER_SAMPLE):
    def _sum_over_batch_size(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            output_tensor = function(*args, **kwargs)
            return tf.nn.compute_average_loss(output_tensor,
                                              global_batch_size=batch_size)
        return wrapper
    return _sum_over_batch_size


def _sum_over_batch_size(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        loss_tensor = function(*args, **kwargs)
        return reduce_weighted_loss(loss_tensor)
    return wrapper


@tf.function
def apply_softmax(logits):
    return keras.activations.softmax(logits)


@_sum_over_batch_size
@tf.function
def compute_l1_loss(fake_outputs, ground_truth):
    return MAE(ground_truth, fake_outputs)


@tf.function
def compute_binary_crossentropy(y_true, y_predicted) -> float:
    """Compute Binary Categorical Crossentropy.

    ### Parameters:
        y_true:
        y_predicted:

    ### Returns:
        the computed loss.
    """
    return BinaryCrossentropy(
        from_logits=True,
        reduction=keras.losses.Reduction.NONE,
    )(y_true, y_predicted)


@tf.function
def compute_categorical_crossentropy(logits, labels) -> float:
    """Compute Sparse Categorical Crossentropy.

    ### Parameters:
        logits: the logits
        labels: the labels

    ### Returns:
        Computed loss.
    """
    return CategoricalCrossentropy(
        reduction=keras.losses.Reduction.NONE)(logits, labels)


@_sum_over_batch_size
@tf.function
def compute_euclidean_distance(fake_outputs, ground_truth) -> float:
    return MSE(ground_truth, fake_outputs)


@tf.function
def normalize(logits, axis: int = None, name: str = None):
    normalized = tf.linalg.normalize(logits, ord='euclidean', axis=axis,
                                     name=name)[0]
    return tf.squeeze(normalized)
