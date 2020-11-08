from functools import wraps

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import (
    BinaryCrossentropy,
    CategoricalCrossentropy,
    MeanAbsoluteError,
    MeanSquaredError,
)


def distributed_sum_over_batch_size(batch_size: int):
    def _distributed_sum_over_batch_size(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            output_tensor = function(*args, **kwargs)
            return tf.nn.compute_average_loss(
                output_tensor, global_batch_size=batch_size
            )

        return wrapper

    return _distributed_sum_over_batch_size


def distributed_mean(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        output_tensor = function(*args, **kwargs)
        return tf.math.reduce_mean(output_tensor) / 2

    return wrapper


@tf.function
def apply_softmax(logits):
    return keras.activations.softmax(logits)


@distributed_mean
@tf.function
def compute_l1_loss(fake_outputs, ground_truth):
    return MeanAbsoluteError(reduction=keras.losses.Reduction.NONE)(
        ground_truth, fake_outputs
    )


@distributed_mean
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
        from_logits=False,
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
    # return CategoricalCrossentropy()(  # reduction=keras.losses.Reduction.NONE)(
    #    logits, labels
    # )
    return CategoricalCrossentropy(reduction=keras.losses.Reduction.NONE)(
        labels, logits
    )


@distributed_mean
@tf.function
def compute_euclidean_distance(fake_outputs, ground_truth) -> float:
    return MeanSquaredError(reduction=keras.losses.Reduction.NONE)(
        ground_truth, fake_outputs
    )


@tf.function
def normalize(logits, axis: int = None, name: str = None):
    normalized = tf.linalg.normalize(logits, ord="euclidean", axis=axis, name=name)[0]
    return tf.squeeze(normalized)
