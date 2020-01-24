"""This module contains losses to be used during training.

Losses:
    - ArcLoss
    - Softmax
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

@tf.function
def _normalize(logits, name: str = None):
    return tf.norm(logits, ord='euclidean', axis=1, name=name)

@tf.function
def _compute_crossentropy(logits, labels) -> float:
    """Compute the Sparse Categorical Crossentropy.

    ## Parameters:
        logits: the logits
        labels: the labels

    ## Returns:
        the computed loss.
    """
    return keras.losses.SparseCategoricalCrossentropy()(
        logits,
        labels
    )

@tf.function
def _apply_softmax(logits):
    return tf.keras.activations.Softmax(logits, name='softmax')

@tf.function
def compute_arcloss(
        embeddings,
        ground_truth,
        fc_weights,
        num_classes: int,
        scale: int,
        margin: float
    ) -> float:
    """Compute the ArcLoss.

    ## Parameters
        embeddings: Batch of Embedding vectors where loss will be calculated on.
        ground_truth: Batch of Ground Truth classes.
        fc_weights: Weights extracted from the last Fully Connected layer of\
 the network (Embedding layer).
        num_classes: Total number of classes in the dataset.
        scale:
        margin:

    ## Returns
        The loss value."""
    normalized_weights = _normalize(fc_weights, 'weights_normalization')
    normalized_embeddings = _normalize(
        embeddings, 'embeddings_normalization') * scale

    dense_layer = Dense(
        units=num_classes,
        use_bias=False,
        name='fully_connected_to_softmax_crossentropy'
    )
    dense_layer.set_weights(normalized_weights)
    original_target_embeddings = dense_layer(normalized_embeddings)

    cos_theta = original_target_embeddings / scale
    theta = tf.acos(cos_theta)

    z = theta + margin
    marginal_target_embeddings = tf.cos(z) * scale

    one_hot_vector = tf.one_hot(ground_truth, depth=num_classes)

    difference = marginal_target_embeddings - original_target_embeddings
    new_one_hot = one_hot_vector * difference

    softmax_output = _apply_softmax(original_target_embeddings + new_one_hot)
    return _compute_crossentropy(softmax_output, one_hot_vector)
