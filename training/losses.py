"""This module contains losses to be used during training.

Losses:
    - ArcLoss
    - Discriminator Loss
    - Generator Loss
    - Joint Loss
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import (
    BinaryCrossentropy,
    MAE,
    MSE,
    SparseCategoricalCrossentropy
)

@tf.function
def normalize(logits, name: str = None):
    return tf.norm(logits, ord='euclidean', axis=1, name=name)

@tf.function
def _compute_l1_loss(fake_outputs, ground_truth, weigth: float = 1e-2):
    return weigth * MAE(ground_truth, fake_outputs)

@tf.function
def _compute_euclidean_distance(fake_outputs, ground_truth) -> float:
    return MSE(ground_truth, fake_outputs)

@tf.function
def _compute_categorical_crossentropy(logits, labels) -> float:
    """Compute Sparse Categorical Crossentropy.

    ### Parameters:
        logits: the logits
        labels: the labels

    ### Returns:
        the computed loss.
    """
    return SparseCategoricalCrossentropy()(
        logits,
        labels,
    )

@tf.function
def _compute_binary_crossentropy(y_true, y_predicted) -> float:
    """Compute Binary Categorical Crossentropy.

    ### Parameters:
        y_true:
        y_predicted:

    ### Returns:
        the computed loss.
    """
    return BinaryCrossentropy(from_logits=True)(
        y_true,
        y_predicted,
    )

@tf.function
def _apply_softmax(logits):
    return keras.activations.Softmax(logits, name='softmax')

@tf.function
def compute_arcloss(
        embeddings,
        ground_truth,
        fc_weights,
        num_classes: int,
        scale: int,
        margin: float,
    ) -> float:
    """Compute the ArcLoss.

    ### Parameters
        embeddings: Batch of Embedding vectors where loss will be calculated on.
        ground_truth: Batch of Ground Truth classes.
        fc_weights: Weights extracted from the last Fully Connected layer of\
 the network (Embedding layer).
        num_classes: Total number of classes in the dataset.
        scale:
        margin:

    ### Returns
        The loss value."""
    normalized_weights = normalize(fc_weights, 'weights_normalization')
    normalized_embeddings = normalize(
        embeddings, 'embeddings_normalization') * scale

    dense_layer = Dense(
        units=num_classes,
        use_bias=False,
        name='fully_connected_to_softmax_crossentropy',
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
    return _compute_categorical_crossentropy(softmax_output, one_hot_vector)

@tf.function
def compute_discriminator_loss(real_output, fake_output) -> float:
    real_loss = _compute_binary_crossentropy(
        tf.ones_like(real_output),
        real_output,
    )
    fake_loss = _compute_binary_crossentropy(
        tf.zeros_like(fake_output),
        fake_output,
    )
    return real_loss + fake_loss

@tf.function
def _create_vgg_model():
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # Removing the activation function from the last conv layer 'block5_conv4'
    vgg.layers[-2].activation = None

    # Creating the model with layers [input...last_conv_layer]
    # I.e. removing the last MaxPooling layer from the model
    return keras.Model(
        inputs=vgg.input,
        outputs=[layer.output for layer in vgg.layers[:-1]],
    )

@tf.function
def _compute_perceptual_loss(
        super_resolution,
        ground_truth,
        weight: float = 1.0,
    ) -> float:
    vgg = _create_vgg_model()
    fake = vgg(super_resolution)
    real = vgg(ground_truth)
    return weight * _compute_euclidean_distance(fake, real)

@tf.function
def compute_generator_loss(super_resolution, ground_truth) -> float:
    perceptual_loss = _compute_perceptual_loss(super_resolution, ground_truth)
    generator_loss = _compute_binary_crossentropy(
        tf.ones_like(super_resolution),
        super_resolution,
    )
    l1_loss = _compute_l1_loss(super_resolution, ground_truth)
    return perceptual_loss + generator_loss + l1_loss

@tf.function
def compute_joint_loss_simple(
        super_resolution,
        embedding,
        ground_truth,
        fc_weights,
        num_classes: int,
        weight: float,
        scale: float = 64,
        margin: float = 0.5,
    ) -> float:
    super_resolution_loss = compute_generator_loss(
        super_resolution,
        ground_truth,
    )
    face_recognition_loss = compute_arcloss(
        embedding,
        ground_truth,
        fc_weights,
        num_classes,
        scale,
        margin,
        )
    return face_recognition_loss + weight * super_resolution_loss

@tf.function
def compute_joint_loss(
        super_resolution,
        ground_truth_images,
        synthetic_face_recognition,
        natural_face_recognition,
        num_classes: int,
        weight: float,
        scale: float = 64.0,
        margin: float = 0.5,
    ) -> float:
    """Computes the Joint Loss for Super Resolution Face Recognition, using\
 outputs from Synthetic SRFR and Natural SRFR.

    ### Parameters
        super_resolution: Outputs from the Super Resolution Generator.
        ground_truth_images: High Resolution inputs.
        synthetic_face_recognition: A tuple of (embeddings, ground_truth_\
classes, fc_weights) from the Synthetic SRFR.
        natural_face_recognition: A tuple of (embeddings, ground_truth_\
classes, fc_weights) from the Natural SRFR.
        num_classes: Total number of classes in the dataset.
        weight: Weight for the SR Loss.
        scale: Scale parameter for ArcLoss.
        margin: Margin penalty for ArcLoss.

    ### Returns:
        The loss value.
    """
    super_resolution_loss = compute_generator_loss(
        super_resolution,
        ground_truth_images,
    )
    synthetic_face_recognition_loss = compute_arcloss(
        synthetic_face_recognition[0],
        synthetic_face_recognition[1],
        synthetic_face_recognition[2],
        num_classes,
        scale,
        margin,
    )
    natural_face_recognition_loss = compute_arcloss(
        natural_face_recognition[0],
        natural_face_recognition[1],
        natural_face_recognition[2],
        num_classes,
        scale,
        margin,
    )
    fr_loss = synthetic_face_recognition_loss + natural_face_recognition_loss
    return fr_loss + weight * super_resolution_loss
