import tensorflow as tf
from tensorflow.python.keras.utils.losses_utils import reduce_weighted_loss

from conftest import transform_to_tf_tensor
from training.metrics import (
    apply_softmax,
    compute_binary_crossentropy,
    compute_euclidean_distance,
    compute_l1_loss,
)


def test_softmax_output(softmax_real_input, softmax_real_output):
    output = apply_softmax(softmax_real_input)
    assert (output == softmax_real_output).numpy().all() == True


def test_softmax_output_sum(softmax_input_minimal):
    output = apply_softmax(softmax_input_minimal)
    assert tf.reduce_sum(output).numpy() == 1.0


def test_softmax_zeros_output(zeros_array, softmax_zeros_array_output):
    output = apply_softmax(zeros_array)
    assert (output == softmax_zeros_array_output).numpy().all() == True


def test_compute_l1_loss(
    super_resolution_images, ground_truth_images, batch_size, l1_loss_output
):
    output = compute_l1_loss(super_resolution_images, ground_truth_images)

    assert output == l1_loss_output / 2


def test_compute_euclidean_distance(
    vgg_output_fakes, vgg_output_reals, vgg_euclidian_distance_output
):
    output = compute_euclidean_distance(vgg_output_fakes, vgg_output_reals)

    assert output == vgg_euclidian_distance_output / 2


def test_compute_binary_crossentropy(
    binary_crossentropy_zeros_like_input,
    binary_crossentropy_softmax_input,
    binary_crossentropy_output,
    batch_size,
):
    output = compute_binary_crossentropy(
        binary_crossentropy_zeros_like_input, binary_crossentropy_softmax_input
    )

    assert output == binary_crossentropy_output / 2


def test_compute_binary_crossentropy2(
    binary_crossentropy_ones_like_input,
    binary_crossentropy_softmax_input2,
    binary_crossentropy_output2,
    batch_size,
):
    output = compute_binary_crossentropy(
        binary_crossentropy_ones_like_input, binary_crossentropy_softmax_input2
    )

    assert output == binary_crossentropy_output2 / 2
