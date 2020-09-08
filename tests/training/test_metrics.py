import tensorflow as tf

from conftest import transform_to_tf_tensor
from training.metrics import apply_softmax


def test_softmax_output(softmax_real_input, softmax_real_output):
    output = apply_softmax(softmax_real_input)
    assert (output == softmax_real_output).numpy().all() == True


def test_softmax_output_sum(softmax_input_minimal):
    output = apply_softmax(softmax_input_minimal)
    assert tf.reduce_sum(output).numpy() == 1.0


def test_softmax_zeros_output(zeros_array, softmax_zeros_array_output):
    output = apply_softmax(zeros_array)
    assert (output == softmax_zeros_array_output).numpy().all() == True
