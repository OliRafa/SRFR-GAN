import numpy as np
import tensorflow as tf


def tensor_to_image(tensor):
    return (np.squeeze(tensor.numpy()).clip(0, 1) * 255).astype(np.uint8)


@tf.function
def denormalize_tensor(tensor):
    tensor = tf.math.multiply(tensor, 128)
    return tf.math.add(tensor, 127.5)


@tf.function
def tensor_to_uint8(tensor):
    return tf.image.convert_image_dtype(tensor, dtype=tf.uint8, saturate=True)
