import tensorflow as tf
from tensorflow.keras.layers import Layer
from training.metrics import normalize


class ArcLossLayer(Layer):
    def __init__(self, units: int, scale: int, **kwargs):
        super(ArcLossLayer, self).__init__(**kwargs)
        self.units = units
        self.scale = scale

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        normalized_weights = normalize(self.w, name="weights_normalization")

        normalized_inputs = normalize(inputs, axis=1, name="embeddings_normalization")
        normalized_inputs *= self.scale

        return tf.matmul(normalized_inputs, normalized_weights)
