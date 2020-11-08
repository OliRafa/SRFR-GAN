import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
from training.metrics import normalize


class ArcLossLayer(Dense):
    def __init__(self, scale: int, **kwargs):
        super(ArcLossLayer, self).__init__(use_bias=False, activation=None, **kwargs)
        self.scale = scale

    def call(self, inputs):
        normalized_weights = normalize(self.kernel, name="weights_normalization")

        normalized_inputs = normalize(inputs, axis=1, name="embeddings_normalization")
        normalized_inputs *= self.scale

        return tf.matmul(normalized_inputs, normalized_weights)

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config
