from tensorflow.keras.layers import Dense
from training.metrics import normalize


class ArcLossLayer(Dense):
    def __init__(self, scale: int, **kwargs):
        super(ArcLossLayer, self).__init__(use_bias=False, activation=None, **kwargs)
        self.scale = scale

    def call(self, inputs):
        self.kernel = normalize(self.kernel, name="weights_normalization")

        normalized_inputs = normalize(inputs, axis=1, name="embeddings_normalization")
        normalized_inputs *= self.scale

        return super().call(normalized_inputs)
