import tensorflow as tf

from models.arcloss_layer import ArcLossLayer
from models.srfr import SRFR


class SrfrFrOnly(SRFR):
    def __init__(
        self,
        depth: int = 50,
        categories: int = 512,
        num_classes: int = 2,
        scale: int = 64,
        training: bool = True,
        input_shape=(28, 28, 3),
    ):
        super(SrfrFrOnly, self).__init__(
            depth=depth,
            categories=categories,
            training=training,
            input_shape=input_shape,
        )
        self._arcloss_layer = ArcLossLayer(
            input_shape=(categories,),
            units=num_classes,
            scale=scale,
            dtype="float32",
            name="arcloss_layer",
        )
        self._arcloss_layer.build(tf.TensorShape([None, categories]))
        del self._synthetic_input
        del self._super_resolution

    def call(self, input_tensor, training: bool = True):
        if training:
            return self._call_training(input_tensor)

        return self._call_evaluating(input_tensor)

    @tf.function
    def _call_evaluating(self, input_tensor):
        return self._face_recognition(input_tensor)

    def _call_training(self, input_tensor):
        outputs = self._face_recognition(input_tensor)
        return self._arcloss_layer(outputs)
