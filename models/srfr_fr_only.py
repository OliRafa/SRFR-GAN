import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from models.arcloss_layer import ArcLossLayer
from models.resnet import ResNet

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_policy(policy)


class SrfrFrOnly(Model):
    def __init__(
        self,
        depth: int = 50,
        categories: int = 512,
        num_classes: int = 2,
        scale: int = 64,
        training: bool = True,
        input_shape=(112, 112, 3),
    ):
        super(SrfrFrOnly, self).__init__()
        self._face_recognition = ResNet(depth, categories, training, input_shape)
        self._arcloss_layer = ArcLossLayer(
            units=num_classes,
            scale=scale,
            dtype="float32",
            name="arcloss_layer",
        )

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
