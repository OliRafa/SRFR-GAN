import tensorflow as tf

from models.srfr import SRFR


class SrfrSrOnly(SRFR):
    def __init__(
        self,
        num_filters: int = 62,
        num_gc: int = 32,
        num_blocks: int = 23,
        residual_scailing: float = 0.2,
        training: bool = True,
        input_shape=(28, 28, 3),
    ):
        super(SrfrSrOnly, self).__init__(
            num_filters=num_filters,
            num_gc=num_gc,
            num_blocks=num_blocks,
            residual_scailing=residual_scailing,
            training=training,
            input_shape=input_shape,
        )
        del self._face_recognition
        del self._fc_classification_syn

    def call(
        self,
        input_tensor_01,
        input_tensor_02=None,
        training: bool = True,
        input_type: str = "syn",
    ):
        if training:
            return self._call_training(input_tensor_01, input_tensor_02)

        return self._call_evaluating(input_tensor_01, input_type)

    @tf.function
    def _call_evaluating(self, input_tensor, input_type: str = "nat"):
        if input_type == "syn":
            outputs = self._synthetic_input(input_tensor)
        else:
            outputs = self._natural_input(input_tensor)

        return self._super_resolution(outputs)

    def _call_training(self, synthetic_images, natural_images=None):
        synthetic_outputs = self._synthetic_input(synthetic_images)
        synthetic_sr_images = self._super_resolution(synthetic_outputs)

        if natural_images:
            natural_outputs = self._natural_input(natural_images)
            natural_sr_images = self._super_resolution(natural_outputs)

            return synthetic_sr_images, natural_sr_images

        return synthetic_sr_images
