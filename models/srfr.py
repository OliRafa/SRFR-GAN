from tensorflow.keras import Model
from models.generator import GeneratorNetwork
from models.resnet import ResNet
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU

class SRFR(Model):
    def __init__(
            self,
            num_filters: int = 62,
            depth: int = 50,
            categories: int = 512,
            num_gc: int = 32,
            num_blocks: int = 23,
            residual_scailing: float = 0.2,
            training: bool = True,
        ):
        super(SRFR, self).__init__()
        self._training = training
        self._natural_input = Input(
            name='natural_input'
        )
        self._natural_conv = Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            name='natural_conv',
            activation=LeakyReLU(alpha=0.2),
        )
        self._synthetic_input = Input(
            name='synthetic_input',
        )
        self._synthetic_conv = Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            name='synthetic_conv',
            activation=LeakyReLU(alpha=0.2),
        )
        self._super_resolution = GeneratorNetwork(
            num_filters,
            num_gc,
            num_blocks,
            residual_scailing,
        )
        self._face_recognition = ResNet(
            depth,
            categories,
            training
        )

    def _call_evaluating(self, input_tensor):
        outputs = self._natural_input(input_tensor)
        outputs = self._natural_conv(outputs)
        super_resolution_image = self._super_resolution(outputs)
        embeddings, _ = self._face_recognition(
            super_resolution_image
        )
        return super_resolution_image, embeddings


    def _call_training(self, synthetic_images, natural_images=None):
        synthetic_outputs = self._synthetic_input(synthetic_images)
        synthetic_outputs = self._synthetic_conv(synthetic_outputs)
        synthetic_sr_images = self._super_resolution(
            synthetic_outputs,
        )
        synthetic_embeddings, synthetic_fc_weights = self._face_recognition(
            synthetic_sr_images
        )
        if natural_images:
            natural_outputs = self._natural_input(natural_images)
            natural_outputs = self._natural_conv(natural_outputs)
            natural_sr_images = self._super_resolution(
                natural_outputs,
            )
            natural_embeddings, natural_fc_weights = self._face_recognition(
                natural_sr_images,
            )
            return (
                synthetic_sr_images,
                synthetic_embeddings,
                synthetic_fc_weights,
                natural_sr_images,
                natural_embeddings,
                natural_fc_weights
            )

        return synthetic_sr_images, synthetic_embeddings, synthetic_fc_weights

    def call(self, input_tensor_01, input_tensor_02=None):
        if self._training:
            return self._call_training(input_tensor_01, input_tensor_02)

        return self._call_evaluating(input_tensor_01)
