import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Add,
    Conv2D,
    UpSampling2D
)
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_addons.activations import mish

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


class ResidualDenseBlock(Model):
    """.

    ### Parameters:
        filters: Number of output filters for each convolutional layer.
        gc:
        residual_scailing: Scailing parameter for each residual concatenation.
    """
    def __init__(
            self,
            filters: int = 64,
            gc: int = 32,
            residual_scailing: float = 0.2
        ):
        super(ResidualDenseBlock, self).__init__()
        self._residual_scailing = residual_scailing
        self._conv1 = Conv2D(
            filters=gc,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=mish,
            kernel_initializer='he_uniform',
        )
        self._conv2 = Conv2D(
            filters=gc,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=mish,
            kernel_initializer='he_uniform',
        )
        self._conv3 = Conv2D(
            filters=gc,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=mish,
            kernel_initializer='he_uniform',
        )
        self._conv4 = Conv2D(
            filters=gc,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=mish,
            kernel_initializer='he_uniform',
        )
        self._conv5 = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer='he_uniform',
        )

    def call(self, input_tensor):
        output1 = self._conv1(input_tensor)
        output2 = self._conv2(tf.concat([input_tensor, output1], -1))
        output3 = self._conv3(tf.concat([input_tensor, output1, output2], -1))
        output4 = self._conv4(tf.concat(
            [input_tensor, output1, output2, output3],
            -1,
        ))
        output5 = self._conv5(tf.concat(
            [input_tensor, output1, output2, output3, output4],
            -1,
        ))
        return output5 * self._residual_scailing + input_tensor


class RRDB(Model):
    """.

    ### Parameters:
        filters: Number of output filters for each convolutional layer.
        gc:
        residual_scailing: Scailing parameter for each residual concatenation.
    """
    def __init__(
            self,
            filters: int = 64,
            gc: int = 32,
            residual_scailing: float = 0.2
        ):
        super(RRDB, self).__init__()
        self._residual_scailing = residual_scailing
        self._rdb_1 = ResidualDenseBlock(filters, gc, residual_scailing)
        self._rdb_2 = ResidualDenseBlock(filters, gc, residual_scailing)
        self._rdb_3 = ResidualDenseBlock(filters, gc, residual_scailing)

    def call(self, input_tensor):
        output = self._rdb_1(input_tensor)
        output = self._rdb_2(output)
        output = self._rdb_3(output)

        return output * self._residual_scailing + input_tensor


class GeneratorNetwork(Model):
    """.

    ### Parameters:
        num_filters: Number of output filters for each convolutional layer.
        num_gc:
        num_blocks: Number of RRDB blocks.
        residual_scailing: Scailing parameter for each residual concatenation.
    """
    def __init__(
            self,
            num_filters: int = 62,
            num_gc: int = 32,
            num_blocks: int = 23,
            residual_scailing: float = 0.2,
        ):
        super(GeneratorNetwork, self).__init__()
        self._rrdb_block = self._generate_layers(
            num_blocks,
            num_filters,
            num_gc,
            residual_scailing,
        )
        self._conv_1 = Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            name='conv_after_rrdb_block',
            kernel_initializer='he_uniform',
        )
        self._upsampling_1 = UpSampling2D(size=(2, 2), interpolation='nearest')
        self._upsampling_conv_1 = Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            name='conv_upsampling_1',
            activation=mish,
            kernel_initializer='he_uniform',
        )
        self._upsampling_2 = UpSampling2D(size=(2, 2), interpolation='nearest')
        self._upsampling_conv_2 = Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            name='conv_upsampling_2',
            activation=mish,
            kernel_initializer='he_uniform',
        )
        self._high_resolution_conv = Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            name='conv_high_resolution',
            activation=mish,
            kernel_initializer='he_uniform',
        )
        self._last_conv = Conv2D(
            filters=3,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            name='conv_last',
            kernel_initializer='he_uniform',
            dtype='float32',
        )

    def _generate_layers(
            self,
            num_blocks: int,
            filters: int,
            gc: int,
            residual_scailing: float,
        ):
        """Generate a number of RRDB convolutional blocks.

        ### Parameters
            num_blocks: Number of RRDB blocks.
            filters: Number of output filters for each convolutional layer.
            gc:
            scailing_parameter: Scailing parameter for each residual\
 concatenation.

        ### Returns
            A Sequential of size num_blocks * RRDB.
        """
        blocks = Sequential(name='rrdb_blocks')
        for _ in range(1, num_blocks):
            blocks.add(RRDB(filters, gc, residual_scailing))

        return blocks

    def call(self, input_tensor):
        trunk = self._rrdb_block(input_tensor)
        trunk = self._conv_1(trunk)
        fea = Add()([input_tensor, trunk])

        fea = self._upsampling_1(fea)
        fea = self._upsampling_conv_1(fea)
        fea = self._upsampling_2(fea)
        fea = self._upsampling_conv_2(fea)

        fea = self._high_resolution_conv(fea)
        return self._last_conv(fea)
