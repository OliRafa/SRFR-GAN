import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Add,
    Conv2D,
    Dense,
    Flatten,
    LeakyReLU,
    UpSampling2D
)

class ResidualDenseBlock(Model):
    """.

    # Arguments:
        filters: List of the filters to be used in the layers.
        stride: Value of the stride to be used in the layers.
        first_conv_layer: If it's the first layer of the block to be created,
        which implies that a downsampler conv layer has to be used on the input.
    """
    def __init__(self, filters=64, gc=32, scaling_parameter=0.2):
        super(ResidualDenseBlock, self).__init__()
        self._scaling_parameter = scaling_parameter
        self._conv1 = Conv2D(
            filters=gc,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=LeakyReLU(alpha=0.2)
        )
        self._conv2 = Conv2D(
            filters=gc,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=LeakyReLU(alpha=0.2)
        )
        self._conv3 = Conv2D(
            filters=gc,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=LeakyReLU(alpha=0.2)
        )
        self._conv4 = Conv2D(
            filters=gc,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=LeakyReLU(alpha=0.2)
        )
        self._conv5 = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same'
        )

    def call(self, input_tensor):
        x1 = self._conv1(input_tensor)
        x2 = self._conv2(tf.concat([input_tensor, x1], 1))
        x3 = self._conv3(tf.concat([input_tensor, x1, x2], 1))
        x4 = self._conv4(tf.concat([input_tensor, x1, x2, x3], 1))
        x5 = self._conv5(tf.concat([input_tensor, x1, x2, x3, x4], 1))
        return x5 * self._scaling_parameter + input_tensor

class RRDB(Model):
    def __init__(self, filters=64, gc=32, scaling_parameter=0.2):
        super(RRDB, self).__init__()
        self._scaling_parameter = scaling_parameter
        self._rdb_1 = ResidualDenseBlock(filters, gc, scaling_parameter)
        self._rdb_2 = ResidualDenseBlock(filters, gc, scaling_parameter)
        self._rdb_3 = ResidualDenseBlock(filters, gc, scaling_parameter)

    def call(self, input_tensor):
        x = self._rdb_1(input_tensor)
        x = self._rdb_2(x)
        x = self._rdb_3(x)

        return x * self._scaling_parameter + input_tensor

class GeneratorNetwork(Model):
    """.

    # Arguments:
        depth: Depth input, only 50, 101 or 152 available.
        categories: Number of output classes for the Fully Connected layer.

    # Return:
        .
    """
    def __init__(
            self,
            num_filters=62,
            num_gc=32,
            num_blocks=23,
            scaling_parameter=0.2
        ):
        super(GeneratorNetwork, self).__init__()
        self._input = Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            name='conv_input'
        )
        self._rrdb_block = self._generate_layers(
            num_blocks,
            num_filters,
            num_gc,
            scaling_parameter
        )
        self._conv_1 = Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            name='conv_after_rrdb_block'
        )
        self._upsampling_1 = UpSampling2D(size=(2, 2), interpolation='nearest')
        self._upsampling_conv_1 = Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            name='conv_upsampling_1',
            activation=LeakyReLU(alpha=0.2)
        )
        self._upsampling_2 = UpSampling2D(size=(2, 2), interpolation='nearest')
        self._upsampling_conv_2 = Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            name='conv_upsampling_2',
            activation=LeakyReLU(alpha=0.2)
        )
        self._high_resolution_conv = Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            name='conv_high_resolution',
            activation=LeakyReLU(alpha=0.2)
        )
        self._last_conv = Conv2D(
            filters=3,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            name='conv_last'
        )

    def _generate_layers(self, layers, filters, gc, scaling_parameter):
        blocks = Sequential(name='rrdb_blocks')
        for _ in range(1, layers[0]):
            blocks.add(RRDB(filters, gc, scaling_parameter))

        return blocks

    def call(self, input_tensor):
        fea = self._input(input_tensor)
        trunk = self._rrdb_block(fea)
        trunk = self._conv_1(trunk)
        fea = Add()([fea, trunk])

        fea = self._upsampling_1(fea)
        fea = self._upsampling_conv_1(fea)
        fea = self._upsampling_2(fea)
        fea = self._upsampling_conv_2(fea)

        fea = self._high_resolution_conv(fea)
        fea = self._last_conv(fea)

        return fea
