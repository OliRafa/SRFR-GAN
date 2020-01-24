import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    LeakyReLU,
    UpSampling2D
)

class BasicBlock(Model):
    """.

    # Arguments:
        filters: List of the filters to be used in the layers.
        stride: Value of the stride to be used in the layers.
        first_conv_layer: If it's the first layer of the block to be created,
        which implies that a downsampler conv layer has to be used on the input.
    """
    def __init__(self, filters=64, block_number=0):
        super(BasicBlock, self).__init__()
        self._conv_1 = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            name='_conv_0{}_01'.format(block_number)
        )
        self._leaky_relu = LeakyReLU(alpha=0.2)
        self._batch_normalization_1 = BatchNormalization(
            name='_bn_0{}_01'.format(block_number)
        )
        self._conv_2 = Conv2D(
            filters=filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name='_conv_0{}_02'.format(block_number)
        )
        self._batch_normalization_2 = BatchNormalization(
            name='_bn_0{}_02'.format(block_number)
        )

    def call(self, input_tensor):
        x = self._conv_1(input_tensor)
        x = self._leaky_relu(x)
        x = self._batch_normalization_1(x)
        x = self._conv_2(x)
        x = self._batch_normalization_2(x)
        return self._leaky_relu(x)

class DiscriminatorNetwork(Model):
    """.

    # Arguments:
        depth: Depth input, only 50, 101 or 152 available.
        categories: Number of output classes for the Fully Connected layer.

    # Return:
        .
    """
    def __init__(self):
        super(DiscriminatorNetwork, self).__init__()
        self._conv_1 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=LeakyReLU(alpha=0.2),
            name='_conv_01_01'
        )
        self._conv_2 = Conv2D(
            filters=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation=LeakyReLU(alpha=0.2),
            name='_conv_01_02'
        )
        self._batch_normalization = BatchNormalization(
            name='_bn_01'
        )
        self._leaky_relu = LeakyReLU(alpha=0.2)
        self._block_2 = BasicBlock(128, 2)
        self._block_3 = BasicBlock(256, 3)
        self._block_4 = BasicBlock(512, 4)
        self._flatten = Flatten(name='flatten')
        self._fully_connected_1 = Dense(
            1024,
            activation=LeakyReLU(alpha=0.2),
            name='fully_connected_01'
        )
        self._fully_connected_2 = Dense(
            1,
            activation='sigmoid',
            name='fully_connected_02'
        )

    def call(self, input_tensor):
        x = self._conv_1(input_tensor)
        x = self._conv_2(x)
        x = self._batch_normalization(x)
        x = self._leaky_relu(x)
        x = self._block_2(x)
        x = self._block_3(x)
        x = self._block_4(x)
        x = self._flatten(x)
        x = self._fully_connected_1(x)
        return self._fully_connected_2(x)
