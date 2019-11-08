import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    MaxPool2D,
    ReLU
)

#class ConvBlock(Model):
#    def __init__(self):
#        super(ConvBlock, self).__init__()
#        self._conv1 = Conv2D(
#            
#        )
#        self._bn1 = BatchNormalization()
#        self._relu1 = ReLU()
#        self._conv2 = Conv2D(
#
#        )
#        self._bn2 = BatchNormalization()
#
#    def call(self, input):
#        x = self._conv1(input)
#        x = self._bn1(x)
#        return x

class Bottleneck(Model):
    """Model class for the Bottleneck Convolutional Block.

    Arguments:
        filters: List of the filters to be used in the layers.
        stride: Value of the stride to be used in the layers.
    """
    def __init__(self, filters, stride):
        super(Bottleneck, self).__init__()
        self._conv1 = Conv2D(
            filters=filters[0],
            kernel_size=(1, 1),
            strides=stride
        )
        self._bn1 = BatchNormalization()
        self._relu1 = ReLU()
        self._conv2 = Conv2D(
            filters=filters[1],
            kernel_size=(3, 3),
            strides=stride
        )
        self._bn2 = BatchNormalization()
        self._relu2 = ReLU()
        self._conv3 = Conv2D(
            filters=filters[2],
            kernel_size=(1, 1),
            strides=stride
        )
        self._bn3 = BatchNormalization()

    def call(self, input):
        residual = input
        x = self._conv1(input)
        x = self._bn1(x)
        x = self._relu1(x)
        x = self._conv2(input)
        x = self._bn2(x)
        x = self._relu2(x)
        x = self._conv3(input)
        x = self._bn3(x)
        x = Add()([x, residual])
        return ReLU(x)


_resnet_config = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}
layer_config = {
    'conv_2': [64, 64, 256],
    'conv_3': [128, 128, 512],
    'conv_4': [256, 256, 1024],
    'conv_5': [512, 512, 2048]
}

class ResNet(Model):
    """Base Class for the ResNet model.
    Currently only working for the 50, 101 and 152 models, with Bottleneck.

    Arguments:
        depth: Depth input, only 50, 101 or 152 available.
        categories: Number of output classes for the Fully Connected layer.

    Return:
        The ResNet model.
    """
    def __init__(self, depth=50, categories=1000):
        super(ResNet, self).__init__()

        self._input = Conv2D(
            input_shape=(224, 224, 3),
            filters=64,
            kernel_size=(7, 7),
            strides=2,
            activation='relu',
            name='conv_1'
        )
        self._max_pool = MaxPool2D(pool_size=(3, 3), strides=2, name='max_pool')
        #(
        #    self._conv2,
        #    self._conv3,
        #    self._conv4,
        #    self._conv5
        #) = self._generate_layers(
        #        self._resnet_config[depth],
        #        self._layer_config
        #    )
        self._avg_pool = AveragePooling2D(name='avg_pooling')
        self._flatten = Flatten(name='flatten')
        self._fully_connected = Dense(
            units=categories,
            activation='softmax',
            name='fully_connected'
        )

    def _generate_layers(self, layers, filters):
        conv2 = Sequential(name='conv_2')
        conv2.add(Bottleneck(filters['conv_2'], 2))
        for _ in range(1, layers[0]):
            conv2.add(Bottleneck(filters['conv_2'], 1))

        conv3 = Sequential(name='conv_3')
        conv3.add(Bottleneck(filters['conv_3'], 2))
        for _ in range(1, layers[1]):
            conv3.add(Bottleneck(filters['conv_3'], 1))

        conv4 = Sequential(name='conv_4')
        conv4.add(Bottleneck(filters['conv_4'], 2))
        for _ in range(1, layers[1]):
            conv4.add(Bottleneck(filters['conv_4'], 1))

        conv5 = Sequential(name='conv_5')
        conv5.add(Bottleneck(filters['conv_5'], 2))
        for _ in range(1, layers[1]):
            conv5.add(Bottleneck(filters['conv_5'], 1))

        return conv2, conv3, conv4, conv5

    def call(self, input):
        x = self._input(input)
        x = self._max_pool(x)
        #x = self._conv2(x)
        #x = self._conv3(x)
        #x = self._conv4(x)
        #x = self._conv5(x)
        x = self._avg_pool(x)
        x = self._flatten(x)
        x = self._fully_connected(x)
        return x
