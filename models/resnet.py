"""ResNet Model.
Only the ResNet50, Resnet101 and ResNet152 were implemented."""
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPool2D,
    ReLU,
    ZeroPadding2D
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

class Shortcut(Model):
    def __init__(self, filter, stride, trainable=False):
        super(Shortcut, self).__init__()
        self._trainable = trainable
        self._conv = Conv2D(
            filters=filter,
            kernel_size=(1, 1),
            strides=stride
        )
        self._bn = BatchNormalization(trainable=self._trainable)

    def call(self, input_tensor):
        output = self._conv(input_tensor)
        return self._bn(output)

class Bottleneck(Model):
    """Model class for the Bottleneck Convolutional Block.

    # Arguments:
        filters: List of the filters to be used in the layers.
        stride: Value of the stride to be used in the layers.
        first_conv_layer: If it's the first layer of the block to be created,\
 which implies that a downsampler conv layer has to be used on the input.
    """
    def __init__(self, filters, stride, first_conv_layer=False, trainable=False):
        super(Bottleneck, self).__init__()
        if first_conv_layer:
            self._sc_layer = True
            self._sc_filter = filters[2]
            self._sc_stride = stride
            self._padding = 'same'
        else:
            self._sc_layer = False
            self._padding = 'valid'

        self._trainable = trainable

        self._conv1 = Conv2D(
            filters=filters[0],
            kernel_size=(1, 1),
            strides=(1, 1)
        )
        self._bn1 = BatchNormalization(trainable=self._trainable)
        self._relu1 = ReLU()
        self._conv2 = Conv2D(
            filters=filters[1],
            kernel_size=(3, 3),
            strides=stride,
            padding='same'
        )
        self._bn2 = BatchNormalization(trainable=self._trainable)
        self._relu2 = ReLU()
        self._conv3 = Conv2D(
            filters=filters[2],
            kernel_size=(1, 1),
            strides=(1, 1)
        )
        self._bn3 = BatchNormalization(trainable=self._trainable)

    def call(self, input_tensor):
        if self._sc_layer:
            residual = Shortcut(
                self._sc_filter,
                self._sc_stride,
                self._trainable
            )(input_tensor)
        else:
            residual = input_tensor

        output = self._conv1(input_tensor)
        output = self._bn1(output)
        output = self._relu1(output)
        output = self._conv2(output)
        output = self._bn2(output)
        output = self._relu2(output)
        output = self._conv3(output)
        output = self._bn3(output)
        output = Add()([output, residual])
        return ReLU()(output)


RESNET_CONFIG = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}
LAYER_CONFIG = {
    'conv_2': [64, 64, 256],
    'conv_3': [128, 128, 512],
    'conv_4': [256, 256, 1024],
    'conv_5': [512, 512, 2048]
}

class ResNet(Model):
    """Base Class for the ResNet model.
    Currently only working for the 50, 101 and 152 models, with Bottleneck.

    # Arguments:
        depth: Depth input, only 50, 101 or 152 available.
        categories: Number of output classes for the Fully Connected layer.

    # Return:
        The ResNet model.
    """
    def __init__(self, depth=50, categories=512, trainable=False):
        super(ResNet, self).__init__()
        global RESNET_CONFIG, LAYER_CONFIG

        self._trainable = trainable

        self._input = Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=2,
            activation='relu',
            padding='valid',
            name='conv_1'
        )
        self._max_pool = MaxPool2D(pool_size=(3, 3), strides=2, name='max_pool')
        (
            self._conv2,
            self._conv3,
            self._conv4,
            self._conv5
        ) = self._generate_layers(
                RESNET_CONFIG[depth],
                LAYER_CONFIG
            )
        #self._avg_pool = AveragePooling2D((1, 1), name='avg_pooling')
        self._flatten = Flatten(name='flatten')
        self._bn = BatchNormalization(
            momentum=0.9,
            epsilon=2e-05,
            trainable=self._trainable
        )
        self._dropout = Dropout(rate=0.4)
        self._fully_connected = Dense(
            units=categories,
            activation='softmax',
            name='fully_connected'
        )
        self._bn2 = BatchNormalization(
            momentum=0.9,
            epsilon=2e-05,
            trainable=self._trainable
        )

    def _generate_layers(self, layers, filters):
        conv2 = Sequential(name='conv_2')
        conv2.add(Bottleneck(filters['conv_2'], 2, True, self._trainable))
        for _ in range(1, layers[0]):
            conv2.add(Bottleneck(filters['conv_2'], 1, self._trainable))

        conv3 = Sequential(name='conv_3')
        conv3.add(Bottleneck(filters['conv_3'], 2, True, self._trainable))
        for _ in range(1, layers[1]):
            conv3.add(Bottleneck(filters['conv_3'], 1, self._trainable))

        conv4 = Sequential(name='conv_4')
        conv4.add(Bottleneck(filters['conv_4'], 2, True, self._trainable))
        for _ in range(1, layers[1]):
            conv4.add(Bottleneck(filters['conv_4'], 1, self._trainable))

        conv5 = Sequential(name='conv_5')
        conv5.add(Bottleneck(filters['conv_5'], 2, True, self._trainable))
        for _ in range(1, layers[1]):
            conv5.add(Bottleneck(filters['conv_5'], 1, self._trainable))

        return conv2, conv3, conv4, conv5

    def call(self, input_tensor):
        output = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_tensor)
        output = self._input(output)
        output = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(output)
        output = self._max_pool(output)
        output = self._conv2(output)
        output = self._conv3(output)
        output = self._conv4(output)
        output = self._conv5(output)
        output = self._bn(output)
        output = self._dropout(output)
        output = self._flatten(output)
        output = self._fully_connected(output)
        output = self._bn2(output)
        return output, self._bn2.get_weights()
