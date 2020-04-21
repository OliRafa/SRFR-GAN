from tensorflow import keras
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Activation


def create_vgg_model():
    vgg = VGG19(include_top=False, weights='imagenet')
    #weights = vgg.get_weights()

    # Removing the activation function from the last conv layer 'block5_conv4'
    vgg.layers[-2].activation = None

    vgg_output = vgg.input
    for layer in vgg.layers[1:-1]:
        vgg_output = layer(vgg_output)

    # Casting from float16 to float32 for MixedPrecisionPolicy
    vgg_output = Activation('linear', dtype='float32')(vgg_output)

    # Creating the model with layers [input, ..., last_conv_layer]
    # I.e. removing the last MaxPooling layer from the model
    model = keras.Model(
        inputs=vgg.input,
        outputs=vgg_output,
    )
    model.trainable = False
    return model
