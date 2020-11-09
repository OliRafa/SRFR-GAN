import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_addons.activations import mish

from models.generator import GeneratorNetwork
from models.resnet import ResNet

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_policy(policy)


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
        input_shape=(28, 28, 3),
        num_classes_syn: int = 2,
        both: bool = False,
        num_classes_nat: int = None,
        scale: int = 64,
    ):
        super(SRFR, self).__init__()
        self._training = training
        self.scale = scale
        if both:
            self._natural_input = Conv2D(
                input_shape=input_shape,
                filters=num_filters,
                kernel_size=(3, 3),
                strides=1,
                padding="same",
                name="natural_input",
                activation=mish,
            )
        self._synthetic_input = Conv2D(
            input_shape=input_shape,
            filters=num_filters,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            name="synthetic_input",
            activation=mish,
        )
        self._super_resolution = GeneratorNetwork(
            num_filters,
            num_gc,
            num_blocks,
            residual_scailing,
        )
        self._face_recognition = ResNet(depth, categories, training, None)
        if self._training:
            if both:
                self._fc_classification_nat = Dense(
                    input_shape=(categories,),
                    units=num_classes_nat,
                    activation=None,
                    use_bias=False,
                    dtype="float32",
                    name="fully_connected_to_softmax_crossentropy_nat",
                )
                self._fc_classification_nat.build(tf.TensorShape([None, 512]))
                self.net_type = "nat"

            self._fc_classification_syn: Dense = Dense(
                input_shape=(categories,),
                units=num_classes_syn,
                activation="softmax",
                use_bias=False,
                dtype="float32",
                name="fully_connected_to_softmax_crossentropy_syn",
            )
            self._fc_classification_syn.build(tf.TensorShape([None, 512]))

    @tf.function
    def _call_evaluating(self, input_tensor, input_type: str = "nat"):
        if input_type == "syn":
            outputs = self._synthetic_input(input_tensor)
        else:
            outputs = self._natural_input(input_tensor)

        super_resolution_image = self._super_resolution(outputs)
        embeddings = self._face_recognition(super_resolution_image)

        # if input_type == "syn":
        #    classification = self._fc_classification_syn(embeddings)
        # else:
        #    classification = self._fc_classification_nat(embeddings)

        return super_resolution_image, embeddings  # , classification

    # def _calculate_normalized_embeddings(self, embeddings, net_type: str = "syn"):
    #    fc_weights = self.get_weights(net_type)
    #    normalized_weights = tf.Variable(
    #        normalize(fc_weights, name="weights_normalization"),
    #        aggregation=tf.VariableAggregation.NONE,
    #    )
    #    normalized_embeddings = (
    #        normalize(embeddings, axis=1, name="embeddings_normalization") * self.scale
    #    )
    #    # replica = tf.distribute.get_replica_context()
    #    # replica.merge_call(self.set_weights,
    #    #                   args=(normalized_weights, net_type))
    #    self.set_weights(normalized_weights, net_type)
    #    return self.call_fc_classification(normalized_embeddings, net_type)

    def _call_training(self, synthetic_images, natural_images=None):
        synthetic_outputs = self._synthetic_input(synthetic_images)
        synthetic_sr_images = self._super_resolution(synthetic_outputs)
        synthetic_embeddings = self._face_recognition(synthetic_sr_images)
        # synthetic_embeddings = self._calculate_normalized_embeddings(
        #    synthetic_embeddings
        # )
        synthetic_classification = self._fc_classification_syn(synthetic_embeddings)
        if natural_images:
            natural_outputs = self._natural_input(natural_images)
            natural_sr_images = self._super_resolution(natural_outputs)
            natural_embeddings = self._face_recognition(natural_sr_images)
            # natural_embeddings = self._calculate_normalized_embeddings(
            #    natural_embeddings
            # )
            natural_classification = self._fc_classification_nat(natural_embeddings)
            return (
                synthetic_sr_images,
                synthetic_embeddings,
                synthetic_classification,
                natural_sr_images,
                natural_embeddings,
                natural_classification,
            )

        return synthetic_sr_images, synthetic_embeddings, synthetic_classification

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

    # def get_weights(self, net_type: str = "syn"):
    #    if net_type == "nat":
    #        return self._fc_classification_nat.get_weights()
    #    return self._fc_classification_syn.get_weights()

    # def set_weights(self, weights, net_type: str = "syn") -> None:
    #    if net_type == "nat":
    #        self._fc_classification_nat.set_weights([weights.read_value()])
    #    else:
    #        self._fc_classification_syn.set_weights([weights.read_value()])

    # def call_fc_classification(self, input, net_type: str = "syn"):
    #    if net_type == "nat":
    #        return self._fc_classification_nat(input)
    #    return self._fc_classification_syn(input)
