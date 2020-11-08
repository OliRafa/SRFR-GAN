import datetime
import json
import os
from pathlib import Path

import pytest

# from base_training import _create_checkpoint_and_manager, _instantiate_models
# from services.losses import Loss
# from services.train import Train

# from utils.input_data import VggFace2, parseConfigsFile

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # isort:skip
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # isort:skip
import tensorflow as tf  # isort:skip


mocks_path = Path.cwd().joinpath("tests", "mocks")


def transform_to_tf_tensor(value, dtype=tf.float32):
    return tf.constant(value, dtype=dtype)


@pytest.fixture
def batch_size():
    # 4 is the Bathc Size Per GPU, and it's multiplied by 2 GPUs
    return 4 * 2


@pytest.fixture
def softmax_real_input():
    with mocks_path.joinpath("softmax_inputs.json").open("r") as obj:
        return transform_to_tf_tensor(json.load(obj))


@pytest.fixture
def softmax_real_output():
    with mocks_path.joinpath("softmax_output.json").open("r") as obj:
        return transform_to_tf_tensor(json.load(obj))


@pytest.fixture
def softmax_input_minimal():
    return transform_to_tf_tensor([[0.1, 0.2, 0.7]])


@pytest.fixture
def zeros_array():
    return transform_to_tf_tensor([[0, 0, 0]])


@pytest.fixture
def l1_loss_output():
    return transform_to_tf_tensor(3.5173749923706055)


@pytest.fixture
def softmax_zeros_array_output():
    return transform_to_tf_tensor(
        [[0.3333333432674408, 0.3333333432674408, 0.3333333432674408]]
    )


@pytest.fixture
def super_resolution_images():
    with mocks_path.joinpath("_step_function", "super_resolution_images.json").open(
        "r"
    ) as obj:
        return transform_to_tf_tensor(json.load(obj))


@pytest.fixture
def weight():
    return transform_to_tf_tensor(0.0010000000474974513)


@pytest.fixture
def scale():
    return transform_to_tf_tensor(64.0)


@pytest.fixture
def margin():
    return transform_to_tf_tensor(0.5)


@pytest.fixture
def perceptual_loss():
    return transform_to_tf_tensor(0.005082819145172834)


@pytest.fixture
def perceptual_loss_distributed():
    return transform_to_tf_tensor(0.005082819610834122)


@pytest.fixture
def generator_loss():
    return transform_to_tf_tensor(0.8455883264541626)
    # return transform_to_tf_tensor(0.7095396518707275)


@pytest.fixture
def inner_generator_loss():
    return transform_to_tf_tensor(0.836988091468811)
    # return transform_to_tf_tensor(0.700939416885376)


@pytest.fixture
def categorical_crossentropy():
    return transform_to_tf_tensor(8.795759201049805)


@pytest.fixture
def joint_loss():
    return transform_to_tf_tensor(8.796605110168457)
    # return transform_to_tf_tensor(8.796468734741211)


@pytest.fixture
def ground_truth_images():
    with mocks_path.joinpath("_step_function", "groud_truth_batch.json").open(
        "r"
    ) as obj:
        return transform_to_tf_tensor(json.load(obj))


@pytest.fixture
def discriminator_sr_predictions():
    with mocks_path.joinpath(
        "_step_function", "discriminator_sr_predictions.json"
    ).open("r") as obj:
        return transform_to_tf_tensor(json.load(obj))


@pytest.fixture
def discriminator_gt_predictions():
    with mocks_path.joinpath(
        "_step_function", "discriminator_gt_predictions.json"
    ).open("r") as obj:
        return transform_to_tf_tensor(json.load(obj))


@pytest.fixture
def vgg_output_fakes():
    with mocks_path.joinpath("_compute_perceptual_loss", "fake.json").open("r") as obj:
        return transform_to_tf_tensor(json.load(obj))


@pytest.fixture
def vgg_output_reals():
    with mocks_path.joinpath("_compute_perceptual_loss", "real.json").open("r") as obj:
        return transform_to_tf_tensor(json.load(obj))


@pytest.fixture
def vgg_euclidian_distance_output():
    return transform_to_tf_tensor(5.08281946182251)


@pytest.fixture
def synthetic_face_recognition():
    with mocks_path.joinpath("_step_function", "embeddings.json").open(
        "r"
    ) as embeddings:
        embeddings = transform_to_tf_tensor(json.load(embeddings))

    with mocks_path.joinpath("_step_function", "predictions.json").open(
        "r"
    ) as predictions:
        predictions = transform_to_tf_tensor(json.load(predictions))

    with mocks_path.joinpath("_step_function", "ground_truth_classes.json").open(
        "r"
    ) as ground_truth_classes:
        ground_truth_classes = json.load(ground_truth_classes)

    num_classes = 8529

    return (embeddings, predictions, ground_truth_classes, num_classes)


@pytest.fixture
def binary_crossentropy_zeros_like_input():
    with mocks_path.joinpath("_generator_loss", "zeros_like.json").open("r") as obj:
        return transform_to_tf_tensor(json.load(obj))


@pytest.fixture
def binary_crossentropy_softmax_input():
    with mocks_path.joinpath("_generator_loss", "softmax.json").open("r") as obj:
        return transform_to_tf_tensor(json.load(obj))


@pytest.fixture
def binary_crossentropy_output():
    return transform_to_tf_tensor(0.2876819372177124)
    # return transform_to_tf_tensor(0.8259393572807312)


@pytest.fixture
def binary_crossentropy_ones_like_input():
    with mocks_path.joinpath("_generator_loss", "ones_like.json").open("r") as obj:
        return transform_to_tf_tensor(json.load(obj))


@pytest.fixture
def binary_crossentropy_softmax_input2():
    with mocks_path.joinpath("_generator_loss", "sr_softmax.json").open("r") as obj:
        return transform_to_tf_tensor(json.load(obj))


@pytest.fixture
def binary_crossentropy_output2():
    return transform_to_tf_tensor(1.3862942457199097)


@pytest.fixture
def summary_writer():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.summary.create_file_writer(
        str(Path.cwd().joinpath("tests", "logs_unit_test", current_time, "train")),
    )


# @pytest.fixture
# def instantiate_training(summary_writer) -> Train:
#    strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
#    network_settings, train_settings, preprocess_settings = parseConfigsFile(
#        ["network", "train", "preprocess"]
#    )
#    BATCH_SIZE = train_settings["batch_size"] * strategy.num_replicas_in_sync
#    synthetic_num_classes = VggFace2(mode="concatenated").get_number_of_classes()
#    (
#        srfr_model,
#        discriminator_model,
#        srfr_optimizer,
#        discriminator_optimizer,
#    ) = _instantiate_models(
#        strategy,
#        network_settings,
#        train_settings,
#        preprocess_settings,
#        synthetic_num_classes,
#    )
#    checkpoint, manager = _create_checkpoint_and_manager(
#        srfr_model, discriminator_model, srfr_optimizer, discriminator_optimizer
#    )
#    loss = Loss(
#        BATCH_SIZE,
#        summary_writer,
#        train_settings["super_resolution_weight"],
#        train_settings["scale"],
#        train_settings["angular_margin"],
#    )
#    # TrainModelUseCase(
#    #    strategy, loss, summary_writer, None, None, checkpoint, manager
#    # )._try_restore_checkpoint()
#    return Train(
#        strategy,
#        srfr_model,
#        srfr_optimizer,
#        discriminator_model,
#        discriminator_optimizer,
#        summary_writer,
#        checkpoint,
#        manager,
#        loss,
#    )


@pytest.fixture
def train_01():
    with mocks_path.joinpath("services", "train", "syn_images_02.json").open(
        "r"
    ) as obj:
        low_resolution_batch = transform_to_tf_tensor(json.load(obj))

    with mocks_path.joinpath("services", "train", "groud_truth_images_02.json").open(
        "r"
    ) as obj:
        groud_truth_images = transform_to_tf_tensor(json.load(obj))

    with mocks_path.joinpath("services", "train", "syn_classes_02.json").open(
        "r"
    ) as obj:
        syn_classes = transform_to_tf_tensor(json.load(obj))

    step = 17002

    return low_resolution_batch, groud_truth_images, syn_classes, step
