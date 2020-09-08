import tensorflow as tf

from training.losses import Loss


def instantiate_loss(batch_size, weight, scale, margin):
    return Loss(None, batch_size, None, weight, scale, margin)


def test_compute_joint_loss(
    batch_size,
    weight,
    scale,
    margin,
    super_resolution_images,
    ground_truth_images,
    discriminator_sr_predictions,
    discriminator_gt_predictions,
    synthetic_face_recognition,
    joint_loss,
):
    loss = instantiate_loss(batch_size, weight, scale, margin)
    output = loss.compute_joint_loss(
        super_resolution_images,
        ground_truth_images,
        discriminator_sr_predictions,
        discriminator_gt_predictions,
        synthetic_face_recognition,
        None,
    )

    # output = tf.multiply(output, 2)
    assert output == joint_loss


def test__compute_generator_loss(
    batch_size,
    weight,
    scale,
    margin,
    super_resolution_images,
    ground_truth_images,
    discriminator_sr_predictions,
    discriminator_gt_predictions,
    generator_loss,
):
    loss = instantiate_loss(batch_size, weight, scale, margin)
    output = loss._compute_generator_loss(
        super_resolution_images,
        ground_truth_images,
        discriminator_sr_predictions,
        discriminator_gt_predictions,
        None,
    )

    # output = tf.multiply(output, 2)
    assert output == generator_loss


def test__compute_perceptual_loss(
    batch_size,
    weight,
    scale,
    margin,
    super_resolution_images,
    ground_truth_images,
    perceptual_loss,
):
    loss = instantiate_loss(batch_size, weight, scale, margin)
    output = loss._compute_perceptual_loss(
        super_resolution_images,
        ground_truth_images,
    )
    assert output == perceptual_loss


def test__generator_loss(
    batch_size,
    weight,
    scale,
    margin,
    discriminator_sr_predictions,
    discriminator_gt_predictions,
    inner_generator_loss,
):
    loss = instantiate_loss(batch_size, weight, scale, margin)
    output = loss._generator_loss(
        discriminator_sr_predictions, discriminator_gt_predictions
    )

    # output = tf.multiply(output, 2)
    assert output == inner_generator_loss


def test__compute_categorical_crossentropy(
    batch_size,
    weight,
    scale,
    margin,
    synthetic_face_recognition,
    categorical_crossentropy,
):
    loss = instantiate_loss(batch_size, weight, scale, margin)
    output = loss._compute_categorical_crossentropy(
        synthetic_face_recognition[1], synthetic_face_recognition[2]
    )

    # output = tf.multiply(output, 2)
    assert output == categorical_crossentropy


# def test_compute_l1_loss(
#    weight,
#    scale,
#    margin,
# ):
#    loss = instantiate_loss(weight, scale, margin)
#    print(output)
