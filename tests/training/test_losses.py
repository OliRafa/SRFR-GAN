import tensorflow as tf

from services.losses import Loss


def instantiate_loss(batch_size, summary_writer, weight, scale, margin):
    return Loss(batch_size, summary_writer, weight, scale, margin)


def test_compute_joint_loss(
    summary_writer,
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
    loss = instantiate_loss(batch_size, summary_writer, weight, scale, margin)
    output = loss.compute_joint_loss(
        super_resolution_images,
        ground_truth_images,
        discriminator_sr_predictions,
        discriminator_gt_predictions,
        synthetic_face_recognition,
        1,
        None,
    )

    assert output == joint_loss / 2


def test__compute_generator_loss(
    summary_writer,
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
    loss = instantiate_loss(batch_size, summary_writer, weight, scale, margin)
    output = loss._compute_generator_loss(
        super_resolution_images,
        ground_truth_images,
        discriminator_sr_predictions,
        discriminator_gt_predictions,
        1,
    )

    assert output == generator_loss / 2


def test__compute_perceptual_loss(
    summary_writer,
    batch_size,
    weight,
    scale,
    margin,
    super_resolution_images,
    ground_truth_images,
    perceptual_loss_distributed,
):
    loss = instantiate_loss(batch_size, summary_writer, weight, scale, margin)
    output = loss._compute_perceptual_loss(
        super_resolution_images,
        ground_truth_images,
    )
    assert output == perceptual_loss_distributed / 2


def test__generator_loss(
    summary_writer,
    batch_size,
    weight,
    scale,
    margin,
    discriminator_sr_predictions,
    discriminator_gt_predictions,
    inner_generator_loss,
):
    loss = instantiate_loss(batch_size, summary_writer, weight, scale, margin)
    output = loss._generator_loss(
        discriminator_sr_predictions, discriminator_gt_predictions
    )

    # output = tf.multiply(output, 2)
    assert output == inner_generator_loss / 2


def test__compute_categorical_crossentropy(
    summary_writer,
    batch_size,
    weight,
    scale,
    margin,
    synthetic_face_recognition,
    categorical_crossentropy,
):
    loss = instantiate_loss(batch_size, summary_writer, weight, scale, margin)
    output = loss._compute_categorical_crossentropy(
        synthetic_face_recognition[1], synthetic_face_recognition[2]
    )

    output = tf.multiply(output, 2)
    assert output == categorical_crossentropy


# def test_compute_l1_loss(
#    weight,
#    scale,
#    margin,
# ):
#    loss = instantiate_loss(weight, scale, margin)
#    print(output)
