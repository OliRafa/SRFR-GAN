"""This module contains functions used for training."""
import logging

import tensorflow as tf
from training.losses import (
    create_vgg_model,
    compute_arcloss,
    compute_discriminator_loss,
    compute_joint_loss,
)

LOGGER = logging.getLogger(__name__)

# Terminar a parte do Discriminator nas duas funções.

def generate_num_epochs(iterations, len_dataset, batch_size):
    LOGGER.info(f' Generating number of epochs for {iterations} iterations,\
 {len_dataset} dataset length and {batch_size} batch size.')
    train_size = tf.math.ceil(len_dataset / batch_size)
    epochs = tf.cast(tf.math.ceil(iterations / train_size), dtype=tf.int32)
    LOGGER.info(f' Number of epochs: {epochs}.')
    return epochs

@tf.function
def adjust_learning_rate(current_learning_rate: float, epoch: int = 1) -> float:
    """Adjusts learning rate based on the current value and a giving epoch.

    ### Parameters
        current_learning_rate: Current value for the learning rate.
        epoch: Epoch number.

    ### Returns
        New value for the learning rate.
    """
    if epoch % 20 == 0:
        return current_learning_rate / 10
    return current_learning_rate

@tf.function
def _train_step(
        model,
        images,
        classes,
        num_classes,
        scale: float,
        margin: float,
    ):
    """Does a training step

    ### Parameters:
        model:
        images: Batch of images for training.
        classes: Batch of classes to compute the loss.
        num_classes: Total number of classes in the dataset.
        s:
        margin:

    ### Returns:
        (loss_value, grads) The loss value and the gradients to be optimized.
    """
    with tf.GradientTape() as tape:
        embeddings, fc_weights = model(images)
        loss_value = compute_arcloss(
            embeddings,
            classes,
            fc_weights,
            num_classes,
            scale,
            margin
        )
    grads = tape.gradient(loss_value, model.trainable_weights)
    return loss_value, grads

@tf.function
def train_model(
        model,
        dataset,
        num_classes,
        batch_size,
        optimizer,
        train_loss_function,
        scale: float,
        margin: float,
    ) -> float:
    """Train the model using the given dataset, compute the loss_function and\
 apply the optimizer.

    ### Parameters:
        model:
        dataset:
        num_classes:
        batch_size:
        optimizer:
        train_loss_function:
        scale:
        margin:

    ### Returns:
        The loss value.
    """
    for step, (image_batch, class_batch) in enumerate(dataset):
        loss_value, grads = _train_step(
            model,
            image_batch,
            class_batch,
            num_classes,
            scale,
            margin
        )
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        if step % 1000 == 0:
            LOGGER.info(
                (
                    f' Training loss (for one batch) at step {step}:'
                    f' {float(loss_value):.3f}'
                )
            )
            LOGGER.info(f' Seen so far: {step * batch_size} samples')
    return train_loss_function(loss_value)

def _train_step_synthetic_only(
        vgg,
        srfr_model,
        sr_discriminator_model,
        low_resolution_batch,
        groud_truth_batch,
        ground_truth_classes,
        num_classes,
        weight: float,
        scale: float,
        margin: float,
        batch_size: int,
        srfr_optimizer,
        discriminator_optimizer,
    ):
    """Does a training step

    ### Parameters:
        model:
        images: Batch of images for training.
        classes: Batch of classes to compute the loss.
        num_classes: Total number of classes in the dataset.
        s:
        margin:

    ### Returns:
        (srfr_loss, srfr_grads, discriminator_loss, discriminator_grads)\
 The loss value and the gradients for SRFR network, as well as the loss value\
 and the gradients for the Discriminative network.
    """
    with tf.GradientTape() as srfr_tape, \
            tf.GradientTape() as discriminator_tape:
        (super_resolution_images, embeddings) = srfr_model(low_resolution_batch)
        fc_weights = srfr_model.get_weights()
        discriminator_sr_predictions = sr_discriminator_model(
            super_resolution_images)
        discriminator_gt_predictions = sr_discriminator_model(
            groud_truth_batch)
        synthetic_face_recognition = (embeddings, ground_truth_classes,
                                      fc_weights, num_classes)
        srfr_loss = compute_joint_loss(
            vgg,
            super_resolution_images,
            groud_truth_batch,
            discriminator_sr_predictions,
            discriminator_gt_predictions,
            synthetic_face_recognition,
            weight=weight,
            scale=scale,
            margin=margin,
        )
        discriminator_loss = compute_discriminator_loss(
            discriminator_sr_predictions,
            discriminator_gt_predictions,
        )
        srfr_loss = tf.reduce_sum(srfr_loss) * (1.0 / batch_size)
        discriminator_loss = tf.reduce_sum(discriminator_loss) * (1.0 / batch_size)

        srfr_scaled_loss = srfr_optimizer.get_scaled_loss(srfr_loss)
        discriminator_scaled_loss = discriminator_optimizer.get_scaled_loss(
            discriminator_loss)

    srfr_grads = srfr_tape.gradient(srfr_scaled_loss,
                                    srfr_model.trainable_weights)
    discriminator_grads = discriminator_tape.gradient(
        discriminator_scaled_loss,
        sr_discriminator_model.trainable_weights,
    )
    return (
        srfr_loss,
        srfr_optimizer.get_unscaled_gradients(srfr_grads),
        discriminator_loss,
        discriminator_optimizer.get_unscaled_gradients(discriminator_grads),
        super_resolution_images,
    )

@tf.function
def _train_step_joint_learn(
        vgg,
        srfr_model,
        sr_discriminator_model,
        natural_batch,
        num_classes_natural,
        synthetic_batch,
        num_classes_synthetic,
        sr_weight: float,
        scale: float,
        margin: float,
    ):
    """Does a training step

    ### Parameters:
        srfr_model:
        sr_discriminator_model:
        natural_batch:
        num_classes_natural: Total number of classes in the natural dataset.
        synthetic_batch:
        num_classes_synthetic: Total number of classes in the synthetic dataset.
        sr_weight: Weight for the SR Loss.
        scale:
        margin:

    ### Returns:
        (srfr_loss, srfr_grads, discriminator_loss, discriminator_grads)\
 The loss value and the gradients for SRFR network, as well as the loss value\
 and the gradients for the Discriminative network.
    """
    natural_images, natural_classes = natural_batch
    synthetic_images, groud_truth_images, synthetic_classes = synthetic_batch
    with tf.GradientTape() as srfr_tape, \
        tf.GradientTape() as discriminator_tape:
        (
            synthetic_sr_images,
            synthetic_embeddings,
            synthetic_fc_weights,
            natural_sr_images,
            natural_embeddings,
            natural_fc_weights
        ) = srfr_model(synthetic_images, natural_images)
        discriminator_sr_predictions = sr_discriminator_model(synthetic_sr_images)
        discriminator_gt_predictions = sr_discriminator_model(groud_truth_images)
        synthetic_face_recognition = (
            synthetic_embeddings,
            synthetic_classes,
            synthetic_fc_weights,
            num_classes_synthetic,
        )
        natural_face_recognition = (
            natural_embeddings,
            natural_classes,
            natural_fc_weights,
            num_classes_natural,
        )
        srfr_loss = compute_joint_loss(
            vgg,
            synthetic_images,
            groud_truth_images,
            discriminator_sr_predictions,
            discriminator_gt_predictions,
            synthetic_face_recognition,
            natural_face_recognition,
            sr_weight,
            scale,
            margin
        )
        discriminator_loss = compute_discriminator_loss(
            discriminator_sr_predictions,
            discriminator_gt_predictions,
        )
    srfr_grads = srfr_tape.gradient(srfr_loss, srfr_model.trainable_weights)
    discriminator_grads = discriminator_tape.gradient(
        discriminator_loss,
        sr_discriminator_model.trainable_weights,
    )
    return srfr_loss, srfr_grads, discriminator_loss, discriminator_grads

@tf.function
def _train_with_natural_images(
        vgg,
        srfr_model,
        discriminator_model,
        batch_size,
        srfr_optimizer,
        discriminator_optimizer,
        train_loss_function,
        synthetic_dataset,
        num_classes_synthetic: int,
        natural_dataset,
        num_classes_natural: int,
        sr_weight: float = 0.1,
        scale: float = 64,
        margin: float = 0.5,
    ) -> float:
    for step, (natural_batch, synthetic_batch) in enumerate(
                zip(natural_dataset, synthetic_dataset)
        ):
        (
            srfr_loss,
            srfr_grads,
            discriminator_loss,
            discriminator_grads,
        ) = _train_step_joint_learn(
            vgg,
            srfr_model,
            discriminator_model,
            natural_batch,
            num_classes_natural,
            synthetic_batch,
            num_classes_synthetic,
            sr_weight,
            scale,
            margin,
        )
        srfr_optimizer.apply_gradients(
            zip(srfr_grads, srfr_model.trainable_weights)
        )
        discriminator_optimizer.apply_gradients(
            zip(discriminator_grads, discriminator_model.trainable_weights)
        )
        if step % 1000 == 0:
            LOGGER.info(
                (
                    f' SRFR Training loss (for one batch) at step {step}:'
                    f' {float(srfr_loss):.3f}'
                )
            )
            LOGGER.info(
                (
                    f' Discriminator loss (for one batch) at step {step}:'
                    f' {float(discriminator_loss):.3f}'
                )
            )
            LOGGER.info(f' Seen so far: {step * batch_size} samples')
    return (
        train_loss_function(srfr_loss),
        train_loss_function(discriminator_loss),
    )

#@tf.function
def _train_with_synthetic_images_only(
        vgg,
        srfr_model,
        discriminator_model,
        batch_size,
        srfr_optimizer,
        discriminator_optimizer,
        train_loss_function,
        strategy,
        dataset,
        num_classes: int,
        sr_weight: float = 0.1,
        scale: float = 64,
        margin: float = 0.5,
        summary_writer=None,
    ) -> float:
    srfr_losses = []
    discriminator_losses = []
    for step, (synthetic_images, groud_truth_images, synthetic_classes) in enumerate(dataset):
        (srfr_loss, srfr_grads, discriminator_loss, discriminator_grads,
         super_resolution_images) = strategy.experimental_run_v2(_train_step_synthetic_only,
                                                 args=(
                                                     vgg,
                                                     srfr_model,
                                                     discriminator_model,
                                                     synthetic_images,
                                                     groud_truth_images,
                                                     synthetic_classes,
                                                     num_classes,
                                                     sr_weight,
                                                     scale,
                                                     margin,
                                                     batch_size,
                                                 ),
        )
        srfr_optimizer.apply_gradients(
            zip(srfr_grads, srfr_model.trainable_weights)
        )
        discriminator_optimizer.apply_gradients(
            zip(discriminator_grads, discriminator_model.trainable_weights)
        )
        srfr_losses.append(srfr_loss)
        discriminator_losses.append(discriminator_loss)
        if step % 1000 == 0:
            LOGGER.info(
                (
                    f' SRFR Training loss (for one batch) at step {step}:'
                    f' {float(srfr_loss):.3f}'
                )
            )
            LOGGER.info(
                (
                    f' Discriminator loss (for one batch) at step {step}:'
                    f' {float(discriminator_loss):.3f}'
                )
            )
            if step == 0:
                LOGGER.info(f' Seen so far: {batch_size} samples')
            else:
                LOGGER.info(f' Seen so far: {step * batch_size} samples')
            with summary_writer.as_default():
                tf.summary.scalar(
                    'srfr_loss_per_batch',
                    float(srfr_loss),
                    step=step,
                )
                tf.summary.scalar(
                    'discriminator_loss_per_batch',
                    float(discriminator_loss),
                    step=step,
                )
                tf.summary.image(
                    'lr_images',
                    synthetic_images,
                    max_outputs=10,
                    step=step
                )
                tf.summary.image(
                    'hr_images',
                    groud_truth_images,
                    max_outputs=10,
                    step=step
                )
                tf.summary.image(
                    'sr_images',
                    super_resolution_images,
                    max_outputs=10,
                    step=step
                )
    return (
        train_loss_function(srfr_losses),
        train_loss_function(discriminator_losses),
    )

#@tf.function
def train_srfr_model(
        strategy,
        srfr_model,
        discriminator_model,
        batch_size,
        srfr_optimizer,
        discriminator_optimizer,
        train_loss_function,
        synthetic_dataset,
        num_classes_synthetic: int,
        natural_dataset=None,
        num_classes_natural: int = None,
        sr_weight: float = 0.1,
        scale: float = 64,
        margin: float = 0.5,
        summary_writer=None,
    ) -> float:
    """Train the model using the given dataset, compute the loss_function and\
 apply the optimizer.

    ### Parameters:
        srfr_model: The Super Resolution Face Recognition model.
        sr_discriminator_model: The Discriminator model.
        batch_size: The Batch size.
        srfr_optimizer: Optimizer for the SRFR network.
        discriminator_optimizer: Optimizer for the Discriminator network.
        train_loss_function:
        sr_weight: Weight for the SR Loss.
        scale:
        margin:
        synthetic_dataset:
        num_classes_synthetic:
        natural_dataset:
        num_classes_natural:

    ### Returns:
        (srfr_loss, discriminator_loss) The loss value for SRFR and\
 Discriminator networks.
    """
    vgg = create_vgg_model()
    if natural_dataset:
        return _train_with_natural_images(
            vgg,
            srfr_model,
            discriminator_model,
            batch_size,
            srfr_optimizer,
            discriminator_optimizer,
            train_loss_function,
            synthetic_dataset,
            num_classes_synthetic,
            natural_dataset,
            num_classes_natural,
            sr_weight,
            scale,
            margin
        )

    return _train_with_synthetic_images_only(
        vgg,
        srfr_model,
        discriminator_model,
        batch_size,
        srfr_optimizer,
        discriminator_optimizer,
        train_loss_function,
        strategy,
        synthetic_dataset,
        num_classes_synthetic,
        sr_weight,
        scale,
        margin,
        summary_writer,
    )
