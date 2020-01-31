"""This module contains functions used to train the network."""
import tensorflow as tf
from training.losses import compute_arcloss
#from losses import compute_arcloss

@tf.function
def adjust_learning_rate(current_learning_rate: float, epoch: int = 1) -> float:
    """Adjusts learning rate based on the current value and a giving epoch.

    ## Parameters
        current_learning_rate: Current value for the learning rate.
        epoch: Epoch number.

    ## Returns
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

    ## Parameters:
        model:
        images: Batch of images for training.
        classes: Batch of classes to compute the loss.
        num_classes: Total number of classes in the dataset.
        s:
        margin:

    ## Returns:
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

    ## Parameters:
        model:
        dataset:
        num_classes:
        batch_size:
        optimizer:
        train_loss_function:
        scale:
        margin:

    ## Returns:
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
        if step % 200 == 0:
            print('Training loss (for one batch) at step {}: {}'.format(
                step + 1,
                float(loss_value)
            ))
            print('Seen so far: {} samples'.format((step + 1) * batch_size))
    return train_loss_function(loss_value)
