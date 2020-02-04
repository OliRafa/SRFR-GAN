"""Main training module for the Joint Learning Super Resolution Face\
 Recognition.
"""
import datetime
import logging

import tensorflow as tf
from tensorflow import keras
from models.discriminator import DiscriminatorNetwork
from models.srfr import SRFR
from training.train import (
    adjust_learning_rate,
    generate_num_epochs,
    train_srfr_model,
)
from utility.input_data import (
    augment_dataset,
    load_dataset,
    load_lfw,
    normalize_images,
)
from utility.timing import TimingLogger
from validation.validate import load_lfw_pairs, validate_model_on_lfw

# Importar Natural DS.
# Mudar o VGG no synthetic_dataset para o dataset correto

logging.basicConfig(
    filename='train_srfr_logs.txt',
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)
AUTOTUNE = tf.data.experimental.AUTOTUNE
NETWORK_SETTINGS = {
    'embedding_size': 512,
    'num_filters': 62,
    'gc': 32,
    'num_blocks': 23,
    'residual_scailing': 0.1,
}

TRAIN_SETTINGS = {
    'batch_size': 64,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'iterations': 400_000,
    'sr_weight': 0.1,
    'scale': 64,
    'angular_margin': 0.5,
}

def main():
    """Main training function."""
    timing = TimingLogger()
    timing.start()
    LOGGER.info(' -------- Importing Datasets --------')
    (
        synthetic_dataset,
        synthetic_num_classes,
        synthetic_dataset_len,
    ) = load_dataset('VGGFace2', concatenate=True)
    synthetic_dataset = synthetic_dataset.map(augment_dataset)
    synthetic_dataset = synthetic_dataset.map(
        lambda image, class_id: (normalize_images(image), class_id)
    )
    synthetic_dataset = synthetic_dataset.shuffle(
        buffer_size=2 * synthetic_dataset_len
    ).batch(TRAIN_SETTINGS['batch_size']).prefetch(buffer_size=AUTOTUNE)

    test_dataset = load_lfw()
    lfw_pairs = load_lfw_pairs()

    distributed_strategy = tf.distribute.MirroredStrategy()

    distributed_synthetic_dataset = distributed_strategy\
        .experimental_distribute_dataset(synthetic_dataset)
    distributed_test_dataset = distributed_strategy\
        .experimental_distribute_dataset(test_dataset)

    LOGGER.info(' -------- Creating Models and Optimizers --------')

    EPOCHS = generate_num_epochs(
        TRAIN_SETTINGS['iterations'],
        synthetic_num_classes,
        TRAIN_SETTINGS['batch_size']
    )

    srfr_model = SRFR(
        num_filters=NETWORK_SETTINGS['num_filters'],
        depth=50,
        categories=NETWORK_SETTINGS['embedding_size'],
        num_gc=NETWORK_SETTINGS['gc'],
        num_blocks=NETWORK_SETTINGS['num_blocks'],
        residual_scailing=NETWORK_SETTINGS['residual_scailing'],
        training=True,
    )
    sr_discriminator_model = DiscriminatorNetwork()

    learning_rate = tf.Variable(
        TRAIN_SETTINGS['learning_rate'],
        trainable=False,
        dtype=tf.float32,
        name='learning_rate'
    )
    srfr_optimizer = keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=TRAIN_SETTINGS['momentum']
    )
    discriminator_optimizer = keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=TRAIN_SETTINGS['momentum']
    )

    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(1),
        srfr_model=srfr_model,
        sr_discriminator_model=sr_discriminator_model,
        srfr_optimizer=srfr_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        learning_rate=learning_rate
    )
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory='./training_checkpoints',
        max_to_keep=3
    )
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_writer = tf.summary.create_file_writer(
        'logs/gradient_tape/{}/train'.format(current_time)
    )
    test_summary_writer = tf.summary.create_file_writer(
        'logs/gradient_tape/{}/test'.format(current_time)
    )

    LOGGER.info(' -------- Starting Training --------')
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        LOGGER.info(f' Restored from {manager.latest_checkpoint}')
    else:
        LOGGER.info(' Initializing from scratch.')

    for epoch in range(1, EPOCHS + 1):
        LOGGER.info(f' Start of epoch {epoch}')

        timing.start(train_srfr_model.__name__)
        srfr_loss, discriminator_loss = train_srfr_model(
            srfr_model,
            sr_discriminator_model,
            TRAIN_SETTINGS['batch_size'],
            srfr_optimizer,
            discriminator_optimizer,
            train_loss,
            distributed_synthetic_dataset,
            synthetic_num_classes,
            # natural_ds,
            # num_classes_natural,
            TRAIN_SETTINGS['sr_weight'],
            TRAIN_SETTINGS['scale'],
            TRAIN_SETTINGS['angular_margin'],
        )
        elapsed_time = timing.end(train_srfr_model.__name__, True)
        with train_summary_writer.as_default():
            tf.summary.scalar('srfr_loss', srfr_loss, step=epoch)
            tf.summary.scalar(
                'discriminator_loss',
                discriminator_loss,
                step=epoch,
            )
            tf.summary.scalar(
                'learning_rate',
                learning_rate.read_value(),
                step=epoch
            )
            tf.summary.scalar('training_time', elapsed_time, step=epoch)
        LOGGER.info(f' Epoch {epoch}, SRFR Loss: {srfr_loss:2.5f}, Discriminator\
             Loss: {discriminator_loss:2.5f}')

        if epoch % 100 == 0:
            timing.start(validate_model_on_lfw.__name__)
            (accuracy_mean, accuracy_std, validation_rate, validation_std, \
                far, auc, eer) = validate_model_on_lfw(
                    srfr_model,
                    test_dataset,
                    lfw_pairs,
                )
            elapsed_time = timing.end(validate_model_on_lfw.__name__, True)
            with test_summary_writer.as_default():
                tf.summary.scalar('accuracy_mean', accuracy_mean, step=epoch)
                tf.summary.scalar('accuracy_std', accuracy_std, step=epoch)
                tf.summary.scalar('validation_rate', validation_rate, step=epoch)
                tf.summary.scalar('validation_std', validation_std, step=epoch)
                tf.summary.scalar('far', far, step=epoch)
                tf.summary.scalar('auc', auc, step=epoch)
                tf.summary.scalar('eer', eer, step=epoch)
                tf.summary.scalar('testing_time', elapsed_time, step=epoch)

            LOGGER.info((
                f' Validation on LFW: Epoch {epoch} - '
                f'Accuracy: {accuracy_mean:2.5f} +- {accuracy_std:2.5f} - '
                f'Validation Rate: {validation_rate:2.5f} +- \
                    {validation_std:2.5f} @ FAR {far:2.5f} - '
                f'Area Under Curve (AUC): {auc:1.3f} - '
                f'Equal Error Rate (EER): {eer:1.3f} - '
            ))

        checkpoint.step.assign_add(1)
        if int(checkpoint.step) % 20 == 0:
            save_path = manager.save()
            LOGGER.info(f' Saved checkpoint for epoch {int(checkpoint.step)}: \
                {save_path}')

        learning_rate.assign(
            adjust_learning_rate(learning_rate.read_value(), epoch)
        )
    timing.calculate_mean(train_srfr_model.__name__)
    timing.calculate_mean(validate_model_on_lfw.__name__)

if __name__ == "__main__":
    main()
