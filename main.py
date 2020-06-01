"""Main training module for the Joint Learning Super Resolution Face\
 Recognition.
"""
import datetime
import logging
from functools import partial
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from models.discriminator import DiscriminatorNetwork
from models.srfr import SRFR
from training.train import (
    adjust_learning_rate,
    generate_num_epochs,
    Train,
)
from utils.input_data import LFW, parseConfigsFile, VggFace2
from utils.timing import TimingLogger

# Importar Natural DS.

logging.basicConfig(
    filename='train_logs.txt',
    level=logging.DEBUG,
)
LOGGER = logging.getLogger(__name__)
AUTOTUNE = tf.data.experimental.AUTOTUNE


def main():
    """Main training function."""
    timing = TimingLogger()
    timing.start()
    network_settings, train_settings, preprocess_settings = parseConfigsFile(
        ['network', 'train', 'preprocess'])

    strategy = tf.distribute.MirroredStrategy()
    LOGGER.info(' -------- Importing Datasets --------')
    vgg_dataset = VggFace2(mode='concatenated')
    synthetic_dataset = vgg_dataset.get_dataset()
    synthetic_dataset = vgg_dataset.augment_dataset()
    synthetic_dataset = vgg_dataset.normalize_dataset()
    synthetic_dataset = synthetic_dataset.cache()
    synthetic_dataset_len = vgg_dataset.get_dataset_size()
    synthetic_num_classes = vgg_dataset.get_number_of_classes()
    BATCH_SIZE = train_settings['batch_size'] * strategy.num_replicas_in_sync
    synthetic_dataset = synthetic_dataset.shuffle(
        buffer_size=5_120
    ).repeat().batch(BATCH_SIZE).prefetch(AUTOTUNE)

    lfw_dataset = LFW()
    (left_pairs, left_aug_pairs, right_pairs, right_aug_pairs,
     is_same_list) = lfw_dataset.get_dataset()
    left_pairs = left_pairs.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    left_aug_pairs = left_aug_pairs.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    right_pairs = right_pairs.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    right_aug_pairs = right_aug_pairs.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

    # Using `distribute_dataset` to distribute the batches across the GPUs
    synthetic_dataset = strategy.experimental_distribute_dataset(
        synthetic_dataset)
    left_pairs = strategy.experimental_distribute_dataset(left_pairs)
    left_aug_pairs = strategy.experimental_distribute_dataset(left_aug_pairs)
    right_pairs = strategy.experimental_distribute_dataset(right_pairs)
    right_aug_pairs = strategy.experimental_distribute_dataset(right_aug_pairs)

    LOGGER.info(' -------- Creating Models and Optimizers --------')

    EPOCHS = generate_num_epochs(
        train_settings['iterations'],
        synthetic_dataset_len,
        BATCH_SIZE,
    )

    learning_rate = tf.Variable(
        train_settings['learning_rate'],
        trainable=False,
        dtype=tf.float32,
        name='learning_rate'
    )

    with strategy.scope():
        srfr_model = SRFR(
            num_filters=network_settings['num_filters'],
            depth=50,
            categories=network_settings['embedding_size'],
            num_gc=network_settings['gc'],
            num_blocks=network_settings['num_blocks'],
            residual_scailing=network_settings['residual_scailing'],
            training=True,
            input_shape=preprocess_settings['image_shape_low_resolution'],
            num_classes_syn=synthetic_num_classes,
        )
        sr_discriminator_model = DiscriminatorNetwork()

        srfr_optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=train_settings['momentum']
        )
        srfr_optimizer = mixed_precision.LossScaleOptimizer(srfr_optimizer,
                                                        loss_scale='dynamic')
        discriminator_optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=train_settings['momentum']
        )
        discriminator_optimizer = mixed_precision.LossScaleOptimizer(
            discriminator_optimizer, loss_scale='dynamic')

        train_loss = partial(
            strategy.reduce,
            reduce_op=tf.distribute.ReduceOp.MEAN,
            axis=0,
        )

    checkpoint = tf.train.Checkpoint(
        epoch=tf.Variable(1),
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
        max_to_keep=5
    )

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_writer = tf.summary.create_file_writer(
        str(Path.cwd().joinpath('logs', 'gradient_tape', current_time,
                                'train')),
    )
    test_summary_writer = tf.summary.create_file_writer(
        str(Path.cwd().joinpath('logs', 'gradient_tape', current_time,
                                'test')),
    )

    LOGGER.info(' -------- Starting Training --------')
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        LOGGER.info(f' Restored from {manager.latest_checkpoint}')
    else:
        LOGGER.info(' Initializing from scratch.')

    with strategy.scope():
        for epoch in range(int(checkpoint.epoch), EPOCHS + 1):
            timing.start(Train.__name__)
            LOGGER.info(f' Start of epoch {epoch}')

            train = Train(strategy, srfr_model, srfr_optimizer,
                          sr_discriminator_model, discriminator_optimizer,
                          train_summary_writer, test_summary_writer,
                          checkpoint, manager)
            srfr_loss, discriminator_loss = train.train_srfr_model(
                BATCH_SIZE,
                train_loss,
                synthetic_dataset,
                synthetic_num_classes,
                # natural_ds,
                # num_classes_natural,
                test_dataset,
                lfw_pairs,
                sr_weight=train_settings['super_resolution_weight'],
                scale=train_settings['scale'],
                margin=train_settings['angular_margin'],
            )
            elapsed_time = timing.end(Train.__name__, True)
            with train_summary_writer.as_default():
                tf.summary.scalar('srfr_loss_per_epoch', srfr_loss, step=epoch)
                tf.summary.scalar(
                    'discriminator_loss_per_epoch',
                    discriminator_loss,
                    step=epoch,
                )
                tf.summary.scalar(
                    'learning_rate_per_epoch',
                    learning_rate.read_value(),
                    step=epoch
                )
                tf.summary.scalar('training_time_per_epoch', elapsed_time,
                                  step=epoch)
            LOGGER.info((f' Epoch {epoch}, SRFR Loss: {srfr_loss:.3f},'
                         f' Discriminator Loss: {discriminator_loss:.3f}'))

            train.save_model()

            learning_rate.assign(
                adjust_learning_rate(
                    learning_rate.read_value(),
                    int(checkpoint.epoch),
                )
            )
            checkpoint.epoch.assign_add(1)


if __name__ == "__main__":
    main()
