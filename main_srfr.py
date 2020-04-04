"""Main training module for the Joint Learning Super Resolution Face\
 Recognition.
"""
import datetime
import logging
import sys
from functools import partial
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from models.discriminator import DiscriminatorNetwork
from models.srfr import SRFR
from training.train import (
    adjust_learning_rate,
    generate_num_epochs,
    train_srfr_model,
)
from utils.input_data import LFW, parseConfigsFile, VggFace2
from utils.timing import TimingLogger
from validation.validate import validate_model_on_lfw

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
    #synthetic_dataset_len = vgg_dataset.get_dataset_size()
    synthetic_num_classes = vgg_dataset.get_number_of_classes()

    BATCH_SIZE = train_settings['batch_size'] * strategy.num_replicas_in_sync

    synthetic_dataset = synthetic_dataset.shuffle(
        buffer_size=1024
    ).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    lfw_dataset = LFW()
    test_dataset = lfw_dataset.get_dataset()
    lfw_pairs = lfw_dataset.load_lfw_pairs()

    synthetic_dataset = strategy.experimental_distribute_dataset(synthetic_dataset)
    #test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    #distributed_strategy = tf.distribute.MirroredStrategy()

    #distributed_synthetic_dataset = distributed_strategy\
    #    .experimental_distribute_dataset(synthetic_dataset)
    #distributed_test_dataset = distributed_strategy\
    #    .experimental_distribute_dataset(test_dataset)

    LOGGER.info(' -------- Creating Models and Optimizers --------')

    EPOCHS = generate_num_epochs(
        train_settings['iterations'],
        synthetic_num_classes,
        train_settings['batch_size']
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
        )
        sr_discriminator_model = DiscriminatorNetwork()

        
        srfr_optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=train_settings['momentum']
        )
        discriminator_optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=train_settings['momentum']
        )

        train_loss = partial(
            strategy.reduce,
            reduce_op=tf.distribute.ReduceOp.MEAN,
            axis=0,
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
    #train_loss = tf.keras.metrics.Mean(name='train_loss')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_writer = tf.summary.create_file_writer(
        str(Path.cwd().joinpath('logs', 'gradient_tape', current_time, 'train')),
    )
    test_summary_writer = tf.summary.create_file_writer(
        str(Path.cwd().joinpath('logs', 'gradient_tape', current_time, 'test')),
    )

    LOGGER.info(' -------- Starting Training --------')
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        LOGGER.info(f' Restored from {manager.latest_checkpoint}')
    else:
        LOGGER.info(' Initializing from scratch.')

    with strategy.scope():
        for epoch in range(int(checkpoint.step), EPOCHS + 1):
            timing.start(train_srfr_model.__name__)
            LOGGER.info(f' Start of epoch {epoch}')

            srfr_loss, discriminator_loss = train_srfr_model(
                strategy,
                srfr_model,
                sr_discriminator_model,
                BATCH_SIZE,
                srfr_optimizer,
                discriminator_optimizer,
                train_loss,
                synthetic_dataset,
                synthetic_num_classes,
                # natural_ds,
                # num_classes_natural,
                sr_weight=train_settings['super_resolution_weight'],
                scale=train_settings['scale'],
                margin=train_settings['angular_margin'],
                summary_writer=train_summary_writer,
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
            LOGGER.info((f' Epoch {epoch}, SRFR Loss: {srfr_loss:.3f},'
                        f' Discriminator Loss: {discriminator_loss:.3f}'))

            if epoch % 2 == 0:
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
                    f' Validation on LFW: Epoch {epoch} -'
                    f' Accuracy: {accuracy_mean:.3f} +- {accuracy_std:.3f} -'
                    f' Validation Rate: {validation_rate:.3f} +-'
                    f' {validation_std:.3f} @ FAR {far:.3f} -'
                    f' Area Under Curve (AUC): {auc:.3f} -'
                    f' Equal Error Rate (EER): {eer:.3f} -'
                ))

            save_path = manager.save()
            LOGGER.info((f' Saved checkpoint for epoch {int(checkpoint.step)}:'
                        f' {save_path}'))

            learning_rate.assign(
                adjust_learning_rate(
                    learning_rate.read_value(),
                    int(checkpoint.step),
                )
            )
            checkpoint.step.assign_add(1)


if __name__ == "__main__":
    main()
