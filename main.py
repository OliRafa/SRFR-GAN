"""Main training module for the Joint Learning Super Resolution Face\
 Recognition.
"""
import datetime
import logging
from functools import partial
from pathlib import Path

import tensorflow as tf
from tensorflow_addons.optimizers import NovoGrad
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('set_memory_growth ok!')
    except RuntimeError as e:
        print('set_memory_growth failed!')
        print(str(e))
#tf.debugging.set_log_device_placement(True)
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from models.discriminator import DiscriminatorNetwork
from models.srfr import SRFR
from training.train import (
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
    BATCH_SIZE = train_settings['batch_size'] * strategy.num_replicas_in_sync
    temp_folder = Path.cwd().joinpath('temp', 'synthetic_ds')

    LOGGER.info(' -------- Importing Datasets --------')

    vgg_dataset = VggFace2(mode='concatenated')
    synthetic_dataset = vgg_dataset.get_dataset()
    synthetic_dataset = vgg_dataset.augment_dataset()
    synthetic_dataset = vgg_dataset.normalize_dataset()
    synthetic_dataset = synthetic_dataset.cache(str(temp_folder))
    #synthetic_dataset_len = vgg_dataset.get_dataset_size()
    synthetic_dataset_len = 3055527
    synthetic_num_classes = vgg_dataset.get_number_of_classes()
    synthetic_dataset = synthetic_dataset.shuffle(
        buffer_size=2_048
    ).repeat().batch(BATCH_SIZE).prefetch(1)

    #lfw_dataset = LFW()
    #test_dataset = lfw_dataset.get_dataset()
    #test_dataset = test_dataset.cache().prefetch(AUTOTUNE)
    #lfw_pairs = lfw_dataset.load_lfw_pairs()

    synthetic_dataset = strategy.experimental_distribute_dataset(
        synthetic_dataset)
    # test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    LOGGER.info(' -------- Creating Models and Optimizers --------')

    EPOCHS = generate_num_epochs(
        train_settings['iterations'],
        synthetic_dataset_len,
        BATCH_SIZE,
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

        srfr_optimizer = NovoGrad(
            learning_rate=train_settings['learning_rate'],
            beta_1=train_settings['momentum'],
            beta_2=train_settings['beta_2'],
            weight_decay=train_settings['weight_decay'],
            name='novograd_srfr',
        )
        srfr_optimizer = mixed_precision.LossScaleOptimizer(
            srfr_optimizer,
            loss_scale='dynamic',
        )
        discriminator_optimizer = NovoGrad(
            learning_rate=train_settings['learning_rate'],
            beta_1=train_settings['momentum'],
            beta_2=train_settings['beta_2'],
            weight_decay=train_settings['weight_decay'],
            name='novograd_discriminator',
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
    with strategy.scope():
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            LOGGER.info(f' Restored from {manager.latest_checkpoint}')
        else:
            LOGGER.info(' Initializing from scratch.')

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
                #test_dataset,
                #lfw_pairs,
                None,
                None,
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
                tf.summary.scalar('training_time_per_epoch', elapsed_time,
                                  step=epoch)
            LOGGER.info((f' Epoch {epoch}, SRFR Loss: {srfr_loss:.3f},'
                         f' Discriminator Loss: {discriminator_loss:.3f}'))

            train.save_model()

            checkpoint.epoch.assign_add(1)


if __name__ == "__main__":
    main()
