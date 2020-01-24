"""Main training module."""
import logging
import datetime

import tensorflow as tf
from tensorflow import keras
from models.resnet_model2 import ResNet
from input_data import augment_dataset, load_dataset, normalize_dataset
from training.train import train

logging.basicConfig(
    filename='train_logs.txt',
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)

AUTOTUNE = tf.data.experimental.AUTOTUNE
NETWORK_SETTINGS = {
    'embedding_size': 512,
    'scale': 64,
    'angular_margin': 0.5,
}

TRAIN_SETTINGS = {
    'batch_size': 64,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'iterations': 400_000
}

def _num_epochs(iterations, len_dataset, batch_size):
    train_size = tf.math.ceil(len_dataset, batch_size)
    return tf.math.ceil(iterations / train_size)

def main():
    """Main training function."""
    train_dataset, num_train_classes, dataset_len = load_dataset('VGGFace2')
    train_dataset = train_dataset.map(augment_dataset)
    train_dataset = train_dataset.map(
        lambda image, class_id: (normalize_dataset(image), class_id)
    )
    train_dataset = train_dataset.shuffle(buffer_size=2 * dataset_len)\
        .batch(TRAIN_SETTINGS['batch_size']).prefetch(buffer_size=AUTOTUNE)

    test_dataset, _, test_dataset_len = load_dataset(
        'LFW',
        remove_overlaps=False,
        sample_ids=True
    )
    test_dataset = test_dataset.map(augment_dataset)
    test_dataset = test_dataset.map(
        lambda image, class_id: (normalize_dataset(image), class_id)
    )
    test_dataset = test_dataset.shuffle(buffer_size=2 * test_dataset_len)\
        .batch(TRAIN_SETTINGS['batch_size']).prefetch(buffer_size=AUTOTUNE)

    distributed_strategy = tf.distribute.MirroredStrategy()

    distributed_train_dataset = distributed_strategy\
        .experimental_distribute_dataset(train_dataset)
    distributed_test_dataset = distributed_strategy\
        .experimental_distribute_dataset(test_dataset)

    EPOCHS = _num_epochs(
        TRAIN_SETTINGS['iterations'],
        dataset_len,
        TRAIN_SETTINGS['batch_size']
    )

    model = ResNet(50, NETWORK_SETTINGS['embedding_size'])

    optimizer = keras.optimizers.SGD(
        learning_rate=TRAIN_SETTINGS['learning_rate'],
        momentum=TRAIN_SETTINGS['momentum']
    )

    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(1),
        model=model,
        optimizer=optimizer
    )
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory='./training_checkpoints',
        max_to_keep=3
    )

    CURRENT_TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_writer = tf.summary.create_file_writer(
        'logs/gradient_tape/{}/train'.format(CURRENT_TIME)
    )
    test_summary_writer = tf.summary.create_file_writer(
        'logs/gradient_tape/{}/test'.format(CURRENT_TIME)
    )
    #current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    #train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    #logdir = "logs/"

    #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    LOGGER.info('-------- Starting Training --------')
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        LOGGER.info("Restored from {}".format(manager.latest_checkpoint))
    else:
        LOGGER.info("Initializing from scratch.")

    for epoch in range(EPOCHS):
        LOGGER.info('Start of epoch {}'.format(epoch + 1))

        loss_value = train(
            model,
            train_dataset,
            num_train_classes,
            TRAIN_SETTINGS['batch_size'],
            optimizer,
            NETWORK_SETTINGS['scale'],
            NETWORK_SETTINGS['angular_margin']
        )
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss_value, step=epoch + 1)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch + 1)


        LOGGER.info('Epoch {}, Loss: {:.10f}'.format(
            epoch + 1,
            float(loss_value)
        ))

        checkpoint.step.assign_add(1)
        if int(checkpoint.step) % 20 == 0:
            save_path = manager.save()
            LOGGER.info("Saved checkpoint for epoch {}: {}".format(
                int(checkpoint.step),
                save_path
            ))

        #if epoch % 50 == 0:
            # Test

if __name__ == "__main__":
    main()
