import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from resnet_model import ResNet
from tfrecords_parser import read_casia, read_lfw, read_ms_celeb, read_vggface2
import datetime

def main():

    raw_image_dataset = tf.data.TFRecordDataset('testing.tfrecords')

    parsed_image_dataset = raw_image_dataset.map(read_vggface2)

    def preprocess(image):
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [224, 224])
        return image

    parsed_image_dataset['image_raw'] = parsed_image_dataset['image_raw'].map(preprocess)

    train_ds = parsed_image_dataset.shuffle(batch_size=len(parsed_image_dataset))

    for image_features, class_id, sample, name, gender in train_ds.take(10):

        #image = tf.io.decode_jpeg(image_features)
        #image_raw = image_features['image_raw'].numpy()
        print('Class: {} - Sample: {} - Name: {} - Gender: {}'.format(
            class_id,
            sample,
            name.numpy().decode('UTF-8'),
            gender
        ))

    model = ResNet(50, 4)

    optimizer = keras.optimizers.SGD(
        learning_rate=0.1,
        momentum=0.9
    )

    mse = keras.losses.MeanSquaredError()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    #logdir = "logs/"

    #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    print('-------- Starting Training --------')
    model._set_inputs(train_ds)

    for epoch in range(2):
        print('Start of epoch {}'.format(epoch + 1))
        #for step, train_lr in enumerate(train_ds):
        print(train_ds.shape)
        with tf.GradientTape() as tape:
            logits = model(train_ds)
            loss_value = mse(tf.zeros([4, 4]), logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', float(loss_value), step=epoch)

            #psnr = tf.image.psnr(logits, train_gt, max_val=1.0)
        #ssim_acc = ssim(logits, train_gt, max_val=255)
        print('Epoch {}, Loss: {:.10f}'.format(
            epoch+1,
            float(loss_value)
        ))

        #psnr.reset_state()
        #ssim.reset_state()
        
        model.save('save/resnet_epoch_{}'.format(epoch+1), save_format='tf')
        model.save_weights('save_w/resnet_epoch_{}'.format(epoch+1), save_format='tf')


if __name__ == "__main__":
    main()