import pathlib
import tensorflow as tf
from tensorflow import keras
from vdsr_model import Vdsr
import IPython.display as display
import h5py
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
LAYERS = 20
SCALE_FACTOR = 2
BATCH_SIZE = 64
EPOCHS = 80
WEIGHT_DECAY = 0.0001
INITIAL_LEARNING_RATE = 0.1

#image_count = len(list(data_dir.glob('*.bmp')))
#image_count = len(list(data_dir.glob('*')))
#print(image_count)
#for image_path in list(data_dir.glob('*'))[:3]:
    #display.display(Image.open(str(image_path)))

#@tf.function
#def low_resolution_preprocess(image, scale_factor):
#    print(image)
#    image = tf.image.resize(
#        image,
#        [
#            image.numpy().shape[0] / scale_factor,
#            image.numpy().shape[1] / scale_factor
#            ],
#        method=tf.image.ResizeMethod.BICUBIC
#        )
#    image = tf.image.resize(
#        image,
#        [
#            image.numpy().shape[0] * scale_factor,
#            image.numpy().shape[1] * scale_factor
#            ],
#        method=tf.image.ResizeMethod.BICUBIC
#        )
#    return image

@tf.function
def adjust_learning_rate(learning_rate, current_epoch):
    return learning_rate * (0.1 ** (current_epoch // 20))

def main():
    hdf5_ds = h5py.File('/mnt/hdd_raid/datasets/VDSR_Train_Dataset/train.h5', 'r')
    #print(hdf5_ds['data'].shape)
    #print(hdf5_ds['label'])
    #data_dir = pathlib.Path('/mnt/hdd_raid/datasets/VDSR_Train_Dataset/PNG/291')
    #print('-------- Loading Train Dataset --------')
    ## Train Dataset
    #print('-------- Listing --------')
    #list_ds = tf.data.Dataset.list_files(str(data_dir/'*'))
    #print('-------- Read File --------')
    #train_ds = list_ds.map(
    #    lambda x: tf.io.read_file(x),
    #    num_parallel_calls=AUTOTUNE
    #    )
    ##train_ds = tf.io.read_file(list_ds)
    #print('-------- Decode PNG --------')
    #train_ds = train_ds.map(
    #    lambda x: tf.image.decode_png(x, channels=3),
    #    num_parallel_calls=AUTOTUNE
    #    )
    #
    #train_ds_lr = train_ds.map(lambda x: low_resolution_preprocess(x, SCALE_FACTOR),
    #    num_parallel_calls=AUTOTUNE
    #    )
    #
    #print('-------- Convert to Float32 --------')
    ## Convert to floats in the [0,1] range (apply normalization)
    #train_ds = train_ds.map(
    #    lambda x: tf.image.convert_image_dtype(x, tf.float32),
    #    num_parallel_calls=AUTOTUNE
    #    )
    #
#
    #
#
    #print('-------- Preprocessing Train Dataset --------')
    ## RGB to YCbCr conversion
    #train_ds = train_ds.map(lambda x:
    #    tf.image.rgb_to_yuv(x),
    #    num_parallel_calls=AUTOTUNE
    #)
    ## Creating the Low Resolution DS and zipping it with the Ground Truth images
#
    #train_ds = tf.data.Dataset.zip(
    #    train_ds_lr,
    #    train_ds
    #    )

    train_ds = tf.data.Dataset.from_tensor_slices((
        np.array(hdf5_ds['data']),
        np.array(hdf5_ds['label'])
    ))
    #train_gt = tf.data.Dataset.from_tensor_slices()
    #train_ds = tf.data.Dataset.zip((train_ds, train_gt))
    
    print('-------- Shuffling --------')
    # Shuffling <https://stackoverflow.com/questions/46444018/meaning-of-buffer-
    # size-in-dataset-map-dataset-prefetch-and-dataset-shuffle/48096625#48096625>
    train_ds = train_ds.shuffle(buffer_size=len(hdf5_ds['data']))
    # repeat()
    print('-------- Batching --------')
    # Create batches of BATCH_SIZE images each
    train_ds = train_ds.batch(BATCH_SIZE)

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    ## Test Dataset
    #test_dir = pathlib.Path('/mnt/hdd_raid/datasets/VDSR_Train_Dataset/PNG/Set5')
    #list_ds = tf.data.Dataset.list_files(str(test_dir/'*'))
    #test_ds = tf.io.read_file(list_ds)
    #test_ds = tf.image.decode_png(test_ds)
    ## Convert to floats in the [0,1] range (apply normalization)
    #test_ds = tf.image.convert_image_dtype(test_ds, tf.float32)
    ## Shuffling <https://stackoverflow.com/questions/46444018/meaning-of-buffer-
    ## size-in-dataset-map-dataset-prefetch-and-dataset-shuffle/48096625#48096625>
    #test_ds = tf.data.Dataset.shuffle(buffer_size=len(test_ds))
    ## repeat()
    ## Create batches of BATCH_SIZE images each
    #test_ds = test_ds.batch(BATCH_SIZE)
#
    ## RGB to YCbCr conversion
    #test_ds = tf.image.rgb_to_yuv(test_ds)
    ## Creating the Low Resolution DS and zipping it with the Ground Truth images
    #test_ds = tf.data.Dataset.zip(
    #    test_ds.map(lambda x: low_resolution_preprocess(x, SCALE_FACTOR),
    #                num_parallel_calls=AUTOTUNE
    #                ),
    #    test_ds
    #    )
    #test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    model = Vdsr(LAYERS, WEIGHT_DECAY)

    #loss = keras.losses.MeanSquaredError()

    optimizer = keras.optimizers.SGD(
        learning_rate=adjust_learning_rate(INITIAL_LEARNING_RATE, EPOCHS),
        momentum=0.9
        )

    #ssim = tf.image.ssim()

    mse = keras.losses.MeanSquaredError()
    mean = tf.keras.metrics.Mean()

    print('-------- Starting Training --------')


    for epoch in range(EPOCHS):
        print('Start of epoch {}'.format(epoch + 1))
        for step, (train_lr, train_gt) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = model(train_lr)
                loss_value = mse(train_gt, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            psnr = tf.image.psnr(logits, train_gt, max_val=1.0)
        #ssim_acc = ssim(logits, train_gt, max_val=255)
        print('Epoch {}, Loss: {:.10f}, PSNR: {}'.format(
            epoch+1,
            float(loss_value),
            mean.update_state(psnr).numpy() / 100
        ))

        #psnr.reset_state()
        #ssim.reset_state()
        model.save_weights('save/vdsr_{}'.format(epoch+1), save_format='tf')

if __name__ == "__main__":
    main()
