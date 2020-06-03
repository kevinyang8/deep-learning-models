import os
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import argparse
import sys
import random
import datetime


@tf.function
def parse(record):
    features = {'image/encoded': tf.io.FixedLenFeature((), tf.string),
                'image/class/label': tf.io.FixedLenFeature((), tf.int64)}
    parsed = tf.io.parse_single_example(record, features)
    image = tf.image.decode_jpeg(parsed['image/encoded'])
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float16)
    # augment images to prevent overfitting
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_saturation(image, 3, 5)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_jpeg_quality(image, 75, 95)
    image = tf.image.random_crop(image, size=[56, 56, 3])
    image = tf.image.resize(image, (224, 224))
    '''
    rand_num = random.random()
    augmentations = [
                     tf.image.central_crop
                     tf.image.adjust_brightness,
                     tf.image.random_saturation,
                     tf.i
                     ]
    if rand_num >= 0 and rand_num < 0.1:
        image = tf.image.flip_up_down(image)
    elif rand_num >= 0.1 and rand_num < 0.2:
        image = tf.image.flip_left_right(image)
    elif rand_num >= 0.2 and rand_num < 0.3:
        image = tf.image.rot90(image)
    elif rand_num >= 0.3 and rand_num < 0.4:
        image = tf.image.central_crop(image, 0.2)
        image = tf.image.resize(image, (224, 224))
    elif rand_num >= 0.4 and rand_num <= 0.5:
        image = tf.image.adjust_brightness(image, 0.2)
    '''

    label = tf.cast(parsed['image/class/label'] - 1, tf.int32)
    return image, label

@tf.function
def parse_validation(record):
    features = {'image/encoded': tf.io.FixedLenFeature((), tf.string),
                'image/class/label': tf.io.FixedLenFeature((), tf.int64)}
    parsed = tf.io.parse_single_example(record, features)
    image = tf.image.decode_jpeg(parsed['image/encoded'])
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float16)
    label = tf.cast(parsed['image/class/label'] - 1, tf.int32)
    return image, label

def add_cli_args():
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--train_data_dir', default='',
                         help="""Path to dataset in TFRecord format
                             (aka Example protobufs). Files should be
                             named 'train-*' and 'validation-*'.""")
    cmdline.add_argument('--validation_data_dir', default='',
                         help="""Path to dataset in TFRecord format
                             (aka Example protobufs). Files should be
                             named 'train-*' and 'validation-*'.""")
    cmdline.add_argument('-b', '--batch_size', default=128, type=int,
                         help="""Size of each minibatch per GPU""")
    cmdline.add_argument('--num_epochs', default=100, type=int,
                         help="""Number of epochs to train for.""")
    cmdline.add_argument('-lr', '--learning_rate', default=0.00125, type=float,
                         help="""Start learning rate""")
    cmdline.add_argument('--momentum', default=0.9, type=float,
                         help="""Start learning rate""")
    cmdline.add_argument('-fp32', '--fp32', 
                         help="""disable mixed precision training""",
                         action='store_true')
    cmdline.add_argument('-xla_off', '--xla_off', 
                         help="""disable xla""",
                         action='store_true')
    cmdline.add_argument('--model',
                         help="""Which model to train. Options are:
                         ResNet50 and ResNeXt50""")
    return cmdline

def create_dataset(data_dir, batch_size, validation):
    filenames = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
    data = tf.data.TFRecordDataset(filenames).shard(hvd.size(), hvd.rank())
    if not validation:
        data = data.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        data = data.map(parse_validation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data.batch(batch_size).prefetch(128)
    return data

def main():
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    cmdline = add_cli_args()
    FLAGS, unknown_args = cmdline.parse_known_args()
    if not FLAGS.xla_off:
        tf.config.optimizer.set_jit(True)
    if not FLAGS.fp32:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    data = create_dataset(FLAGS.train_data_dir, FLAGS.batch_size, validation=False)
    validation_data = create_dataset(FLAGS.validation_data_dir, FLAGS.batch_size, validation=True)

    # add code to support many different models, resnet is temporary
    # also make sure to add difference between training from scratch or with preloaded weights
    # make sure to validate as well between epochs

    model = tf.keras.applications.ResNet50(weights=None, classes=1000)

    opt = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate * hvd.size(), momentum=FLAGS.momentum)
    if not FLAGS.fp32:
       #opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, loss_scale="dynamic")
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt, loss_scale='dynamic')
    opt = hvd.DistributedOptimizer(opt, compression=hvd.Compression.fp16)

    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    model_dir = os.path.join(FLAGS.model + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S"))
    if hvd.rank() == 0:
        os.mkdir(model_dir)
    path_logs = os.path.join(os.getcwd(), model_dir, 'log.csv')

    hvd.allreduce([0], name="Barrier")

    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0), 
                 hvd.callbacks.MetricAverageCallback(),
                 hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1 if hvd.rank() == 0 else 0, steps_per_epoch=1252),
                 hvd.callbacks.LearningRateScheduleCallback(start_epoch=30, end_epoch=60, multiplier=1e-1),
                 hvd.callbacks.LearningRateScheduleCallback(start_epoch=60, end_epoch=80, multiplier=1e-2),
                 hvd.callbacks.LearningRateScheduleCallback(start_epoch=80, multiplier=1e-3)
                ]  

    if hvd.rank() == 0:
        path_checkpoints = os.path.join(os.getcwd(), model_dir, 'checkpoint-{epoch}.h5')
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(path_checkpoints))
        callbacks.append(tf.keras.callbacks.CSVLogger(path_logs, append=True, separator='-'))

    model.compile(
        loss=loss_func,
        optimizer=opt,
        metrics=['accuracy'],
        experimental_run_tf_function=False
    )
    
    model.fit(
        data,
        validation_data=validation_data,
        callbacks=callbacks,
        epochs=FLAGS.num_epochs,
        verbose=1 if hvd.rank() == 0 else 0
    )

if __name__ == '__main__':
    main()
