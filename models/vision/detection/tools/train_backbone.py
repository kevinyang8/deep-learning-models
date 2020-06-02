import os
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import argparse

from awsdet.utils.runner import init_dist

@tf.function
def parse(record):
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
    cmdline.add_argument('-lr', '--learning_rate', default=0.01, type=float,
                         help="""Start learning rate""")
    cmdline.add_argument('--momentum', default=0.01, type=float,
                         help="""Start learning rate""")
    cmdline.add_argument('-fp32', '--fp32', 
                         help="""disable mixed precision training""",
                         action='store_true')
    cmdline.add_argument('-xla_off', '--xla_off', 
                         help="""disable xla""",
                         action='store_true')
    return cmdline

def create_dataset(data_dir, batch_size):
    filenames = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
    data = tf.data.TFRecordDataset(filenames).shard(hvd.size(), hvd.rank())
    data = data.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data.batch(batch_size).prefetch(128)
    return data

def main():
    init_dist()
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    cmdline = add_cli_args()
    FLAGS, unknown_args = cmdline.parse_known_args()
    if not FLAGS.xla_off:
        tf.config.optimizer.set_jit(True)
    if not FLAGS.fp32:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    data = create_dataset(FLAGS.train_data_dir, FLAGS.batch_size)
    # add code to support many different models, resnet is temporary
    # also make sure to add difference between training from scratch or with preloaded weights
    # make sure to validate as well between epochs
    model = tf.keras.applications.ResNet50(weights=None, classes=1000)

    opt = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate * hvd.size(), momentum=FLAGS.momentum)
    if not FLAGS.fp32:
       #opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, loss_scale="dynamic")
       opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt, loss_scale='dynamic')
    opt = hvd.DistributedOptimizer(opt)

    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint('./work_dir/resnet50_checkpoints/checkpoint-{epoch}.h5'))

    model.compile(
        loss=loss_func,
        optimizer=opt,
        metrics=['accuracy'],
        experimental_run_tf_function=False
    )

    validation_data = create_dataset(FLAGS.validation_data_dir, FLAGS.batch_size)
    
    model.fit(
        data,
        validation_data=validation_data,
        callbacks=callbacks,
        epochs=FLAGS.num_epochs,
        verbose=1 if hvd.rank() == 0 else 0
    )

if __name__ == '__main__':
    main()
