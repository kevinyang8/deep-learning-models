import os
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import argparse
import datetime
import random
import logging
from tqdm import tqdm
import sys

from awsdet.utils.schedulers.schedulers import WarmupScheduler

@tf.function
def parse(record):
    features = {'image/encoded': tf.io.FixedLenFeature((), tf.string),
                'image/class/label': tf.io.FixedLenFeature((), tf.int64)}
    parsed = tf.io.parse_single_example(record, features)
    image = tf.image.decode_jpeg(parsed['image/encoded'])
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float16)
    image = tf.image.central_crop(image, random.uniform(0.7, 0.9))
    image = tf.image.resize(image, (224, 224))
    image = tf.image.random_flip_left_right(image)
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
    cmdline.add_argument('-lr', '--learning_rate', default=0.01, type=float,
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
                         resnet50 and resnext50""")
    cmdline.add_argument('--fine_tune',
                         help="""Whether to fine tune on pretrained model or 
                         train the full model from scratch. Must specify weights
                         path if flag is set.""",
                         action='store_true')
    cmdline.add_argument('--weights_path', 
                         help='Path to weights for pretrained model')
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

@tf.function
def train_step(model, opt, loss_func, images, labels, first_batch, fp32=False):
    with tf.GradientTape() as tape:
        probs = model(images, training=True)
        loss_value = loss_func(labels, probs)
        if not fp32:
            scaled_loss = opt.get_scaled_loss(loss_value)
    tape = hvd.DistributedGradientTape(tape, compression=hvd.Compression.fp16)
    if fp32:
        grads = tape.gradient(loss_value, model.trainable_variables)
    else:
        scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
        grads = opt.get_unscaled_gradients(scaled_grads)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)
    return loss_value

@tf.function
def validation_step(images, labels, model, loss_func):
    pred = model(images, training=False)
    loss = loss_func(labels, pred)
    top_1_pred = tf.math.top_k(pred, k=1)[1]
    labels = tf.cast(labels, tf.int32)
    return loss, top_1_pred

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

    if FLAGS.model == 'resnet50':
        if not FLAGS.fine_tune:
            model = tf.keras.applications.ResNet50(weights=None, classes=1000)
        else:
            model = tf.keras.applications.ResNet50(weights='imagenet', classes=1000)

    learning_rate = FLAGS.learning_rate * hvd.size()
    scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    boundaries=[30, 60, 80], 
                    values=[learning_rate, learning_rate * 0.1, learning_rate * 0.01, learning_rate * 0.001])
    scheduler = WarmupScheduler(optimizer=scheduler, initial_learning_rate=learning_rate, warmup_steps=5)
    opt = tf.keras.optimizers.SGD(learning_rate=scheduler, momentum=FLAGS.momentum)
    if not FLAGS.fp32:
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt, loss_scale='dynamic')

    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    if hvd.rank() == 0:
        model_dir = os.path.join(FLAGS.model + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S"))
        path_logs = os.path.join(os.getcwd(), model_dir, 'log.csv')
        os.mkdir(model_dir)

        logging.basicConfig(filename=path_logs,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        logging.info("Training Logs")
        logger = logging.getLogger('logger')
    
    #checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)

    hvd.allreduce(tf.constant(0))

    for epoch in range(FLAGS.num_epochs):
        if hvd.rank() == 0:
            print('Starting training Epoch %d/%d' % (epoch, FLAGS.num_epochs))
        training_score = 0
        for batch, (images, labels) in tqdm(enumerate(data)):
            loss = train_step(model, opt, loss_func, images, labels, batch==0 and epoch==0, fp32=FLAGS.fp32)
            _, predictions = validation_step(images, labels, model, loss_func)
            score = np.sum(np.equal(predictions, labels))
            training_score += score
        training_accuracy = training_score / (FLAGS.batch_size * (batch + 1))
        average_training_accuracy = hvd.allreduce(tf.constant(training_accuracy))
        average_training_loss = hvd.allreduce(tf.constant(loss))

        if hvd.rank() == 0:
            print('Starting validation Epoch %d/%d' % (epoch, FLAGS.num_epochs))
        validation_score = 0
        counter = 0
        for images, labels in tqdm(validation_data):
            loss, predictions = validation_step(images, labels, model, loss_func)
            score = np.sum(np.equal(predictions, labels))
            validation_score += score
            counter += 1
        validation_accuracy = validation_score / (FLAGS.batch_size * counter)
        average_validation_accuracy = hvd.allreduce(tf.constant(validation_accuracy))
        average_validation_loss = hvd.allreduce(tf.constant(loss))

        if hvd.rank() == 0:
            #path_checkpoint = path_logs = os.path.join(os.getcwd(), model_dir)
            info_str = 'Epoch: %d, Train Accuracy: %f, Train Loss: %f, Validation Accuracy: %f, Validation Loss: %f' % \
                    (epoch, average_training_accuracy, average_training_loss, average_validation_accuracy, average_validation_loss)
            print(info_str)
            logger.info(info_str)
            #checkpoint.save(path_checkpoint)


if __name__ == '__main__':
    main()
