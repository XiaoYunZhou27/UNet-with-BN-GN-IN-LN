from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import OrderedDict

import tensorflow as tf
import os
import scipy.io as sio
import numpy as np

# this is the code for Unet with batch normalization, instance normalization, layer normalization, or group normalization,
# please comment others when using one
# please cite "Xiao-Yun Zhou, Guang-Zhong Yang. Normalization in Training U-Net for 2D Biomedical Semantic Segmentation, RAL accepted}, 2019
# \href{https://arxiv.org/pdf/1809.03783.pdf}{\underline{PDF}}
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
LR = 0.1
BATCH_SIZE = 1

os.system('rm -rf /data/XIAOYUN_ZHOU/Trained/Trained_%d/*'%FOLDER)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("Train_record", "***", "Train data dir") # please specify your tf.record file for training
tf.app.flags.DEFINE_string("Eval_record", "***", "Evaluation data dir") # please specify your tf.record file for evaluation
tf.app.flags.DEFINE_string("Test_record", "***", "Test data dir") # please specify your tf.record file for test
tf.app.flags.DEFINE_string("Model_dir", "***", "Model dir") # please specify your tf.record file for saving trained models

STEP_TRAIN = "***" # please specify your training steps here
STEP_EVAL = "***" # please specify your evaluation steps here
EPOCH = 2
BOUNDARIES = [STEP_TRAIN, STEP_TRAIN*2]
VALUES = [LR, LR/5, LR/25]
DROPOUT_RATE = 0
N_CLASS = 2
SHUFFLE = 500
N_BLOCK = 6
N_CONV = 2
N_CHANNEL = 1
CONV_SIZE = 3
N_FEATURE = 16
IMAGE_SIZE = 256
STEP_MAX = STEP_TRAIN*EPOCH
STEP_LOGGING = 20

def unet_model(features, labels, mode):

    features = tf.reshape(features, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CHANNEL])
    features_out = OrderedDict()

    for layer in range(0, N_BLOCK):

        feature_num = (2**layer)*N_FEATURE
        stddev = np.sqrt(2 / (CONV_SIZE ** 2 * feature_num))

        for layer_conv in range(0, N_CONV):

            features = tf.layers.conv2d(inputs=features, filters=feature_num, kernel_size=CONV_SIZE,
                                        activation=None, padding='same',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        bias_initializer=tf.constant_initializer(value=0.1))
            features = tf.nn.relu(tf.layers.dropout(inputs=features, rate=DROPOUT_RATE))

        features_out[layer] = features

        if layer < N_BLOCK-1:
            features = tf.layers.max_pooling2d(inputs=features_out[layer], pool_size=2, strides=2)

    for layer in range(N_BLOCK-2, -1, -1):

        if layer == N_BLOCK-2:
            features = features_out[N_BLOCK-1]

        feature_num = (2**layer)*N_FEATURE
        stddev = np.sqrt(2 / (CONV_SIZE ** 2 * feature_num))

        features = tf.layers.conv2d_transpose(inputs=features, filters=feature_num, strides=2, padding='same',
                                              kernel_size=CONV_SIZE, activation=None,
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                              bias_initializer=tf.constant_initializer(value=0.1))
        features = tf.nn.relu(tf.layers.dropout(inputs=features, rate=DROPOUT_RATE))
        features = tf.concat([features_out[layer], features], axis=3)

        for layer_conv in range(0, N_CONV):

            features = tf.layers.conv2d(inputs=features, filters=feature_num, kernel_size=CONV_SIZE,
                                        activation=None, padding='same',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        bias_initializer=tf.constant_initializer(value=0.1))
            features = tf.nn.relu(tf.layers.dropout(inputs=features, rate=DROPOUT_RATE))

    logits = tf.layers.conv2d(inputs=features, filters=N_CLASS, kernel_size=1,
                              activation=None, padding='same',
                              kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                              bias_initializer=tf.constant_initializer(value=0.1))

    classes_1 = tf.cast(x=tf.greater_equal(x=tf.slice(input_=tf.nn.softmax(logits=logits),
                                                      begin=[0, 0, 0, 1],
                                                      size=[-1, -1, -1, N_CLASS-1]),
                                           y=0.5),
                        dtype=tf.int32)
    class_0 = tf.subtract(x=1, y=classes_1)
    classes = tf.concat(values=[class_0, classes_1], axis=3)

    predictions = {"probabilities": tf.nn.softmax(logits=logits, name="probabilities"),
                   "classes": classes}

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels), logits=logits))
        tf.summary.scalar('loss', loss)
    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels), logits=tf.stop_gradient(logits)))
        tf.summary.scalar('loss', loss)

    if mode != tf.estimator.ModeKeys.PREDICT:

        for index in range(1, N_CLASS):
            labels_slice = tf.slice(labels, [0, 0, 0, index], [-1, -1, -1, 1])
            classes_slice = tf.cast(x=tf.slice(classes, [0, 0, 0, index], [-1, -1, -1, 1]), dtype=tf.float32)
            IoU = IoU_calculation(labels_slice=labels_slice, classes_slice=classes_slice)
            tf.summary.scalar("IoU_%s"%(index), IoU)

        for index in range(1, N_CLASS):
            segmentation = tf.slice(predictions['probabilities'], [0, 0, 0, index], [-1, -1, -1, 1])
            tf.summary.image("segmentation_%s"%(index), segmentation, BATCH_SIZE)

        for index in range(1, N_CLASS):
            groundtruth = tf.slice(labels, [0, 0, 0, index], [-1, -1, -1, 1])
            tf.summary.image("groundtruth_%s"%(index), groundtruth, BATCH_SIZE)

    if mode == tf.estimator.ModeKeys.TRAIN:
        step = tf.train.get_global_step()
        logging_hook_train = tf.train.LoggingTensorHook({"step": step, "loss": loss, "IoU": IoU}, every_n_iter=STEP_LOGGING)
        lr = tf.train.piecewise_constant(x=step, boundaries=BOUNDARIES, values=VALUES)

        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook_train])

    if mode == tf.estimator.ModeKeys.EVAL:
        logging_hook_eval = tf.train.LoggingTensorHook({"IoU": IoU}, every_n_iter=1)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, evaluation_hooks=[logging_hook_eval])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

def IoU_calculation(labels_slice, classes_slice):

    And = tf.multiply(labels_slice, classes_slice)
    Or = tf.add(labels_slice, classes_slice)
    if tf.equal(tf.count_nonzero(Or), 0)==True:
        Or_tmp = 1
    else:
        Or_tmp = tf.count_nonzero(Or)
    IoU = tf.divide(tf.cast(tf.count_nonzero(And), tf.float32), tf.cast(Or_tmp, tf.float32))

    return IoU

def parser(record):

    features = tf.parse_single_example(record,
        features={'Image': tf.FixedLenFeature([], tf.string),
                  'Label': tf.FixedLenFeature([], tf.string)})
    Images = tf.decode_raw(features["Image"], tf.float32)
    Images = tf.reshape(Images, [IMAGE_SIZE, IMAGE_SIZE])
    Labels = tf.decode_raw(features['Label'], tf.float32)
    Labels = tf.reshape(Labels, [IMAGE_SIZE, IMAGE_SIZE, N_CLASS])

    return Images, Labels

def Train_input_fn():

    dataset = tf.data.TFRecordDataset(FLAGS.Train_record)
    dataset = dataset.map(parser)

    dataset = dataset.shuffle(SHUFFLE)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels

def Eval_input_fn():

    dataset = tf.data.TFRecordDataset(FLAGS.Eval_record)
    dataset = dataset.map(parser)

    dataset = dataset.shuffle(SHUFFLE)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels

def Test_input_fn():

    dataset = tf.data.TFRecordDataset(FLAGS.Test_record)
    dataset = dataset.map(parser)

    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels

def main(unused_argv):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(session_config=config,
                                        save_checkpoints_steps=STEP_LOGGING,
                                        save_summary_steps=STEP_LOGGING,
                                        keep_checkpoint_max=10,
                                        log_step_count_steps=STEP_MAX)
    acnn = tf.estimator.Estimator(model_fn=unet_model, model_dir=FLAGS.Model_dir, config=run_config)

    train_spec = tf.estimator.TrainSpec(input_fn=Train_input_fn, max_steps=STEP_MAX)
    eval_spec = tf.estimator.EvalSpec(input_fn=Eval_input_fn, steps=STEP_EVAL, start_delay_secs=STEP_MAX, throttle_secs=STEP_MAX)
    tf.estimator.train_and_evaluate(acnn, train_spec, eval_spec)

    predictions = acnn.predict(input_fn=Test_input_fn)
    Num = 0
    for pred_dict in predictions:
        Num = Num + 1
        Classes = pred_dict['probabilities']
        sio.savemat(FLAGS.Model_dir+'Classes_%s.mat'%(Num), {'Classes': Classes})
    print(Num)

if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()