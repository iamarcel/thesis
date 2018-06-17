from __future__ import print_function, division
from future.builtins import *
from future.builtins.disabled import *

import os
import itertools
import logging
import json
import os.path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import common.visualize

import data
from sequence_embedder import SequenceEmbedder
from sequence_decoder import SequenceDecoder

# logging.basicConfig(level=logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)


def rnn_model_fn(features, labels, mode, params):
    """TensorFlow model function for an RNN

    Args:
        features: Dict<feature_name, Tensor(feature_value)>
        labels: Tensor(labels or targets)
        mode: a tf.estimator.ModeKeys indicating the context.
            One of tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}
        params: a dict of other parameters
            learning_rate
            n_labels: output dimensionality
            hidden_size: size of the thought vector
            output_type: one of {'classes','sequences'}
    """

    logging.info("Features: {}".format(features.keys()))
    if labels is not None:
        logging.info("Labels: {}".format(labels.keys()))
    logging.info("Params: {}".format(params.values()))

    hidden_state = tf.feature_column.input_layer(features, params.feature_columns)

    train_op = None
    loss = None
    predictions = None
    if params.output_type == 'sequences':
        n_labels = params.n_labels
        hidden_state = tf.layers.dense(hidden_state, n_labels)

        decoder = SequenceDecoder(n_labels, hidden_state, cell_type=params.rnn_cell)

        seq_labels = None
        len_labels = None

        if mode != tf.estimator.ModeKeys.PREDICT:
            seq_labels = labels['angles'] if 'angles' in labels else None
            len_labels = tf.cast(labels['n_frames'], tf.int32) if seq_labels is not None else None

        output = decoder.decode(seq_labels, len_labels)

        if mode != tf.estimator.ModeKeys.PREDICT:
            for var in decoder.cell.trainable_weights:
                tf.summary.histogram(var.name, var)

            seq_weights = tf.tile(
                tf.transpose([tf.sequence_mask(len_labels)]),
                multiples=[1, 1, n_labels])

            position_loss = tf.losses.mean_squared_error(
                output,
                seq_labels,
                weights=seq_weights)

            output_diff = tf.square(output[1:, :, :] - output[:-1, :, :])
            label_diff = tf.square(seq_labels[1:, :, :] - seq_labels[:-1, :, :])
            motion_loss = tf.losses.mean_squared_error(
                output_diff,
                label_diff,
                weights=seq_weights[:-1, :, :])

            loss = (1.0 - params.motion_loss_weight) * position_loss + \
                   params.motion_loss_weight * motion_loss

        predictions = data.unnormalize(
            output, params.labels_mean, params.labels_std)
    elif params.output_type == 'classes':
        dropout = tf.layers.dropout(
            inputs=hidden_state,
            rate=params.dropout,
            training=mode == tf.estimator.ModeKeys.PREDICT)
        logits = tf.layers.dense(
            inputs=dropout,
            units=params.n_classes,
            activation=tf.nn.relu)

        if mode != tf.estimator.ModeKeys.PREDICT:
            onehot_labels = tf.one_hot(
                indices=tf.cast(labels['class'], tf.int32),
                depth=params.n_classes)
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels,
                logits=logits)

        predictions = tf.argmax(input=logits, axis=1)
        tf.summary.histogram("class", predictions)
    else:
        raise NotImplementedError("Output type must be one of: sequences, classes")

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)
    else:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=params.learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)


def generate_model_spec_name(params):
    name = ''
    name += 'output_type={},'.format(params.output_type)

    if params.output_type == 'sequences':
        name += 'hidden_size={},'.format(params.hidden_size)
        name += 'cell={},'.format(params.rnn_cell)
        name += 'motion_loss_weight={},'.format(params.motion_loss_weight)
    elif params.output_type == 'classes':
        name += 'n_classes={},'.format(params.n_classes)

    name += 'dropout={},'.format(params.dropout)
    name += 'learning_rate={},'.format(params.learning_rate)
    name += 'batch_size={}'.format(params.batch_size)

    return name


def run_experiment(custom_params=dict()):
    """Runs an experiment with parameters changed as specified.
    The results are saved in log/xxx where xxx is specified by the
    specific parameters.

    Arguments:
      params: dict parameters to override
    """

    # Load normalization parameters
    config_path = common.data_utils.DEFAULT_CONFIG_PATH
    if os.path.isfile(config_path):
        with open(config_path) as config_file:
            config = json.load(config_file)
        mean = config['angle_stats']['mean']
        std = config['angle_stats']['std']
    else:
        mean = 0
        std = 1

    feature_columns = [
        hub.text_embedding_column(
            'subtitle',
            'https://tfhub.dev/google/universal-sentence-encoder/1',
            trainable=False)
    ]

    default_params = tf.contrib.training.HParams(
        feature_columns=feature_columns,
        output_type='sequences',
        rnn_cell='BasicRNNCell',
        dropout=0.3,
        n_labels=10,
        n_classes=8,
        hidden_size=512,
        batch_size=8,
        learning_rate=0.001,
        motion_loss_weight=0.5,
        labels_mean=mean,
        labels_std=std)

    model_params = default_params.override_from_dict(custom_params)

    model_spec_name = generate_model_spec_name(model_params)
    run_config = tf.estimator.RunConfig(
        model_dir='log/{}'.format(model_spec_name),
        save_checkpoints_secs=60,
        save_summary_steps=100)

    estimator = tf.estimator.Estimator(
        model_fn=rnn_model_fn,
        config=run_config,
        params=model_params)



    do_train = False
    if do_train:
        # profiler_hook = tf.train.ProfilerHook(save_steps=200, output_dir='profile')
        debug_hook = tf_debug.TensorBoardDebugHook("f9f267322738:7000")
        estimator.train(lambda: data.input_fn(
            '../clips.tfrecords',
            batch_size=model_params.batch_size,
            n_epochs=100
        ), hooks=[])

    do_predict = True
    if do_predict:
        subtitle = 'up and down and up and down'

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={
                'subtitle': np.array([subtitle]),
            },
            num_epochs=1,
            shuffle=False)
        preds = np.array(list(estimator.predict(input_fn=predict_input_fn)))
        n_frames = 100

        if model_params.output_type == 'sequences':
            frames = preds[:n_frames, 0, :].tolist()
            angles = list(map(common.pose_utils.get_named_angles, frames))

            with open("predicted_angles.json", "w") as write_file:
                serialized = json.dumps({'clip': angles}).decode('utf-8')
                write_file.write(serialized)
                print("Wrote angles to predicted_angles.json")

            pose = list(map(
                common.pose_utils.format_joint_dict, map(
                    common.pose_utils.get_pose_from_angles, angles)))
            common.visualize.animate_3d_poses(pose, save=True)
        elif model_params.output_type == 'classes':
            print("Prediction: {}".format(preds))

if __name__ == '__main__':
    run_experiment({
        'output_type': 'sequences',
        'motion_loss_weight': 0.9,
        'rnn_cell': 'BasicLSTMCell'
    })
