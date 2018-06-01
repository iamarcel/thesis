from __future__ import print_function, division
from future.builtins import *
from future.builtins.disabled import *

import os
import itertools
import logging

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
        decoder = SequenceDecoder(params.hidden_size, params.n_labels, hidden_state, params.batch_size)

        if mode != tf.estimator.ModeKeys.PREDICT:
            outputs = decoder.decode_train(labels['poses'], labels['poses_lengths'])

            for var in decoder.base_cell.trainable_weights:
                tf.summary.histogram(var.name, var)

            loss = tf.losses.mean_squared_error(
                outputs.rnn_output,
                labels['poses'],
                weights=tf.tile(
                    tf.stack(
                        [tf.sequence_mask(labels['poses_lengths'])],
                        axis=2),
                    multiples=[1, 1, params.n_labels]))

        prediction_output = decoder.decode_predict(reuse=True)
        predictions = prediction_output.rnn_output
    elif params.output_type == 'classes':
        dropout = tf.layers.dropout(
            inputs=hidden_state,
            rate=0.4,
            training=mode == tf.estimator.ModeKeys.PREDICT)
        logits = tf.layers.dense(
            inputs=dropout,
            units=params.n_labels,
            activation=tf.nn.relu)

        if mode != tf.estimator.ModeKeys.PREDICT:
            onehot_labels = tf.one_hot(indices=tf.cast(labels['class'], tf.int32), depth=params.n_labels)
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels,
                logits=logits)

        predictions = tf.argmax(input=logits, axis=1)
        tf.summary.histogram("class", predictions)
    else:
        raise NotImplementedError()


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


if __name__ == '__main__':
    # Set up learning
    batch_size = 1

    # Use this if you want word ids:
    # input_fn, feature_columns, vocab_size, vocab, n_labels = data.get_input_angles(
    #     batch_size=batch_size,
    #     n_epochs=8192)

    feature_columns = [
        hub.text_embedding_column(
            'subtitle',
            'https://tfhub.dev/google/universal-sentence-encoder/1',
            trainable=False)
    ]

    # Use this if you want plain sentences:
    # input_fn, feature_columns, vocab_size, vocab, n_labels = data.get_input_sentences(
    #     batch_size=batch_size,
    #     n_epochs=8192)

    model_params = tf.contrib.training.HParams(
        feature_columns=feature_columns,
        output_type='classes',
        n_labels=8,
        hidden_size=512,
        batch_size=batch_size,
        learning_rate=0.001)

    run_config = tf.estimator.RunConfig(
        model_dir='log',
        save_checkpoints_secs=60,
        save_summary_steps=100)

    estimator = tf.estimator.Estimator(
        model_fn=rnn_model_fn,
        config=run_config,
        params=model_params)

    do_train = True
    if do_train:
        # profiler_hook = tf.train.ProfilerHook(save_steps=200, output_dir='profile')
        debug_hook = tf_debug.TensorBoardDebugHook("f9f267322738:7000")
        estimator.train(lambda: data.input_fn('../clips.tfrecords'), hooks=[])

    do_predict = True
    if do_predict:
        import json

        subtitle = 'we are creating a machine learning algorithm'
        feature, feature_len = data.subtitle2subtitle(subtitle, vocab)

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={
                'subtitles': np.array([feature]),
                'subtitle_lengths': np.array([feature_len])
            },
            num_epochs=1,
            shuffle=False)
        preds = np.array(list(estimator.predict(input_fn=predict_input_fn)))
        n_frames = 100

        logging.info(preds)
        pass

        frames = preds[0, :n_frames, :-1].tolist() # Cut off the mask
        angles = list(map(common.pose_utils.get_named_angles, frames))

        with open("predicted_angles.json", "w") as write_file:
            json.dump({'clip': angles}, write_file)

        # pose = preds[0, :n_frames, 0:30]
        # pose = np.reshape(pose, (n_frames, 10, 3))
        # pose_complete = np.tile(data.REFERENCE_POSE, (n_frames, 1, 1))
        # pose_complete[:, data.FILTERED_INDICES, :] = pose

        # common.visualize.animate_3d_poses(pose_complete)
