from __future__ import print_function
from __future__ import division

import os
import itertools
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import common.visualize

import data
from sequence_encoder import SequenceEncoder
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
    """

    logging.info("Building RNN Model.")
    logging.info("Features: {}".format(features.keys()))
    if labels is not None:
        logging.info("Labels: {}".format(labels.keys()))
    logging.info("Params: {}".format(params.values()))

    def _get_input_tensors(features):
        """Converts the input dict into a feature and target tensor"""
        embeddings = tf.get_variable(
            'word_embeddings',
            [params.vocab_size, params.embedding_size])
        ids = tf.nn.embedding_lookup(embeddings, features['characters'])
        tf.summary.histogram(
            "characters_length",
            features['characters_lengths'])
        return ids

    def _add_fc_layers(final_state, output_size):
        """Add final dense layer to get the correct output size
        """
        return tf.layers.dense(final_state, output_size, activation=None)

    with tf.variable_scope('encoding'):
        inputs = _get_input_tensors(features)
        encoder = SequenceEncoder(params.hidden_size, data.POSE_DTYPE, params.batch_size)
        hidden_state = encoder.encode(inputs)

    decoder = SequenceDecoder(params.hidden_size, params.n_labels, hidden_state, params.batch_size)

    train_op = None
    loss = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        outputs, _, _ = decoder.decode_train(labels['poses'], labels['poses_lengths'])

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

        optimizer = tf.train.AdamOptimizer(
            learning_rate=params.learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

    prediction_output, _, _ = decoder.decode_predict(
        reuse=True)
    predictions = prediction_output.rnn_output

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)


if __name__ == '__main__':
    # Set up learning
    batch_size = 1
    input_fn, feature_columns, vocab_size, vocab, n_labels = data.get_input_angles(
        batch_size=batch_size,
        n_epochs=8192)

    model_params = tf.contrib.training.HParams(
        vocab_size=vocab_size,
        embedding_size=8,
        n_labels=n_labels,
        hidden_size=128,
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
        # debug_hook = tf_debug.TensorBoardDebugHook("24c83624d92b:7000")
        estimator.train(input_fn, hooks=[])

    do_predict = False
    if do_predict:
        import json

        subtitle = 'we are creating a machine learning algorithm'
        feature, feature_len = data.subtitle2features(subtitle, vocab)

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={
                'characters': np.array([feature]),
                'characters_lengths': np.array([feature_len])
            },
            num_epochs=1,
            shuffle=False)
        preds = np.array(list(estimator.predict(input_fn=predict_input_fn)))
        n_frames = 100

        frames = preds[0, :n_frames, :-1].tolist() # Cut off the mask
        angles = list(map(common.pose_utils.get_named_angles, frames))

        with open("predicted_angles.json", "w") as write_file:
            json.dump({'clip': angles}, write_file)

        # pose = preds[0, :n_frames, 0:30]
        # pose = np.reshape(pose, (n_frames, 10, 3))
        # pose_complete = np.tile(data.REFERENCE_POSE, (n_frames, 1, 1))
        # pose_complete[:, data.FILTERED_INDICES, :] = pose

        # common.visualize.animate_3d_poses(pose_complete)
