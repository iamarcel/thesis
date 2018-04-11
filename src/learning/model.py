from __future__ import print_function
from __future__ import division

import os
import itertools
import logging

import numpy as np
import tensorflow as tf

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation

import data

tf.logging.set_verbosity(tf.logging.INFO)


class ContinuousTrainingHelper(tf.contrib.seq2seq.TrainingHelper):

    def __init__(self, inputs, sequence_length, sample_ids_shape, time_major=False, name=None):
        tf.contrib.seq2seq.TrainingHelper.__init__(
            self, inputs, sequence_length, time_major, name)
        self._sample_ids_shape = tf.TensorShape([inputs.get_shape()[-1]])

    def sample(self, time, outputs, name=None, **unused_kwargs):
        with tf.name_scope(name, "TrainingHelperSample", [time, outputs]):
            return outputs

    @property
    def sample_ids_shape(self):
        return self._sample_ids_shape

    @property
    def sample_ids_dtype(self):
        return data.POSE_DTYPE


class ContinuousSamplingHelper(tf.contrib.seq2seq.GreedyEmbeddingHelper):

    def __init__(self, start_tokens, end_token):
        # NOTE End token will be checked at the first feature of a sample
        self._embedding_fn = tf.identity

        self._start_tokens = tf.convert_to_tensor(
            start_tokens, name="start_tokens", dtype=data.POSE_DTYPE)
        self._end_token = tf.convert_to_tensor(
            end_token, name="end_token", dtype=data.POSE_DTYPE)
        self._batch_size = tf.shape(start_tokens)[0]
        self._start_inputs = self._embedding_fn(self._start_tokens)
        self._sample_ids_shape = tf.TensorShape([len(start_tokens[0])])

    def sample(self, time, outputs, state, name=None):
        del time, state
        return outputs

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        del time, outputs
        finished = tf.equal(sample_ids[:, 0], self._end_token)
        all_finished = tf.reduce_all(finished)
        logging.debug("Next inputs for sampling. Got sample shape {}".format(sample_ids.get_shape()))
        next_inputs = tf.cond(
            all_finished,
            lambda: self._start_inputs,
            lambda: sample_ids)
        return (finished, next_inputs, state)

    @property
    def sample_ids_shape(self):
        return self._sample_ids_shape

    @property
    def sample_ids_dtype(self):
        return data.POSE_DTYPE


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
    logging.info("Labels: {}".format(labels.keys()))
    logging.info("Params: {}".format(params.values()))

    def _get_input_tensors(features):
        """Converts the input dict into a feature and target tensor"""
        embeddings = tf.get_variable(
            'word_embeddings',
            [params.vocab_size, params.embedding_size])
        ids = tf.nn.embedding_lookup(embeddings, features['characters'])
        return ids

    def _create_rnn_cell(cell_size, name=None):
        cell = tf.nn.rnn_cell.BasicRNNCell(cell_size, name=name)
        initial_state = cell.zero_state(params.batch_size, data.POSE_DTYPE)
        return cell, initial_state

    def _add_rnn_layers(inputs, cell_size, input_lengths=None, name=None):
        """Adds RNN layers after inputs and returns output, state

        Args:
            inputs: Tensor[batch_size, max_time, ?]
            num_units: Size of the cell state (defaults to 128)
        Returns: Tensor[batch_size, max_time, num_units] Output
        """
        cell, initial_state = _create_rnn_cell(cell_size, name=name)
        output, state = tf.nn.dynamic_rnn(
            cell,
            inputs,
            sequence_length=input_lengths,
            initial_state=initial_state,
            dtype=data.POSE_DTYPE)
        return output, state

    def _add_fc_layers(final_state, output_size):
        """Add final dense layer to get the correct output size
        """
        return tf.layers.dense(final_state, output_size, activation=None)

    def _decode(helper, cell, initial_state, reuse=False, name=None):
        with tf.variable_scope('decode_{}'.format(name), reuse=reuse):
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=cell,
                helper=helper,
                initial_state=hidden_state)

            return tf.contrib.seq2seq.dynamic_decode(decoder=decoder)

    def _decode_train(cell, initial_state, reuse=False, name=None):
        helper = ContinuousTrainingHelper(
            labels['poses'],
            labels['poses_lengths'],
            tf.TensorShape([params.n_labels]))

        return _decode(helper, cell, initial_state, reuse=reuse, name=name)

    def _decode_infer(cell, initial_state, reuse=False, name=None):
        start_input = np.array(
            [data.get_empty_output()] * params.batch_size,
            dtype=np.float32)
        logging.debug("Starting input shape {}".format(start_input.shape))

        helper = ContinuousSamplingHelper(
            start_input,
            0)

        return _decode(helper, cell, initial_state, reuse=reuse, name=name)

    inputs = _get_input_tensors(features)
    _, hidden_state = _add_rnn_layers(
        inputs,
        params.hidden_size,
        input_lengths=features['characters_lengths'],
        name='encoder_cell')
    logging.debug("Hidden state shape: {}".format(hidden_state.get_shape()))
    decoder_cell, _ = _create_rnn_cell(params.hidden_size, name='decoder_cell')
    out_cell = tf.contrib.rnn.OutputProjectionWrapper(
        decoder_cell,
        output_size=params.n_labels)

    train_op = None
    loss = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        outputs, _, _ = _decode_train(out_cell, hidden_state, name='train')
        logging.debug("Training decoder output: {}".format(outputs.rnn_output))

        loss = tf.losses.mean_squared_error(
            outputs.rnn_output,
            labels['poses'],
            weights=tf.tile(
                tf.stack([tf.sequence_mask(labels['poses_lengths'])], axis=2),
                multiples=[1, 1, params.n_labels]))

        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

    prediction_output, _, _ = _decode_infer(
        out_cell,
        hidden_state,
        reuse=True,
        name='infer')
    predictions = prediction_output.rnn_output

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)


if __name__ == '__main__':
    # Set up learning
    batch_size = 16
    input_fn, feature_columns, vocab_size = data.get_input(
        batch_size=batch_size)

    model_params = tf.contrib.training.HParams(
        vocab_size=vocab_size,
        embedding_size=16,
        n_labels=data.N_POSE_FEATURES,
        hidden_size=128,
        batch_size=batch_size,
        learning_rate=0.0001)

    run_config = tf.estimator.RunConfig(
        model_dir='models',
        save_checkpoints_secs=60,
        save_summary_steps=100)

    estimator = tf.estimator.Estimator(
        model_fn=rnn_model_fn,
        config=run_config,
        params=model_params)

    # Train
    estimator.train(input_fn)
