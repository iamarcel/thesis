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

    def sample(self, time, outputs, name=None, **unused_kwargs):
        with tf.name_scope(name, "TrainingHelperSample", [time, outputs]):
            return outputs


class ContinuousSamplingHelper(tf.contrib.seq2seq.GreedyEmbeddingHelper):

    def __init__(self, start_tokens, end_token):
        # NOTE End token will be checked at the first feature of a sample
        tf.contrib.seq2seq.GreedyEmbeddingHelper.__init__(
            self, tf.identity, start_tokens, end_token)

    def sample(self, time, outputs, state, name=None):
        del time, state
        return outputs

    def next_inputs(self, time, outputs, state, samples, name=None):
        del time, outputs
        finished = tf.equal(samples[:, 0], self._end_token)
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(
            all_finished,
            lambda: self._start_inputs,
            lambda: samples)
        return (finished, next_inputs, state)

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
        return features['characters']

    def _create_rnn_cell(cell_size):
        cell = tf.nn.rnn_cell.BasicRNNCell(cell_size)
        initial_state = cell.zero_state(params.batch_size, tf.float32)
        return cell, initial_state

    def _add_rnn_layers(inputs, cell_size, input_lengths=None):
        """Adds RNN layers after inputs and returns output, state

        Args:
            inputs: Tensor[batch_size, max_time, ?]
            num_units: Size of the cell state (defaults to 128)
        Returns: Tensor[batch_size, max_time, num_units] Output
        """
        cell, initial_state = _create_rnn_cell(cell_size)
        output, state = tf.nn.dynamic_rnn(
            cell,
            inputs,
            sequence_length=input_lengths,
            initial_state=initial_state)
        return output, state

    def _add_fc_layers(final_state, output_size):
        """Add final dense layer to get the correct output size
        """
        return tf.layers.dense(final_state, output_size, activation=None)

    def _decode(helper, cell, initial_state, reuse=False):
        with tf.variable_scope('decode', reuse=reuse):
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=cell,
                helper=helper,
                initial_state=hidden_state)

            return tf.contrib.seq2seq.dynamic_decode(decoder=decoder)

    def _decode_train(cell, initial_state):
        helper = ContinuousTrainingHelper(
            labels['poses'],
            labels['poses_lengths'])

        return _decode(helper, cell, initial_state)

    def _decode_infer(cell, initial_state):
        helper = ContinuousSamplingHelper(
            np.tile(data.get_empty_input(), (params.batch_size, 1)),
            0)

        return _decode(helper, cell, initial_state)

    inputs = _get_input_tensors(features)
    # TODO Add embedding for input (RNN works on floats)
    _, hidden_state = _add_rnn_layers(
        inputs,
        params.hidden_size,
        input_lengths=features['characters_lengths'])
    decoder_cell, _ = _create_rnn_cell(params.n_labels)

    train_op = None
    loss = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        outputs = _decode_train(decoder_cell, hidden_state)

        loss = tf.contrib.seq2seq.sequence_loss(
            outputs,
            labels,
            weights=tf.sequence_mask(params.output_lengths))

        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

    predictions = _decode_infer(decoder_cell, hidden_state, reuse=True)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)


if __name__ == '__main__':
    # Set up learning
    batch_size = 16
    input_fn, feature_columns = data.get_input(
        batch_size=batch_size)

    model_params = tf.contrib.training.HParams(
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
