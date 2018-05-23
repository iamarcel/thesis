import logging

import tensorflow as tf
import numpy as np

import data
import rnn_helpers


class SequenceDecoder():

    def __init__(self, hidden_size, output_size, initial_state, batch_size):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.initial_state = initial_state
        self.batch_size = batch_size

        self._build_model()

    def decode_train(self, inputs, input_lengths, reuse=False, name='decode_train'):
        helper = ContinuousTrainingHelper(
            inputs,
            input_lengths,
            tf.TensorShape([self.output_size]))

        return self._decode(helper, reuse=reuse, name=name)

    def decode_predict(self, reuse=False, name='decode_predict'):
        start_input = np.array(
            [data.get_empty_output(self.output_size)] * self.batch_size,
            dtype=np.float32)

        helper = ContinuousSamplingHelper(
            start_input,
            0)

        return self._decode(helper, reuse=reuse, name=name)

    def _build_model(self):
        self.base_cell, cell = rnn_helpers.create_rnn_cell(self.hidden_size, name='decoder_cell')
        self.cell = tf.contrib.rnn.OutputProjectionWrapper(
            cell,
            output_size=self.output_size)

    def _decode(self, helper, reuse=False, name=None):
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.cell,
            helper=helper,
            initial_state=self.initial_state)

        return tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            maximum_iterations=1024)


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
