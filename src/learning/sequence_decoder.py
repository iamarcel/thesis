import logging

import tensorflow as tf
import numpy as np

import data
import rnn_helpers


class SequenceDecoder():

    def __init__(self, input_size, initial_state):
        self.input_size = input_size
        self.initial_state = initial_state
        self.batch_size = _best_effort_batch_size(self.initial_state)

        self._build_model()

    def decode(self, labels=None, label_lengths=None, name='decode'):
        get_zero_input = lambda: tf.zeros([self.batch_size, self.input_size])

        if labels is not None:
            inputs_ta = tf.TensorArray(
                dtype=tf.float32,
                size=tf.shape(labels)[0],
                element_shape=labels.get_shape()[1:])
            inputs_ta = inputs_ta.unstack(labels)

            def loop_fn(time, cell_output, cell_state, loop_state):
                emit_output = cell_output

                elements_finished = (time >= label_lengths)
                finished = tf.reduce_all(elements_finished)

                if cell_output is None:
                    next_cell_state = self.initial_state
                    next_input = get_zero_input()
                else:
                    next_cell_state = cell_state
                    next_input = tf.cond(
                        finished,
                        get_zero_input,
                        lambda: inputs_ta.read(time))

                next_loop_state = None
                return (elements_finished, next_input, next_cell_state,
                        emit_output, next_loop_state)

        else:
            def loop_fn(time, cell_output, cell_state, loop_state):
                emit_output = cell_output

                if cell_output is None:
                    next_cell_state = self.initial_state
                    next_input = get_zero_input()
                else:
                    next_cell_state = cell_state
                    next_input = cell_output

                elements_finished = (time >= 30)  # TODO Use cell_output

                next_loop_state = None
                return (elements_finished, next_input, next_cell_state,
                        emit_output, next_loop_state)

        output_ta, state, loop_state = tf.nn.raw_rnn(
            cell=self.cell,
            loop_fn=loop_fn)

        return output_ta.stack()

    def _build_model(self):
        self.cell, cell = rnn_helpers.create_rnn_cell(self.input_size, name='decoder_cell')

        # Make correctly-shaped initial state if it's a tuple (LSTMCell)
        if isinstance(self.cell.state_size, tuple):
            self.initial_state = tf.nn.rnn_cell.LSTMStateTuple(
                c=tf.zeros([self.batch_size, self.cell.state_size[1]]),
                h=self.initial_state)


def _best_effort_batch_size(input_):
    """Get static input batch size if available, with fallback to the dynamic one.
    Args:
        input: An iterable of time major input Tensors of shape
        `[batch_size, ...]`.
        All inputs should have compatible batch sizes.
    Returns:
        The batch size in Python integer if available, or a scalar Tensor otherwise.
    Raises:
        ValueError: if there is any input with an invalid shape.
    """
    shape = input_.shape
    if shape.ndims is None:
        raise ValueError('No dimensions in shape')
    if shape.ndims < 2:
        raise ValueError(
            "Expected state tensor %s to have rank at least 2" % input_)
    batch_size = shape[0].value
    if batch_size is not None:
        return batch_size
    # Fallback to the dynamic batch size of the first input.
    return tf.shape(input_)[0]
