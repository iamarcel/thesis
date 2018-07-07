
import tensorflow as tf

import rnn_helpers


class SequenceDecoder():

  def __init__(self,
               output_size=32,
               initial_state=None,
               dropout=0.5,
               cell_type='BasicRNNCell',
               memory=None,
               memory_sequence_length=None,
               label_lengths=None):

    self.output_size = output_size
    self.state_size = output_size
    self.initial_state = initial_state
    self.batch_size = _best_effort_batch_size(self.initial_state)
    self.dropout = dropout
    self.label_lengths = label_lengths

    self._build_model(cell_type,
                      memory=memory,
                      memory_sequence_length=memory_sequence_length)

  def decode(self, labels=None, label_lengths=None, name='decode'):
    get_zero_input = lambda: tf.zeros([self.batch_size, self.output_size])

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
              finished, get_zero_input,
              lambda: inputs_ta.read(time)
          )

        next_loop_state = None
        return (elements_finished, next_input, next_cell_state, emit_output,
                next_loop_state)

    else:
      def loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output

        if cell_output is None:
          next_cell_state = self.initial_state
          next_input = get_zero_input()
        else:
          next_cell_state = cell_state
          next_input = cell_output

        if cell_output is None:
          elements_finished = tf.tile([False], [self.batch_size])
        else:
          elements_finished = tf.logical_or(tf.less(cell_output[:, 0], 0.0), time > 300)

        next_loop_state = None
        return (elements_finished, next_input, next_cell_state, emit_output,
                next_loop_state)

    output_ta, state, loop_state = tf.nn.raw_rnn(
        cell=self.cell, loop_fn=loop_fn)

    return output_ta.stack()

  def _build_model(self, cell_type, memory=None, memory_sequence_length=None):
    self.cell = rnn_helpers.create_rnn_cell(
        self.state_size,
        name='decoder_cell',
        cell_type=cell_type)
    # self.cell = rnn_helpers.add_dropout(self.cell, self.dropout)

    # Make correctly-shaped initial state if it's a tuple (LSTMCell)
    if isinstance(self.cell.state_size, tuple):
      self.initial_state = tf.nn.rnn_cell.LSTMStateTuple(
          c=tf.zeros([self.batch_size, self.cell.state_size[1]]),
          h=self.initial_state)

    if memory is not None:
      self.cell = tf.contrib.seq2seq.AttentionWrapper(
          self.cell,
          tf.contrib.seq2seq.LuongAttention(
              num_units=self.output_size,
              memory=tf.transpose(memory, [1, 0, 2]),
              memory_sequence_length=memory_sequence_length),
          initial_cell_state=self.initial_state,
          attention_layer_size=self.output_size)
      # self.cell = tf.contrib.rnn.AttentionCellWrapper(
      #     self.cell,
      #     attn_length=self.attention_size)
      tf.summary.histogram('attention_weights', self.cell.weights)

      self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

      # self.cell = rnn_helpers.add_dropout(self.cell, self.dropout)


def _best_effort_batch_size(input_):
  """Get static input batch size if available, with fallback to the dynamic one.
    Args:
        input: An iterable of time major input Tensors of shape
        `[batch_size, ...]`.
        All inputs should have compatible batch sizes.
    Returns:
        The batch size in Python integer if available, or a scalar Tensor
            otherwise.
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
