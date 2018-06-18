import tensorflow as tf

import rnn_helpers


class SequenceEncoder():

  def __init__(self, hidden_size, dtype, batch_size):
    self.hidden_size = hidden_size
    self.dtype = dtype
    self.batch_size = batch_size

    self._build_model()

  def encode(self, inputs):
    _, hidden_state = rnn_helpers.add_rnn_layers(
        inputs,
        self.hidden_size,
        self.batch_size,
        name='encoder_cell',
        dtype=self.dtype)

    return hidden_state

  def _build_model(self):
    pass
