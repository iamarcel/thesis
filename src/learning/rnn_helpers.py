import tensorflow as tf


def create_rnn_cell(cell_size, name=None, dropout=0.5, cell_type='BasicRNNCell'):
    base_cell = getattr(tf.nn.rnn_cell, cell_type)(cell_size, name=name)
    return base_cell


def add_dropout(cell, dropout=0.5):
    keep = 1.0 - dropout

    return tf.nn.rnn_cell.DropoutWrapper(
        cell,
        input_keep_prob=keep)


def add_rnn_layers(inputs, cell_size, batch_size, input_lengths=None, name=None, dtype=tf.float32):
    """Adds RNN layers after inputs and returns output, state

    Args:
        inputs: Tensor[batch_size, max_time, ?]
        num_units: Size of the cell state (defaults to 128)
    Returns: Tensor[batch_size, max_time, num_units] Output
    """
    # cell = tf.nn.rnn_cell.BasicRNNCell(cell_size, name=name)
    base_cell = create_rnn_cell(cell_size, name=name)
    cell = add_dropout(base_cell)
    outputs, state = tf.nn.dynamic_rnn(
        cell,
        inputs,
        sequence_length=input_lengths,
        dtype=dtype,
        time_major=True)

    for var in base_cell.trainable_weights:
        tf.summary.histogram(var.name, var)

    return outputs, state
