import tensorflow as tf

def create_rnn_cell(cell_size, name=None):
    base_cell = tf.nn.rnn_cell.GRUCell(cell_size, name=name)
    cell = tf.nn.rnn_cell.DropoutWrapper(
        base_cell,
        input_keep_prob=0.5)
    return base_cell, cell

def add_rnn_layers(inputs, cell_size, batch_size, input_lengths=None, name=None, dtype=tf.float32):
    """Adds RNN layers after inputs and returns output, state

    Args:
        inputs: Tensor[batch_size, max_time, ?]
        num_units: Size of the cell state (defaults to 128)
    Returns: Tensor[batch_size, max_time, num_units] Output
    """
    # cell = tf.nn.rnn_cell.BasicRNNCell(cell_size, name=name)
    base_cell, cell = create_rnn_cell(cell_size, name=name)
    initial_state = cell.zero_state(batch_size, dtype)
    output, state = tf.nn.dynamic_rnn(
        cell,
        inputs,
        sequence_length=input_lengths,
        initial_state=initial_state,
        dtype=dtype)

    for var in base_cell.trainable_weights:
        tf.summary.histogram(var.name, var)

    return output, state