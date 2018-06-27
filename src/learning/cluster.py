from __future__ import print_function, division
from future.builtins import *
from future.builtins.disabled import *

import tensorflow as tf

from sequence_encoder import SequenceEncoder
from sequence_decoder import SequenceDecoder
import data

tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, labels, mode, params):
  with tf.variable_scope('input'):
    input_layer = labels['angles']
    lengths = tf.cast(labels['n_frames'], tf.int32)

    batch_size = tf.shape(input_layer)[0]
    input_dims = input_layer.get_shape()[2]

  with tf.variable_scope('encoder'):
    encoder = SequenceEncoder(
        hidden_size=params.hidden_size,
        dtype=tf.float32,
        batch_size=batch_size)

    encoding_outputs, hidden_state = encoder.encode(input_layer)

  with tf.variable_scope('squeeze'):
    hidden_state = tf.layers.dense(hidden_state, 2)
    hidden_state = tf.layers.dense(hidden_state, params.hidden_size)

  with tf.variable_scope('decoder'):
    decoder = SequenceDecoder(
        input_size=params.hidden_size,
        initial_state=hidden_state)

    output = decoder.decode(
        labels=input_layer,
        label_lengths=lengths)

  if mode != tf.estimator.ModeKeys.PREDICT:
    for var in decoder.cell.trainable_weights:
      tf.summary.histogram(var.name, var)

    with tf.variable_scope('loss'):
      seq_weights = tf.tile(
          tf.transpose([tf.sequence_mask(lengths)]),
          multiples=[1, 1, input_dims])

      loss = tf.losses.mean_squared_error(
          output,
          input_layer,
          weights=seq_weights)

  predictions = output

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  else:
    with tf.variable_scope('train'):
      optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
      train_op = optimizer.minimize(
          loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


if __name__ == '__main__':
  feature_columns = [
  ]

  model_params = tf.contrib.training.HParams(
      # feature_columns=feature_columns,
      hidden_size=10,
      learning_rate=0.001,
      batch_size=32)

  run_config = tf.estimator.RunConfig(
      model_dir='log/clustering',
      save_checkpoints_secs=60,
      save_summary_steps=100)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn, config=run_config, params=model_params)

  estimator.train(lambda: data.input_fn(
      '../clips.tfrecords',
      batch_size=model_params.batch_size,
      n_epochs=1000,
      split_sentences=False
  ), hooks=[])
