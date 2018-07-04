from __future__ import print_function, division
from future.builtins import *
from future.builtins.disabled import *

import argparse
import os
import logging
import json
import os.path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python import debug as tf_debug

import common.visualize

import data
from sequence_encoder import SequenceEncoder
from sequence_embedder import SequenceEmbedder
from sequence_decoder import SequenceDecoder

# logging.basicConfig(level=logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)

LEARNING_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


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
            output_type: one of {'classes','sequences'}
    """

  logging.info("Features: {}".format(features.keys()))
  if labels is not None:
    logging.info("Labels: {}".format(labels.keys()))
  logging.info("Params: {}".format(params.values()))

  with tf.variable_scope('text_embedding'):
    if params.use_pretrained_encoder:
      hidden_state = tf.feature_column.input_layer(features, params.feature_columns)
      embedding_outputs = None
    else:
      batch_major_hidden_state, sequence_length = tf.contrib.feature_column.sequence_input_layer(features, params.feature_columns)
      real_batch_size = tf.shape(batch_major_hidden_state)[0]
      # Make time major
      hidden_state = tf.transpose(batch_major_hidden_state, [1, 0, 2])

      with tf.variable_scope('sequence_encoder'):
        encoder = SequenceEncoder(hidden_size=params.hidden_size, dtype=tf.float32,
                                  batch_size=real_batch_size)
        embedding_outputs, hidden_state = encoder.encode(hidden_state)

  train_op = None
  loss = None
  predictions = None
  if params.output_type == 'sequences':
    n_labels = params.n_labels
    hidden_state = tf.layers.dense(hidden_state, n_labels)

    seq_labels = None
    len_labels = None
    seq_labels = labels['angles'] if labels and 'angles' in labels else None
    len_labels = tf.cast(labels['n_frames'],
                         tf.int32) if seq_labels is not None else None

    decoder = SequenceDecoder(
        n_labels,
        hidden_state,
        cell_type=params.rnn_cell,
        memory=tf.nn.dropout(embedding_outputs, 1.0 - params.dropout),
        memory_sequence_length=sequence_length,
        label_lengths=len_labels,
        attention_size=params.attention_size)

    output = decoder.decode(seq_labels, len_labels)

    if mode != tf.estimator.ModeKeys.PREDICT:
      with tf.variable_scope('loss_parts'):
        seq_weights = tf.tile(
            tf.transpose([tf.sequence_mask(len_labels)]),
            multiples=[1, 1, n_labels])

        with tf.variable_scope('position'):
          position_loss = tf.losses.mean_squared_error(
              output, seq_labels, weights=seq_weights)
        tf.summary.scalar('position', position_loss)

        with tf.variable_scope('motion'):
          output_diff = output[1:, :, :] - output[:-1, :, :]
          label_diff = seq_labels[1:, :, :] - seq_labels[:-1, :, :]
          motion_loss = tf.losses.mean_squared_error(
              output_diff, label_diff, weights=seq_weights[:-1, :, :])
        tf.summary.scalar('motion', motion_loss)

      loss = ((1.0 - params.motion_loss_weight) * position_loss +
              params.motion_loss_weight * motion_loss)

    predictions = data.unnormalize(output, params.labels_mean,
                                   params.labels_std)
    tf.summary.histogram('predictions', predictions)
  elif params.output_type == 'classes':
    dropout = tf.layers.dropout(
        inputs=hidden_state,
        rate=params.dropout,
        training=mode == tf.estimator.ModeKeys.PREDICT)
    logits = tf.layers.dense(
        inputs=dropout, units=params.n_classes, activation=tf.nn.relu)

    if mode != tf.estimator.ModeKeys.PREDICT:
      onehot_labels = tf.one_hot(
          indices=tf.cast(labels['class'], tf.int32), depth=params.n_classes)
      loss = tf.losses.softmax_cross_entropy(
          onehot_labels=onehot_labels, logits=logits)

    predictions = tf.argmax(input=logits, axis=1)
    tf.summary.histogram("class", predictions)
  else:
    raise NotImplementedError("Output type must be one of: sequences, classes")

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  else:
    optimizer = tf.train.RMSPropOptimizer(
      learning_rate=params.learning_rate,
      momentum=0.0)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def generate_model_spec_name(params):
  name = ''
  name += 'output_type={},'.format(params.output_type)

  if params.output_type == 'sequences':
    name += 'hidden_size={},'.format(params.hidden_size)
    name += 'cell={},'.format(params.rnn_cell)
    name += 'motion_loss_weight={},'.format(params.motion_loss_weight)
  elif params.output_type == 'classes':
    name += 'n_classes={},'.format(params.n_classes)

  name += 'use_pretrained_encoder={},'.format(params.use_pretrained_encoder)
  name += 'dropout={},'.format(params.dropout)
  name += 'learning_rate={},'.format(params.learning_rate)
  name += 'batch_size={},'.format(params.batch_size)
  name += 'attention_size={},'.format(params.attention_size)
  name += 'embedding_size={}'.format(params.embedding_size)

  if params.note is not None:
    name += ',note={}'.format(params.note)

  return name


def setup_estimator(custom_params=dict()):
  # Load normalization parameters
  config_path = common.data_utils.DEFAULT_CONFIG_PATH
  if os.path.isfile(config_path):
    with open(config_path) as config_file:
      config = json.load(config_file)
    mean = config['angle_stats']['mean']
    std = config['angle_stats']['std']
  else:
    mean = 0
    std = 1

  default_params = tf.contrib.training.HParams(
      feature_columns=[None],
      output_type='sequences',
      rnn_cell='BasicRNNCell',
      dropout=0.3,
      n_labels=10,
      n_classes=8,
      hidden_size=512,
      batch_size=8,
      learning_rate=0.001,
      motion_loss_weight=0.5,
      labels_mean=mean,
      labels_std=std,
      use_pretrained_encoder=False,
      attention_size=8,
      embedding_size=32,
      note=None)

  model_params = default_params.override_from_dict(custom_params)

  if model_params.use_pretrained_encoder:
    feature_columns = [
        hub.text_embedding_column(
            'subtitle',
            'https://tfhub.dev/google/universal-sentence-encoder/1',
            trainable=False)
    ]
  else:
    feature_columns = [
        tf.feature_column.embedding_column(
            tf.contrib.feature_column.sequence_categorical_column_with_vocabulary_file(
                key='subtitle',
                vocabulary_file='vocab.txt',
                vocabulary_size=512),
            model_params.embedding_size)
    ]

  model_params = model_params.override_from_dict(dict(
      feature_columns=feature_columns))

  model_spec_name = generate_model_spec_name(model_params)
  run_config = tf.estimator.RunConfig(
      model_dir=os.path.join(LEARNING_DIRECTORY, 'log', model_spec_name),
      save_checkpoints_secs=120,
      save_summary_steps=200)

  estimator = tf.estimator.Estimator(
      model_fn=rnn_model_fn, config=run_config, params=model_params)

  return estimator, model_params


def predict_class(subtitle):
    estimator, model_params = setup_estimator({
       'output_type': 'classes',
       'motion_loss_weight': 0.8,
       'rnn_cell': 'BasicLSTMCell',
       'batch_size': 32,
       'use_pretrained_encoder': True
    })

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={
            'subtitle': np.array([subtitle]),
        }, num_epochs=1, shuffle=False)
    preds = np.array(list(estimator.predict(input_fn=predict_input_fn)))
    return preds[0]


def run_experiment(custom_params=dict(), options=None):
  """Runs an experiment with parameters changed as specified.
    The results are saved in log/xxx where xxx is specified by the
    specific parameters.

    Arguments:
      params: dict parameters to override
    """

  estimator, model_params = setup_estimator(custom_params)

  if options.train:
    # profiler_hook = tf.train.ProfilerHook(save_steps=200, output_dir='profile')
    debug_hook = tf_debug.TensorBoardDebugHook("localhost:7000")
    # debug_hook = tf_debug.LocalCLIDebugHook()
    estimator.train(lambda: data.input_fn(
        '../clips.tfrecords',
        batch_size=model_params.batch_size,
        n_epochs=256,
        split_sentences=(model_params.use_pretrained_encoder is False)
    ), hooks=[])

  if options.predict:
    subtitle = 'oh my god now it actually produces a different output for different subtitles'

    if not model_params.use_pretrained_encoder:
      subtitle = subtitle.split(' ')
      subtitle = list(filter(lambda x: len(x.strip()) > 0, map(common.data_utils.clean_word, subtitle)))

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={
            'subtitle': np.array([subtitle]),
        }, num_epochs=1, shuffle=False)
    preds = np.array(list(estimator.predict(input_fn=predict_input_fn)))
    n_frames = 70

    if model_params.output_type == 'sequences':
      frames = preds[:n_frames, 0, :].tolist()
      angles = list(map(common.pose_utils.get_named_angles, frames))

      with open("predicted_angles.json", "w") as write_file:
        serialized = json.dumps({'clip': angles}).decode('utf-8')
        write_file.write(serialized)
        print("Wrote angles to predicted_angles.json")

      pose = list(
          map(common.pose_utils.get_encoded_pose,
              map(common.pose_utils.get_pose_from_angles, angles)))
      common.visualize.animate_3d_poses(pose, save=True)
    elif model_params.output_type == 'classes':
      print("Prediction: {}".format(preds))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run a pose prediction model.')
  parser.add_argument('--train',
                      dest='train',
                      action='store_true',
                      help='whether to train the model')
  parser.add_argument('--predict',
                      dest='predict',
                      action='store_true',
                      help='whether do a prediction (after training)')
  parser.set_defaults(
    train=False,
    predict=False)

  run_experiment({
      'output_type': 'sequences',
      'motion_loss_weight': 0.5,
      'rnn_cell': 'GRUCell',
      'batch_size': 32,
      'use_pretrained_encoder': False,
      'hidden_size': 128,
      'learning_rate': 0.001,
      'dropout': 0.3,
      'embedding_size': 16,
      'note': 'attention_wrapper'
  }, parser.parse_args())
