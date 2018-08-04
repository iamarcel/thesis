from __future__ import print_function, division
from future.builtins import *
from future.builtins.disabled import *

import argparse
import os
import logging
import json
import os.path
import csv

# Itertools has a different name in Python 2/3
import itertools
try:
  from itertools import zip_longest as zip_longest
except:
  from itertools import izip_longest as zip_longest

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python import debug as tf_debug

import common.visualize

import data
from sequence_encoder import SequenceEncoder
from sequence_embedder import SequenceEmbedder
from sequence_decoder import SequenceDecoder

logging.basicConfig(level=logging.WARN)
tf.logging.set_verbosity(tf.logging.INFO)

LEARNING_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CLUSTER_CENTERS_PATH = '../cluster-centers.json'


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


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
      memory = None
      sequence_length = None
    else:
      batch_major_hidden_state, sequence_length = tf.contrib.feature_column.sequence_input_layer(features, params.feature_columns)
      real_batch_size = tf.shape(batch_major_hidden_state)[0]
      # Make time major
      hidden_state = tf.transpose(batch_major_hidden_state, [1, 0, 2])

      with tf.variable_scope('sequence_encoder'):
        encoder = SequenceEncoder(hidden_size=params.hidden_size, dtype=tf.float32,
                                  batch_size=real_batch_size)
        memory, hidden_state = encoder.encode(hidden_state)
        memory = tf.layers.dense(memory, params.hidden_size)
        memory = tf.layers.dense(memory, params.hidden_size)
        memory = tf.nn.dropout(memory, 1.0 - params.dropout)

  train_op = None
  loss = None
  predictions = None
  accuracy_metric = None

  n_labels = params.n_labels

  seq_labels = None
  len_labels = None
  seq_labels = labels['angles'] if labels and 'angles' in labels else None
  len_labels = tf.cast(labels['n_frames'],
                        tf.int32) if seq_labels is not None else None

  if params.output_type == 'sequences':
    hidden_state = tf.layers.dense(hidden_state, n_labels)

    decoder = SequenceDecoder(
        output_size=n_labels,
        initial_state=hidden_state,
        cell_type=params.rnn_cell,
        memory=memory,
        memory_sequence_length=sequence_length,
        label_lengths=len_labels)

    output = decoder.decode(seq_labels, len_labels)

    # Add 0 at the end to make sure *some* value indicates the end frame
    length_indicator = tf.concat([output[:, 0, 0], [0]], axis=0)
    length_indicator = tf.Print(length_indicator, [length_indicator], '=== LENGTH INDICATOR')
    end_indices = tf.where(length_indicator < 0.3)
    end_indices = tf.Print(end_indices, [end_indices], '=== END INDICES ')
    predicted_length = end_indices[0]
    tf.summary.histogram('predicted_length', predicted_length)
  elif params.output_type == 'classes':
    with tf.variable_scope('classification_decoder'):
      logits = hidden_state
      logits = tf.layers.dropout(
          inputs=logits,
          rate=params.dropout,
          training=mode == tf.estimator.ModeKeys.PREDICT)
      logits = tf.layers.dense(
          inputs=logits, units=params.n_classes, activation=tf.nn.relu)

      if mode != tf.estimator.ModeKeys.PREDICT:
        predicted_class = tf.argmax(input=logits, axis=1)
        output = tf.gather(params.centers, predicted_class, axis=1)

        # Pad/slice output so it matches ground truth
        output = tf.pad(output, [[0, tf.maximum(0, tf.shape(labels['angles'])[0] - tf.shape(output)[0])], [0, 0], [0, 0]])
        output = tf.slice(output, [0, 0, 0], tf.shape(labels['angles']))
  else:
    raise NotImplementedError("Output type must be one of: sequences, classes")

  if mode != tf.estimator.ModeKeys.PREDICT:
    with tf.variable_scope('gesture_loss'):
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

      with tf.variable_scope('total'):
        gesture_loss = ((1.0 - params.motion_loss_weight) * position_loss +
                        params.motion_loss_weight * motion_loss)
      tf.summary.scalar('total', gesture_loss)
      gesture_loss_metric = tf.metrics.mean(gesture_loss)

    if params.output_type == 'sequences':
      loss = gesture_loss
    else:
      onehot_labels = tf.one_hot(
          indices=tf.cast(labels['class'], tf.int32), depth=params.n_classes)
      loss = tf.losses.softmax_cross_entropy(
          onehot_labels=onehot_labels, logits=logits)
      accuracy_metric = tf.metrics.accuracy(
        labels=labels['class'],
        predictions=tf.argmax(logits, axis=1))

  # Output needs to be unnormalized for predictions
  predictions = data.unnormalize(output, params.labels_mean,
                                  params.labels_std)
  tf.summary.histogram('predictions', predictions)

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  else:
    optimizer = tf.train.RMSPropOptimizer(
      learning_rate=params.learning_rate,
      momentum=0.5)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    metric_ops = dict(
      gesture_loss=gesture_loss_metric)

    if accuracy_metric is not None:
      metric_ops['accuracy'] = accuracy_metric

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metric_ops)


def generate_model_spec_name(params):
  name = ''
  name += 'output_type={},'.format(params.output_type)
  name += 'hidden_size={},'.format(params.hidden_size)

  if params.output_type == 'sequences':
    name += 'cell={},'.format(params.rnn_cell)
    name += 'motion_loss_weight={},'.format(params.motion_loss_weight)
  elif params.output_type == 'classes':
    name += 'n_classes={},'.format(params.n_classes)

  name += 'use_pretrained_encoder={},'.format(params.use_pretrained_encoder)
  name += 'dropout={},'.format(params.dropout)
  name += 'learning_rate={},'.format(params.learning_rate)
  name += 'batch_size={},'.format(params.batch_size)
  name += 'embedding_size={}'.format(params.embedding_size)

  if params.note is not None:
    name += ',note={}'.format(params.note)

  return name


def setup_estimator(custom_params=dict(), cluster_centers_path=CLUSTER_CENTERS_PATH):
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

  # Preprocess centers
  if os.path.isfile(cluster_centers_path):
    with open(cluster_centers_path) as centers_file:
      centers = json.load(centers_file)['clusters']
      centers_lengths = [len(center) for center in centers]

      centers = [([common.pose_utils.get_angle_list(frame) for frame in center]) for center in centers]
      centers = [data.add_length_indicator(center) for center in centers]
      centers = [data.normalize(np.array(center), mean, std) for center in centers]

      # Pad until maximum length
      frame_length = len(centers[0][0])
      max_center_len = max(x.shape[0] for x in centers)
      centers = np.array([np.concatenate([center, np.zeros((max_center_len - center.shape[0], frame_length))]) for center in centers])
      centers = np.swapaxes(centers, 0, 1)
  else:
    raise ValueError('Cluster centers file is not present at {}'.format(cluster_centers_path))

  default_params = tf.contrib.training.HParams(
      feature_columns=[None],
      output_type='sequences',
      rnn_cell='BasicRNNCell',
      dropout=0.3,
      n_labels=11,
      n_classes=8,
      hidden_size=32,
      batch_size=8,
      learning_rate=0.001,
      motion_loss_weight=0.5,
      labels_mean=mean,
      labels_std=std,
      use_pretrained_encoder=False,
      embedding_size=32,
      centers=centers,
      note=None)

  model_params = default_params.override_from_dict(custom_params)

  if model_params.use_pretrained_encoder:
    feature_columns = [
        hub.text_embedding_column(
            'subtitle',
            'https://tfhub.dev/google/universal-sentence-encoder/2',
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
      model_dir=os.path.join(LEARNING_DIRECTORY, 'log', model_spec_name))

  estimator = tf.estimator.Estimator(
      model_fn=rnn_model_fn, config=run_config, params=model_params)

  return estimator, model_params


def predict_classes(subtitles, cluster_centers_path=CLUSTER_CENTERS_PATH):
    estimator, model_params = setup_estimator({
       'output_type': 'classes',
       'motion_loss_weight': 0.8,
       'rnn_cell': 'BasicLSTMCell',
       'batch_size': 32,
       'use_pretrained_encoder': True
    }, cluster_centers_path=cluster_centers_path)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={
            'subtitle': np.array(subtitles),
        }, num_epochs=1, shuffle=False)
    preds = list(estimator.predict(input_fn=predict_input_fn))
    return preds


def predict_sequences(subtitles):
  estimator, model_params = setup_estimator({
      'output_type': 'sequences',
      'motion_loss_weight': 0.9,
      'rnn_cell': 'GRUCell',
      'batch_size': 32,
      'use_pretrained_encoder': False,
      'hidden_size': 128,
      'learning_rate': 0.001,
      'dropout': 0.5,
      'embedding_size': 64,
      'note': '2_extra_layers'
  })

  if not model_params.use_pretrained_encoder:
    def parse_subtitle(subtitle):
      subtitle = subtitle.split(' ')
      subtitle = list(filter(lambda x: len(x.strip()) > 0, map(common.data_utils.clean_word, subtitle)))

      return subtitle
    subtitles = list(map(parse_subtitle, subtitles))

    max_len = max(len(x) for x in subtitles)
    for sub in subtitles:
      sub += [''] * (max_len - len(sub))

  print(np.array(subtitles))
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={
          'subtitle': np.array(subtitles),
      }, num_epochs=1, shuffle=False)
  all_preds = np.array(list(estimator.predict(input_fn=predict_input_fn)))

  def process_prediction(preds):
    print(preds[:, 0])
    end_markers = np.where(preds[:, 0] < 0.3)[0]
    if len(end_markers) == 0:
      n_frames = 150
    else:
      n_frames = end_markers[0]
    print("Length: {} frames.".format(n_frames))
    frames = preds[:n_frames, 1:].tolist()
    angles = list(map(lambda x: common.pose_utils.get_named_pose(x, fmt='angle'), frames))

    return angles

  all_angles = []
  for i in range(all_preds.shape[1]):
    all_angles += [process_prediction(all_preds[:, i, :])]

  return all_angles


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
    # debug_hook = tf_debug.TensorBoardDebugHook("localhost:7000")
    # debug_hook = tf_debug.LocalCLIDebugHook()

    train_input_fn = lambda: data.input_fn(
      '../train.tfrecords',
      batch_size=model_params.batch_size,
      n_epochs=512,
      split_sentences=(model_params.use_pretrained_encoder is False)
    )
    train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn,
      max_steps=15000)

    eval_input_fn = lambda: data.input_fn(
      '../validate.tfrecords',
      batch_size=model_params.batch_size,
      n_epochs=1,
      split_sentences=(model_params.use_pretrained_encoder is False)
    )
    eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      start_delay_secs=60,
      throttle_secs=120)

    # tf.estimator.train_and_evaluate(
    #   estimator,
    #   train_spec,
    #   eval_spec)

    estimator.train(train_input_fn, max_steps=12000)
    train_data_eval = estimator.evaluate(train_input_fn, steps=10)
    eval_data_eval = estimator.evaluate(eval_input_fn)

    return (train_data_eval, eval_data_eval)

  if options.predict:
    subtitle = 'no dont worry its something else'

    if not model_params.use_pretrained_encoder:
      subtitle = subtitle.split(' ')
      subtitle = list(filter(lambda x: len(x.strip()) > 0, map(common.data_utils.clean_word, subtitle)))

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={
            'subtitle': np.array([subtitle]),
        }, num_epochs=1, shuffle=False)
    preds = np.array(list(estimator.predict(input_fn=predict_input_fn)))

    if model_params.output_type == 'sequences':
      print(preds[:, 0, 0])
      end_markers = np.where(preds[:, 0, 0] < 0.3)[0]
      if len(end_markers) == 0:
        n_frames = 150
      else:
        n_frames = end_markers[0]
      print("Length: {} frames.".format(n_frames))
      frames = preds[:n_frames, 0, 1:].tolist()
      n_frames = 500
      angles = list(map(lambda x: common.pose_utils.get_named_pose(x, fmt='angle'), frames))

      with open("predicted_angles.json", "w") as write_file:
        serialized = json.dumps({'clip': angles}).decode('utf-8')
        write_file.write(serialized)
        print("Wrote angles to predicted_angles.json")

      pose = list(
          map(common.pose_utils.get_point_list,
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
  parser.add_argument('--subtitle',
                      dest='subtitle',
                      action='store',
                      help='the subtitle to use for prediction')
  parser.set_defaults(
    train=False,
    predict=False)

  cmd_options = parser.parse_args()

  space = dict(
    hidden_size=[8, 64, 128, 512],
    output_type=['sequences', 'classes'],
    embedding_size=[8, 16, 32, 256],
    dropout=[0.2, 0.5, 0.8],
    motion_loss_weight=[0.1, 0.5, 0.9],
    use_pretrained_encoder=[True, False]
  )

  stars = itertools.product(*space.values())
  names = space.keys()
  csv_names = list(names) + ['loss', 'gesture_loss', 'accuracy', 'global_step', 'train_loss']

  finished_stars = []
  with open('hp-results.csv') as in_file:
    reader = csv.DictReader(in_file, fieldnames=csv_names)
    for constellation in reader:
      finished_stars.append(constellation)

  with open('hp-results.csv', 'ab') as out_file:
    writer = csv.DictWriter(out_file, fieldnames=csv_names)
    writer.writeheader()

    for constellation in stars:
      setup = dict(zip(names, constellation))

      if setup in finished_stars:
        print('Already calculated: {}'.format(setup))
        continue

      train_results, results = run_experiment(setup, cmd_options)
      if 'accuracy' not in results:
        results['accuracy'] = 0.0 # Sequence predictor has no accuracy

      train_to_write = dict(
        train_loss=train_results['gesture_loss'])

      results_to_write = merge_two_dicts(results, setup)
      results_to_write = merge_two_dicts(results_to_write, train_to_write)
      print(results_to_write)
      writer.writerow(results_to_write)
