#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from future.builtins import *
from future.builtins.disabled import *

import math
import logging
import os.path
import json

# Itertools has a different name in Python 2/3
try:
  from itertools import zip_longest as zip_longest
except:
  from itertools import izip_longest as zip_longest

import numpy as np
import tensorflow as tf

import common.data_utils
from common.pose_utils import Gesture

logger = logging.getLogger(__name__)

H36M_NAMES = [''] * 32
H36M_NAMES[0] = 'Hip'
H36M_NAMES[1] = 'RHip'
H36M_NAMES[2] = 'RKnee'
H36M_NAMES[3] = 'RFoot'
H36M_NAMES[6] = 'LHip'
H36M_NAMES[7] = 'LKnee'
H36M_NAMES[8] = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

FILTERED_NAMES = [
    'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist'
]

FILTERED_INDICES = [i for i, s in enumerate(H36M_NAMES) if s in FILTERED_NAMES]

# N_POSE_FEATURES = len(FILTERED_NAMES) * 3 + 1
POSE_DTYPE = tf.float32

# GET TO DA CHOPPA
TERMINATOR_CHAR = '¶'
TERMINATOR_INDEX = 0

REFERENCE_POSE = np.asarray([[0.0, 0.0, 0.0], [
    -0.0887402690899655, -0.010927448423188232, -1.3783571838113866e-06
], [-0.0496766891833195, 0.28102529317484287, 0.07819976467458818], [
    -0.0177358881823574, 0.5584250598640099, 0.20064917741479613
], [0.02217396190646866, 0.4953691367041059, 0.10915803454885464], [
    0.021757968272520696, 0.49177365478202334, 0.10837633257042446
], [0.0887397178773482, 0.010927383830083987, 1.4078720891584565e-06], [
    0.13316234461459953, 0.31040605919950554, 0.02736759508205667
], [0.1896239296630329, 0.5747375465845473, 0.14496773111464492], [
    0.02239526476081422, 0.4996991633139912, 0.1092275258098906
], [0.021777027291472136, 0.49155472207438705, 0.10735401760005389], [
    -2.64987594476859e-06, -5.7410552748649905e-05, -1.263302212493416e-05
], [-0.004648935490533353, -0.16217688980944528, 0.010245545947778711], [
    -0.021594194090241, -0.3231288880157715, -0.04285522506092727
], [-0.027409924426055627, -0.3573723350024327, -0.11204637255034618], [
    -0.025996103555896884, -0.42526245341545266, -0.10276095602461485
], [-0.013724559997372158, -0.2904125788249102, -0.06386276622049653], [
    0.06868420186327358, -0.2873658519829194, -0.035238680070871976
], [0.16282767130768935, -0.14371808992288698, 0.006981914953063249], [
    0.08745360266308361, -0.02542532429423124, -0.06706648380628974
], [-0.004210430861146672, -0.08021130232454704, -0.018401380521093785], [
    -0.005246016755371665, -0.10131008377762633, -0.02293184655974524
], [-0.003989437466705651, -0.07195110437303866, -0.016646778633422576], [
    -0.003989437466705651, -0.07195110437303866, -0.016646778633422576
], [-0.013724559997372158, -0.2904125788249102, -0.06386276622049653], [
    -0.10594628737328486, -0.28114379053840266, -0.015146674570486504
], [-0.1706416447960403, -0.13772706535133283, 0.054841519444767554], [
    -0.14000681171449467, -0.030145894458783185, -0.02751885964926468
], [-0.006089899035947899, -0.1125800079119398, -0.02420975829783157], [
    -0.007053163382197237, -0.13408596033560885, -0.029019971522748305
], [-0.006213603162041361, -0.11071320657246192, -0.023790547315948714], [
    -0.006213603162041361, -0.11071320657246192, -0.023790547315948714
]])


def create_vocab(clips):
  subtitles = map(lambda clip: clip['subtitle'], clips)

  unique_chars = set()
  for sentence in subtitles:
    unique_chars = unique_chars | set(sentence)

  unique_chars = list(unique_chars)
  vocab_size = len(unique_chars)
  vocab_index_dict = {'¶': TERMINATOR_INDEX}  # Terminator character
  index_vocab_dict = {0: TERMINATOR_CHAR}

  for i, char in enumerate(unique_chars):
    vocab_index_dict[char] = i
    index_vocab_dict[i] = char
  return vocab_index_dict, index_vocab_dict, vocab_size


def get_empty_input():
  return TERMINATOR_INDEX


def get_empty_output(n_labels):
  return np.zeros((n_labels,))


def char2feature(char, vocab):
  return vocab[char]


def features2chars(ids, vocab):
  return [vocab[i] for i in ids]


def subtitle2features(subtitle, vocab):
  """Converts a subtitle text to a list of features.

    Args:
      subtitle: string of subtitle
      vocab: dict<character, index> to map the subtitle
    Returns:
      ndarray (subtitle_length + 1, vocab_size): converted subtitle with
        terminator symbol added (i.e. time = subtitle_length + 1)
    """
  features = [vocab[char] for char in subtitle] + [TERMINATOR_INDEX]
  length = len(subtitle)

  return (features, length)


def subtitle2subtitle(subtitle, vocab=None):
  """Converts a subtitle text to a list of features.

    Args:
      subtitle: string of subtitle
      vocab: dict<character, index> to map the subtitle
    Returns:
      ndarray (subtitle_length + 1, vocab_size): converted subtitle with
        terminator symbol added (i.e. time = subtitle_length + 1)
    """
  length = len(subtitle)

  return (subtitle, length)


def clip2labels(clip):
  """Converts a list of poses to a list of labels.

    Args:
      clip
    Returns:
      ndarray (n_poses + 1, n_filtered_indices + 1): list of labels
        with terminator label + terminator pose added
        (i.e. time = n_poses + 1)
    """
  poses = clip['points_3d']

  # Filter keypoints
  poses = np.array(poses)[:, FILTERED_INDICES, :]

  # Flatten array
  poses = np.reshape(poses, (-1, len(FILTERED_INDICES) * 3))

  # Add mask indices (extra first "keypoint" with value 1)
  poses = np.concatenate((np.ones((poses.shape[0], 1)), poses), axis=1)

  # Append terminator pose (all indices 0)
  poses = np.concatenate((poses, np.zeros((1, poses.shape[1]))), axis=0)

  poses_length = poses.shape[0]

  return (poses, poses_length)


def clip2angles(clip):
  """Converts a list of poses to a list of angles.

    Args:
      clip
    Returns:
      ndarray (n_poses + 1, n_joints + 1): list of angles
        with terminator label + terminator pose added
        (i.e. time = n_poses + 1)
    """
  poses = clip['points_3d']

  angles = np.asarray(list(map(common.pose_utils.get_pose_angle_list, poses)))

  # Add mask indices (extra first "joint" with value 1)
  angles = np.concatenate((np.ones((angles.shape[0], 1)), angles), axis=1)

  # Append terminator pose (all indices 0)
  angles = np.concatenate((angles, np.zeros((1, angles.shape[1]))), axis=0)

  angles_length = angles.shape[0]

  return (angles, angles_length, clip['cluster'])


def pad_features(features, vocab, n):
  return np.append(features, np.repeat([TERMINATOR_INDEX], n, axis=0), axis=0)


def get_label_masks(labels, n):
  #    n * [0] ("not started")
  # ++ len(labels) * [1] ("look at me")
  # ++ 1 * [0] ("terminated")
  return np.append(
      np.zeros((n, 1)), np.ones((labels.shape[0], 1)), np.zeros((1, 1)), axis=0)


def pad_labels(labels, n):
  # This will set the terminator label to 0
  # The rest of the values don't matter then
  return np.insert(labels, np.zeros((n, labels.shape[1])), axis=0)


def clip2sample(clip,
                vocab,
                get_features=subtitle2features,
                get_labels=clip2labels):
  subtitle = clip['subtitle']
  poses_3d = clip['points_3d']

  features, features_length = get_features(subtitle, vocab)
  labels, labels_length, cluster = get_labels(clip)

  return (features, labels, cluster)


def get_poses(path=common.data_utils.DEFAULT_CLIPS_PATH):
  clips = common.data_utils.get_clips(path)
  poses = list(
      map(lambda c: c['points_3d'],
          filter(lambda c: 'points_3d' in c and len(c['points_3d']) > 0,
                 clips)))
  poses = np.vstack(poses)[:, FILTERED_INDICES, :]
  return poses


def create_examples(clips,
                    get_features=subtitle2features,
                    get_labels=clip2labels):
  """Creates a list of examples from given clips.
    Examples all have a different length.
    """
  vocab, _, _ = create_vocab(clips)
  return (list(
      map(lambda clip: clip2sample(clip, vocab, get_features=get_features, get_labels=get_labels),
          filter(
              lambda clip: 'points_3d' in clip and len(clip['points_3d']) > 0,
              clips))), vocab)


def create_examples_angles(clips):
  return create_examples(clips, get_labels=clip2angles)


def create_batches(characters, poses, batch_size):
  """Splits samples in batch_size batches and pads each batch as
    necessary.

    Args:
      characters: list of subtitles
      poses: list of corresponding poses
      batch_size: number of samples in a batch
    Returns:
      list of (batch_inputs, batch_outputs,
        batch_input_lengths, batch_output_lengths).
      batch_inputs[i].shape == (batch_size, max_input_time, n_features)
      batch_outputs[i].shape == (batch_size, max_output_time, n_labels)
    """
  n_batches = math.ceil(len(characters) / batch_size)
  logging.debug("Got {} samples in total".format(len(characters)))

  n_labels = poses[0][0].shape[0]

  logging.info("Total poses: {}".format(sum(len(i) for i in poses)))

  # inputs = characters + ([get_empty_input()] * n_padding_samples)
  # outputs = poses + ([get_empty_output()] * n_padding_samples)
  inputs = characters
  outputs = poses
  logging.debug("Batching inputs {} and outputs {} with batch size {}".format(
      len(inputs), len(outputs), batch_size))

  def generator():
    for i in range(n_batches):
      batch_start = i * batch_size
      batch_end = min((i + 1) * batch_size, len(characters))
      this_batch_size = batch_end - batch_start

      batch_inputs = inputs[batch_start:batch_end]
      batch_outputs = outputs[batch_start:batch_end]

      batch_input_lengths = np.array(
          [len(i) for i in batch_inputs], dtype=np.int32)
      logging.debug("Input lengths: {}".format(batch_input_lengths))
      batch_output_lengths = np.array(
          [len(o) for o in batch_outputs], dtype=np.int32)
      logging.debug("Output lengths: {}".format(batch_output_lengths))

      padded_inputs = np.array(
          list(zip_longest(*batch_inputs, fillvalue=get_empty_input())))
      padded_inputs = np.swapaxes(padded_inputs, 0, 1)

      padded_outputs = np.array(
          list(
              zip_longest(*batch_outputs,
                          fillvalue=get_empty_output(n_labels))))
      padded_outputs = np.swapaxes(padded_outputs, 0, 1)

      if this_batch_size < batch_size:
        n_padding_samples = batch_size - this_batch_size
        max_input_time = padded_inputs.shape[1]
        max_output_time = padded_outputs.shape[1]

        padded_inputs = np.concatenate(
            (padded_inputs,
             np.tile([get_empty_input()] * max_input_time,
                     (n_padding_samples, 1))))
        padded_outputs = np.concatenate(
            (padded_outputs,
             np.tile(
                 get_empty_output(n_labels),
                 (n_padding_samples, max_output_time, 1))))

        batch_input_lengths = np.append(batch_input_lengths,
                                        [0] * n_padding_samples)
        batch_output_lengths = np.append(batch_output_lengths,
                                         [0] * n_padding_samples)

      logging.debug("Padded input shape {}".format(padded_inputs.shape))
      logging.debug("Padded outputs shape {}".format(padded_outputs.shape))

      yield ({
          'characters': padded_inputs,
          'characters_lengths': batch_input_lengths
      }, {
          'poses': padded_outputs,
          'poses_lengths': batch_output_lengths
      })

  return generator


def create_batches_sentences(sentences, poses, classes, batch_size):
  """Splits samples in batch_size batches and pads each batch as
    necessary.

    Args:
      sentences: list of subtitles
      poses: list of corresponding poses
      batch_size: number of samples in a batch
    Returns:
      list of (batch_inputs, batch_outputs,
        batch_input_lengths, batch_output_lengths).
      batch_inputs[i].shape == (batch_size, max_input_time, n_features)
      batch_outputs[i].shape == (batch_size, max_output_time, n_labels)
    """
  n_batches = int(math.ceil(len(sentences) / batch_size))
  logging.debug("Got {} samples in total".format(len(sentences)))
  logging.info("Creating {} batches.".format(n_batches))

  n_labels = poses[0][0].shape[0]

  logging.info("Total poses: {}".format(sum(len(i) for i in poses)))

  inputs = sentences
  outputs = poses
  logging.debug("Batching inputs {} and outputs {} with batch size {}".format(
      len(inputs), len(outputs), batch_size))

  def generator():
    for i in range(n_batches):
      batch_start = i * batch_size
      batch_end = min((i + 1) * batch_size, len(sentences))
      this_batch_size = batch_end - batch_start

      batch_inputs = inputs[batch_start:batch_end]
      batch_outputs = outputs[batch_start:batch_end]

      batch_input_lengths = np.array(
          [len(i) for i in batch_inputs], dtype=np.int32)
      logging.debug("Input lengths: {}".format(batch_input_lengths))
      batch_output_lengths = np.array(
          [len(o) for o in batch_outputs], dtype=np.int32)
      logging.debug("Output lengths: {}".format(batch_output_lengths))

      padded_inputs = batch_inputs

      padded_outputs = np.array(
          list(
              zip_longest(*batch_outputs,
                          fillvalue=get_empty_output(n_labels))))
      padded_outputs = np.swapaxes(padded_outputs, 0, 1)

      if this_batch_size < batch_size:
        n_padding_samples = batch_size - this_batch_size
        max_output_time = padded_outputs.shape[1]

        padded_inputs = np.concatenate((padded_inputs,
                                        np.asarray([''] * n_padding_samples)))
        padded_outputs = np.concatenate(
            (padded_outputs,
             np.tile(
                 get_empty_output(n_labels),
                 (n_padding_samples, max_output_time, 1))))

        batch_input_lengths = np.append(batch_input_lengths,
                                        [0] * n_padding_samples)
        batch_output_lengths = np.append(batch_output_lengths,
                                         [0] * n_padding_samples)

      logging.debug("Padded outputs shape {}".format(padded_outputs.shape))

      yield ({
          'subtitles': padded_inputs,
          'subtitle_lengths': batch_input_lengths
      }, {
          'poses': padded_outputs,
          'poses_lengths': batch_output_lengths,
          'class': classes[batch_start:batch_end]
      })

  return generator


def get_input(clips_path=common.data_utils.DEFAULT_CLIPS_PATH,
              batch_size=16,
              n_epochs=1,
              get_examples=create_examples):

  clips = common.data_utils.get_clips(path=clips_path)
  samples, vocab = get_examples(clips)
  characters, poses = map(list, zip(*samples))
  gen_batches = create_batches(characters, poses, batch_size)

  feature_columns = [
      tf.feature_column.categorical_column_with_identity(
          key='characters', num_buckets=len(vocab))
  ]

  n_labels = poses[0][0].shape[0]

  def input_fn():
    dataset = tf.data.Dataset.from_generator(
        gen_batches, ({
            'characters': tf.int32,
            'characters_lengths': tf.int32
        }, {
            'poses': POSE_DTYPE,
            'poses_lengths': tf.int32
        }), ({
            'characters': tf.TensorShape([batch_size, None]),
            'characters_lengths': tf.TensorShape([batch_size])
        }, {
            'poses': tf.TensorShape([batch_size, None, n_labels]),
            'poses_lengths': tf.TensorShape([batch_size])
        }))
    dataset = dataset.repeat(n_epochs)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

  return input_fn, feature_columns, len(vocab), vocab, n_labels


def get_input_angles(**kwargs):
  return get_input(get_examples=create_examples_angles, **kwargs)


def get_input_sentences(clips_path=common.data_utils.DEFAULT_CLIPS_PATH,
                        batch_size=16,
                        n_epochs=1):

  clips = common.data_utils.get_clips(path=clips_path)
  samples, vocab = create_examples(
      clips, get_features=subtitle2subtitle, get_labels=clip2angles)
  subtitles, poses, classes = map(list, zip(*samples))
  gen_batches = create_batches_sentences(subtitles, poses, classes, batch_size)
  n_labels = poses[0][0].shape[0]

  feature_columns = [
      tf.feature_column.categorical_column_with_identity(
          key='characters', num_buckets=len(vocab))
  ]

  def input_fn():
    dataset = tf.data.Dataset.from_generator(
        gen_batches, ({
            'subtitles': tf.string,
            'subtitle_lengths': tf.int32
        }, {
            'poses': POSE_DTYPE,
            'poses_lengths': tf.int32,
            'class': tf.int32
        }), ({
            'subtitles': tf.TensorShape([batch_size]),
            'subtitle_lengths': tf.TensorShape([batch_size])
        }, {
            'poses': tf.TensorShape([batch_size, None, n_labels]),
            'poses_lengths': tf.TensorShape([batch_size]),
            'class': tf.TensorShape([batch_size])
        }))
    dataset = dataset.repeat(n_epochs)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

  return input_fn, feature_columns, len(vocab), vocab, n_labels


def add_sequence_length_indicator(angle_list):
  return [1.] + angle_list


def add_length_indicator(frames):
  n_frames = float(len(frames))
  output = []
  for i, angles in enumerate(frames):
    output += [[(n_frames - float(i) - 1) / n_frames] + angles]
  return output


def add_zero_pose(frames):
  return frames + [[-1.] + [0.] * len(common.pose_utils.ANGLE_NAMES_ORDER)]


def _str_feature(value):
  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _floats_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def create_tfrecords(clips_path=common.data_utils.DEFAULT_CLIPS_PATH,
                     tfrecords_path='./'):
  clips = common.data_utils.get_clips(clips_path)
  n_clips = len(clips)

  splits = [
    (
      0.7,
      os.path.join(tfrecords_path, 'train.tfrecords')
    ),
    (
      0.15,
      os.path.join(tfrecords_path, 'test.tfrecords')
    ),
    (
      0.15,
      os.path.join(tfrecords_path, 'validate.tfrecords')
    ),
  ]

  start_index = 0
  for size, file_name in splits:
    end_index = min(int(math.ceil(n_clips * size + start_index)), n_clips)
    write_clips_to_tfrecords(clips[start_index:end_index], file_name)
    start_index = end_index


def write_clips_to_tfrecords(clips, file_name):
  writer = tf.python_io.TFRecordWriter(file_name)

  for clip in clips:
    points_3d = np.asarray(clip['points_3d'])
    angles = list(map(common.pose_utils.get_angle_list, clip['angles']))
    angles = add_length_indicator(angles)
    angles = add_zero_pose(angles)
    n_frames = points_3d.shape[0] + 1

    context = {
        'id': _str_feature(clip['id']),
        'class': _int_feature(clip['class'] - 1), # R results are 1-indexed, fix it in dataset
        'n_frames': _int_feature(n_frames),
        'subtitle': _str_feature(clip['subtitle'])
    }
    context_features = tf.train.Features(feature=context)

    sequences = {
        'points_3d':
        tf.train.FeatureList(feature=[
            _floats_feature(np.array(frame).flatten().tolist())
            for frame in clip['points_3d']
        ]),
        'angles':
        tf.train.FeatureList(
            feature=[_floats_feature(frame) for frame in angles])
    }
    sequence_features = tf.train.FeatureLists(feature_list=sequences)

    example = tf.train.SequenceExample(
        context=context_features, feature_lists=sequence_features)

    serialized = example.SerializeToString()
    writer.write(serialized)

def parse_tfrecord(serialized):
  context_features = {
      'class': tf.FixedLenFeature(shape=[], dtype=tf.int64),
      'subtitle': tf.FixedLenFeature(shape=[], dtype=tf.string),
      'n_frames': tf.FixedLenFeature(shape=[], dtype=tf.int64)
  }

  sequence_features = {
      'points_3d':
      tf.FixedLenSequenceFeature(shape=[32, 3], dtype=tf.float32),
      'angles':
      tf.FixedLenSequenceFeature(
          shape=[len(common.pose_utils.ANGLE_NAMES_ORDER) + 1], dtype=tf.float32)
  }

  context, sequence = tf.parse_single_sequence_example(
      serialized=serialized,
      context_features=context_features,
      sequence_features=sequence_features)

  return context, sequence


def create_feature_label_pair(context, sequence):
  return ({
      'subtitle': context['subtitle']
  }, {
      'class': context['class'],
      'n_frames': context['n_frames'],
      'angles': sequence['angles'],
      'points': sequence['points_3d']
  })


def tf_clean_word(word):
  return tf.regex_replace(
      word,
      '[ 0123456789\.\,;:\?!\(\)\[\]\{\}"\'\<\>%]',
      '')


def split_subtitle(features, labels):
  words = tf.string_split([features['subtitle']])
  clean_words = tf.SparseTensor(words.indices, tf.map_fn(tf_clean_word, words.values), words.dense_shape)
  clean_words = tf.sparse_tensor_to_dense(clean_words, default_value='')[0]

  return ({
      'subtitle': clean_words,
  }, labels)


def normalize(data, mean, std):
  return (data - mean) / std


def unnormalize(normalized_data, mean, std):
  return normalized_data * std + mean


def normalize_angles(features, labels, mean, std):
  return (features, {
      'class': labels['class'],
      'n_frames': labels['n_frames'],
      'angles': normalize(labels['angles'], mean, std),
      'points': labels['points']
  })


def make_time_major(features, labels):
  angles = tf.transpose(labels['angles'], [1, 0, 2])
  points = tf.transpose(labels['points'], [1, 0, 2, 3])

  return (features, {
      'class': labels['class'],
      'n_frames': labels['n_frames'],
      'angles': angles,
      'points': points
  })


def input_fn(filenames, batch_size=32, buffer_size=2048, n_epochs=None,
             split_sentences=False, do_shuffle=True):
  dataset = tf.data.TFRecordDataset(filenames=filenames)

  config_path = common.data_utils.DEFAULT_CONFIG_PATH
  if os.path.isfile(config_path):
    with open(config_path) as config_file:
      config = json.load(config_file)

    mean = config['angle_stats']['mean']
    std = config['angle_stats']['std']
  else:
    logger.warn('No angle mean or stddev found')
    mean = 0
    std = 1

  dataset = dataset.map(parse_tfrecord)
  dataset = dataset.map(create_feature_label_pair)
  if split_sentences:
    dataset = dataset.map(split_subtitle)
  dataset = dataset.map(lambda x, y: normalize_angles(x, y, mean, std))
  if do_shuffle:
    dataset = dataset.shuffle(buffer_size=buffer_size)
  dataset = dataset.padded_batch(
      batch_size,
      padded_shapes=({
          'subtitle': [None] if split_sentences else []
      }, {
          'class': [],
          'n_frames': [],
          'angles': [None, len(common.pose_utils.ANGLE_NAMES_ORDER) + 1],
          'points': [None, len(common.pose_utils.H36M_NAMES), 3]
      }))
  dataset = dataset.map(make_time_major)

  dataset = dataset.repeat(n_epochs)

  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()

  return features, labels


def count_tfrecords(file_name):
  return sum(1 for _ in tf.python_io.tf_record_iterator(file_name))


def print_some_tfrecords(file_name, motion_loss_weight=0.5):

  # Load normalization parameters
  config_path = common.data_utils.DEFAULT_CONFIG_PATH
  if os.path.isfile(config_path):
    with open(config_path) as config_file:
      config = json.load(config_file)
    mean = config['angle_stats']['mean']
    std = config['angle_stats']['std']
  else:
    logger.warn('No angle stats found')
    mean = 0
    std = 1

  # Get centers
  centers = common.data_utils.get_clusters()
  centers_lengths = [len(center) for center in centers]
  centers = [Gesture(frames).as_list(fmt='angles') for frames in centers]
  centers = [add_length_indicator(center) for center in centers]
  centers = [normalize(np.asarray(center), np.asarray(mean), np.asarray(std)) for center in centers]

  # Pad until maximum length
  frame_length = len(centers[0][0])
  max_center_len = max(x.shape[0] for x in centers)
  centers = np.array([np.concatenate([center, np.zeros((max_center_len - center.shape[0], frame_length))]) for center in centers])
  centers = np.swapaxes(centers, 0, 1)

  with tf.Session() as sess:
    centers = tf.constant(centers)
    print('Shape of centers: {}'.format(centers.shape))
    # centers.shape == (time_index, cluster_index, angle_index)

    gesture_losses = []
    for r in tf.python_io.tf_record_iterator(file_name):
      record = parse_tfrecord(r)
      cluster, angles, len_labels = record[0]['class'], record[1]['angles'], record[0]['n_frames']

      seq_weights = tf.tile(
          tf.transpose([tf.sequence_mask(len_labels)]),
          multiples=[1, 11])

      output = tf.gather(centers, cluster, axis=1)
      label = angles

      # Pad/slice output so it matches ground truth
      output = tf.pad(output, [[0, tf.maximum(0, tf.shape(label)[0] - tf.shape(output)[0])], [0, 0]])
      output = tf.slice(output, [0, 0], tf.shape(label))

      position_loss = tf.losses.mean_squared_error(
        output, label, weights=seq_weights)

      output_diff = output[1:, :] - output[:-1, :]
      label_diff = label[1:, :] - label[:-1, :]
      motion_loss = tf.losses.mean_squared_error(
          output_diff, label_diff, weights=seq_weights[:-1, :])

      gesture_loss = ((1.0 - motion_loss_weight) * position_loss +
                      motion_loss_weight * motion_loss)

      gesture_losses.append(gesture_loss)

    gesture_losses = tf.stack(gesture_losses)
    mean_loss = tf.reduce_mean(gesture_losses)
    return sess.run(mean_loss)
