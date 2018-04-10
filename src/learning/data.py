#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import math
import logging

import numpy as np
import tensorflow as tf

import common.config_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

H36M_NAMES = ['']*32
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
    'Spine',
    'Thorax',
    'Neck/Nose',
    'Head',
    'LShoulder',
    'LElbow',
    'LWrist',
    'RShoulder',
    'RElbow',
    'RWrist'
]

FILTERED_INDICES = [i for i, s in enumerate(H36M_NAMES) if s in FILTERED_NAMES]

N_POSE_FEATURES = len(FILTERED_NAMES) * 3 + 1

# GET TO DA CHOPPA
TERMINATOR_CHAR = '¶'
TERMINATOR_INDEX = 0

DATASET_FILENAME = 'data.hdf5'


def get_input(data_filename='samples.npy',
              force_refresh=False,
              config_filename='config.json',
              batch_size=16,
              n_epochs=1,
              path='/samples/all'):

    config = common.config_utils.load_config(path=config_filename)
    samples, vocab = create_examples(config['clips'])
    characters, poses = map(list, zip(*samples))
    gen_batches = create_batches(characters, poses, vocab, batch_size)

    feature_columns = [
        tf.feature_column.categorical_column_with_identity(
            key='characters', num_buckets=len(vocab))
    ]

    def input_fn():
        dataset = tf.data.Dataset.from_generator(
            gen_batches,
            ({
                'characters': tf.int64,
                'characters_lengths': tf.int64
            }, {
                'poses': tf.float32,
                'poses_lengths': tf.int64
            }),
            ({
                'characters': tf.TensorShape([batch_size, None, len(vocab)]),
                'characters_lengths': tf.TensorShape([batch_size])
            }, {
                'poses': tf.TensorShape([batch_size, None, N_POSE_FEATURES]),
                'poses_lengths': tf.TensorShape([batch_size])
            }))
        dataset = dataset.repeat(n_epochs)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    return input_fn, feature_columns


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


def get_empty_input(vocab):
    return np.zeros((len(vocab),))


def get_empty_output():
    return np.zeros((len(FILTERED_INDICES) * 3 + 1,))


def char2feature(char, vocab):
    index = vocab[char]
    return np.eye(len(vocab))[index]


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
    indices = [vocab[char] for char in subtitle] + [TERMINATOR_INDEX]
    features = np.eye(len(vocab))[np.array(indices).flatten()]
    length = len(subtitle)

    return (features, length)


def poses2labels(poses):
    """Converts a list of poses to a list of labels.

    Args:
      poses: list of (n_poses, n_indices, 3): 3D poses
    Returns:
      ndarray (n_poses + 1, n_filtered_indices + 1): list of labels
        with terminator label + terminator pose added
        (i.e. time = n_poses + 1)
    """
    # Filter keypoints
    poses = np.array(poses)[:, FILTERED_INDICES, :]

    # Flatten array
    poses = np.reshape(poses, (-1, len(FILTERED_INDICES) * 3))

    # Add mask indices (extra first "keypoint" with value 1)
    poses = np.concatenate((
        np.ones((poses.shape[0], 1)),
        poses
    ), axis=1)

    # Append terminator pose (all indices 0)
    poses = np.concatenate((
        poses,
        np.zeros((1, poses.shape[1]))
    ), axis=0)

    poses_length = poses.shape[0]

    return (poses, poses_length)


def pad_features(features, vocab, n):
    return np.append(
        features,
        np.repeat([TERMINATOR_INDEX], n, axis=0),
        axis=0)


def get_label_masks(labels, n):
    #    n * [0] ("not started")
    # ++ len(labels) * [1] ("look at me")
    # ++ 1 * [0] ("terminated")
    return np.append(
        np.zeros((n, 1)),
        np.ones((labels.shape[0], 1)),
        np.zeros((1, 1)),
        axis=0
    )


def pad_labels(labels, n):
    # This will set the terminator label to 0
    # The rest of the values don't matter then
    return np.insert(labels, np.zeros((n, labels.shape[1])), axis=0)


def clip2sample(clip, vocab):
    subtitle = clip['subtitle']
    poses_3d = clip['points_3d']

    features, features_length = subtitle2features(subtitle, vocab)
    labels, labels_length = poses2labels(poses_3d)

    return (features, labels)


def create_examples(clips):
    """Creates a list of examples from given clips.
    Examples all have a different length.
    """
    vocab, _, _ = create_vocab(clips)
    return (
        list(map(lambda clip: clip2sample(clip, vocab),
                 filter(lambda clip: 'points_3d' in clip
                        and len(clip['points_3d']) > 0, clips))), vocab)


def create_batches(characters, poses, vocab, batch_size):
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
    total_length = n_batches * batch_size
    n_padding_samples = total_length - len(characters)
    logging.debug("Got {} samples in total".format(len(characters)))
    logging.debug("Adding {} padding samples".format(n_padding_samples))

    inputs = characters + ([get_empty_input(vocab)] * n_padding_samples)
    outputs = poses + ([get_empty_output()] * n_padding_samples)
    logging.debug("Batching inputs {} and outputs {} with batch size {}"
                  .format(len(inputs), len(outputs), batch_size))

    def generator():
        for i in range(n_batches):
            batch_inputs = inputs[i*batch_size:(i+1)*batch_size]
            batch_outputs = outputs[i*batch_size:(i+1)*batch_size]

            batch_input_lengths = np.array([len(i) for i in batch_inputs],
                                           dtype=np.int32)
            logging.debug("Input lengths: {}".format(batch_input_lengths))
            batch_output_lengths = np.array([len(o) for o in batch_outputs],
                                            dtype=np.int32)
            logging.debug("Output lengths: {}".format(batch_output_lengths))

            padded_inputs = np.array(
                list(itertools.zip_longest(
                    *batch_inputs,
                    fillvalue=get_empty_input(vocab))))
            padded_inputs = np.swapaxes(padded_inputs, 0, 1)
            logging.debug("Padded input shape {}".format(padded_inputs.shape))

            padded_outputs = np.array(
                list(itertools.zip_longest(
                    *batch_outputs,
                    fillvalue=get_empty_output())))
            padded_outputs = np.swapaxes(padded_outputs, 0, 1)
            logging.debug("Padded outputs shape {}"
                          .format(padded_outputs.shape))

            yield ({
                'characters': padded_inputs,
                'characters_lengths': batch_input_lengths
            }, {
                'poses': padded_outputs,
                'outputs_lengths': batch_output_lengths
            })

    return generator
