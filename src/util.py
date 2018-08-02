#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import naoqi  # Needs to be imported before random? Why? No clue.
from random import randint
import os
import os.path
import math
import random
import time
import itertools

import json
import jsonlines

import common.data_utils
import common.pose_utils
import common.bot
import common.visualize
from common.data_utils import try_for_random_clip
from survey import create_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Command(object):

  def __init__(self, parser):
    self._setup_parser(parser)

  def setup_parser(self, parser):
    pass

  def _setup_parser(self, parser):
    self.setup_parser(parser)
    parser.set_defaults(func=self.run)

  def run(self, args):
    raise NotImplementedError()


class RunnerCommand(Command):

  def setup_choices(self):
    raise NotImplementedError()

  def setup_options(self, parser):
    pass

  def setup_parser(self, parser):
    self.setup_choices()
    parser.add_argument(
      'command',
      metavar='command',
      type=str,
      nargs=1,
      help="The command to execute. Options: {}".format(self.command_choices),
      choices=self.command_choices)
    self.setup_options(parser)


class DatasetCommand(RunnerCommand):

  def setup_choices(self):
    self.command_choices = [
      'preprocess',
      'remove-duplicates',
      'add-to',
      'normalize',
      'add-angles',
      'create-tfrecords',
      'create-vocabulary',
      'count-tfrecords'
    ]

  def run(self, args):
    command_name = args.command[0]

    if command_name == 'remove-duplicates':
      common.data_utils.remove_duplicate_clips(*args.args)
    elif command_name == 'add-to':
      common.data_utils.add_clips_to(*args.args)
    elif command_name == 'merge':
      common.data_utils.add_clips_to(*args.args)
    elif command_name == 'normalize':
      common.data_utils.normalize_clips(*args.args)
    elif command_name == 'add-angles':
      common.data_utils.add_clip_angles(*args.args)
    elif command_name == 'preprocess':
      # Cleans, adds angles and overwrites previous clips file
      common.data_utils.normalize_clips(write_path='clips.jsonl')
      common.data_utils.add_clip_angles(write_path='clips.jsonl')
      common.data_utils.create_vocabulary()
    elif command_name == 'write-poses-from-angle-files':
      """Reads a list of files with frames (angle dict on each line)
      and saves those to a file with a pose animation on each line."""
      output_file_name = args.args[0]
      file_names = args.args[1:]

      with jsonlines.open(output_file_name, 'w') as writer:
        for file_name in file_names:
          poses = []
          frames = []
          with jsonlines.open(file_name, 'r') as reader:
            for angles in reader:
              frames.append(angles)
              poses.append(
                  common.pose_utils.get_encoded_pose(
                      common.pose_utils.get_pose_from_angles(angles)).tolist())
          writer.write(dict(points_3d=poses, angles=frames))
    elif command_name == 'create-tfrecords':
      import learning.data
      learning.data.create_tfrecords(*args.args)
    elif command_name == 'create-vocabulary':
      common.data_utils.create_vocabulary()
    elif command_name == 'count-tfrecords':
      import learning.data
      print(learning.data.count_tfrecords(args.args[0]))
    else:
      logger.error('Could not find command {}.\nYour options: {}'
                  .format(command_name, self.command_choices))


class VisualizeCommand(RunnerCommand):

  def setup_choices(self):
    self.command_choices = [
      'pose-format-comparison',
      'sanity-check-2d',
      'crappy-3d-detection-plot',
      'sanity-check-2d-3d',
      'sanity-check-pipeline',
      'pose-vector-plot',
      'sanity-check-cluster-centers',
      'sanity-check-cluster-samples'
    ]

  def run(self, args):
    command_name = args.command[0]

    if command_name == 'pose-format-comparison':
      try_for_random_clip(common.visualize.create_pose_format_comparison)
    elif command_name == 'sanity-check-2d':
      try_for_random_clip(common.visualize.create_sanity_check_2d)
    elif command_name == 'crappy-3d-detection-plot':
      common.visualize.create_crappy_3d_detection_animation()
    elif command_name == 'sanity-check-2d-3d':
      try_for_random_clip(common.visualize.create_sanity_check_2d_3d)
    elif command_name == 'sanity-check-pipeline':
      try_for_random_clip(
        lambda x:common.visualize.create_sanity_check_pipeline(
          x, openpose_output_dir='./output/'))
    elif command_name == 'sanity-check-pipeline-animation':
      try_for_random_clip(
        lambda x:common.visualize.create_sanity_check_pipeline_animation(
          x, openpose_output_dir='./output/'))
    elif command_name == 'pose-vector-plot':
      try_for_random_clip(common.visualize.create_pose_vector_plot)
    elif command_name == 'sanity-check-cluster-centers':
      with open('./cluster-centers.json') as centers_file:
        centers = json.load(centers_file)['clusters']
      common.visualize.create_sanity_check_gesture_grid_animation(centers, 'cluster-centers')
    elif command_name == 'sanity-check-cluster-samples':
      clips = common.data_utils.get_clips()
      cluster_ids = range(1, 9)
      clusters = [filter(lambda x: x['class'] == c, clips) for c in cluster_ids]

      for i, cluster in enumerate(clusters):
        if len(cluster) <= 4:
          print('WARN: Number cluster elements is only {}'.format(len(cluster)))
        sample = random.sample(cluster, 4)
        gestures = [clip['angles'] for clip in sample]
        common.visualize.create_sanity_check_gesture_grid_animation(
          gestures,
          'cluster-{}-samples'.format(str(cluster_ids[i])),
          figsize=(3, 3))
    else:
      logger.error('Could not find command {}.\nYour options: {}'
                  .format(command_name, self.command_choices))


class TestCommand(RunnerCommand):

  def setup_choices(self):
    self.command_choices = [
      'normalization',
      'angle-conversion',
      'zero-pose',
      'angle-plot'
    ]

  def run(self, args):
    command_name = args.command[0]

    if command_name == 'normalization':
      clip = common.data_utils.get_random_clip()
      poses = clip['points_3d']

      common.visualize.animate_3d_poses(poses)
      new_poses = common.data_utils.straighten_frames(poses)
      common.visualize.animate_3d_poses(list(new_poses))
    elif command_name == 'angle-conversion':
      clip = common.data_utils.get_random_clip()
      poses = clip['points_3d']

      angles = map(common.pose_utils.get_pose_angles, poses)
      new_poses = map(common.pose_utils.get_pose_from_angles, angles)
      h36m_poses = map(common.pose_utils.get_encoded_pose, new_poses)
    elif command_name == 'zero-pose':
      nao_h36m = [[0, 0, 0]] * 32
      for key, value in common.pose_utils.NAO_ZERO_POSE.iteritems():
        nao_h36m[common.pose_utils.H36M_NAMES.index(key)] = value
      common.visualize.show_3d_pose(nao_h36m)

      common.visualize.animate_3d_poses(poses)
      common.visualize.animate_3d_poses(list(h36m_poses))
    elif command_name == 'angle-plot':
      clips = common.data_utils.get_clips()
      clip = clips[randint(0, len(clips))]
      if 'angles' not in clip:
        raise ValueError('No angles in this clip')
      angles = clip['angles']
      print json.dumps(angles[0], indent=4)

      common.visualize.show_3d_pose(clip['points_3d'][0])

      named_pose = common.pose_utils.get_pose_from_angles(angles[0])
      nao_h36m = [[0, 0, 0]] * 32
      for key, value in named_pose.iteritems():
        nao_h36m[common.pose_utils.H36M_NAMES.index(key)] = value

      common.visualize.show_3d_pose(nao_h36m)
    else:
      logger.error('Could not find command {}.\nYour options: {}'
                    .format(command_name, self.command_choices))


class BotCommand(RunnerCommand):

  def setup_choices(self):
    self.command_choices = [
      'play-random-clip',
      'play-clusters',
      'say'
    ]

  def setup_options(self, parser):
    parser.add_argument(
        '--bot-address',
        dest='bot_address',
        action='store',
        default='127.0.0.1'
    )
    parser.add_argument(
        '--bot-port',
        dest='bot_port',
        action='store',
        default=9559,
        type=int
    )
    parser.add_argument(
      '--gesture-type',
      dest='gesture_type',
      action='store',
      nargs=1,
      default=['naoqi'],
      help='The type of gesture to generate for the `say` command.',
      choices=['naoqi', 'cluster', 'sequence'])
    parser.add_argument(
      '--subtitle',
      dest='subtitle',
      action='store',
      nargs=1,
      default=['hello world'],
      help='The subtitle to pronouce for the `say` command.')

  def run(self, args):
    command_name = args.command[0]

    if command_name == 'play-random-clip':
      bot = common.bot.BotController(address=args.bot_address, port=args.bot_port)
      clip = common.data_utils.get_random_clip()
      print(clip['subtitle'])
      bot.play_angles(clip['angles'])
    # elif command_name == 'play-file':
    #   bot = common.bot.BotController(address=args.bot_address, port=args.bot_port)
    #   angles = []
    #   with open(args.args[0], "r") as read_file:
    #     angles = json.load(read_file)['clip']
    #   bot.play_angles(angles)
    elif command_name == 'play-clusters':
      clusters = common.data_utils.get_clusters()

      all_angles = []
      for i in args.args[1:]:
        all_angles += clusters[int(i)]

      bot = common.bot.BotController(address=args.bot_address, port=args.bot_port)
      bot.play_angles(all_angles)
    elif command_name == 'say':
      import learning.model

      gesture_type = args.gesture_type[0]
      subtitles = args.subtitle
      bot = common.bot.BotController(address=args.bot_address, port=args.bot_port)

      if gesture_type == 'naoqi':
        bot.say(subtitle)
      elif gesture_type == 'cluster':
        clusters = common.data_utils.get_clusters()
        predictions = learning.model.predict_classes(subtitles, cluster_centers_path='cluster-centers.json')
        animation = list(itertools.chain.from_iterable(clusters[x] for x in predictions))
        bot.play_angles(animation)
      elif gesture_type == 'sequence':
        animation = list(itertools.chain.from_iterable(learning.model.predict_sequences(subtitles)))
        bot.play_angles(animation)
      else:
        logger.error('Did not recognize that gesture type.')
    else:
      logger.error('Could not find command {}.\nYour options: {}'
                  .format(command_name, self.command_choices))


class SurveyCommand(RunnerCommand):

  def setup_choices(self):
    self.command_choices = [
      'create-question'
    ]

  def setup_options(self, parser):
    parser.add_argument(
        '--bot-address',
        dest='bot_address',
        action='store',
        default='127.0.0.1'
    )
    parser.add_argument(
        '--bot-port',
        dest='bot_port',
        action='store',
        default=9559,
        type=int
    )

  def run(self, args):
    command_name = args.command[0]

    if command_name == 'create-question':
      create_question(args.args[0], do_record_screen=False, do_generate_tts=False)
    else:
      logger.error('Could not find command {}.\nYour options: {}'
                  .format(command_name, self.command_choices))


class MiscCommand(RunnerCommand):

  def setup_choices(self):
    self.command_choices = [
      'move-2d-processed-images'
    ]

  def run(self, args):
    command_name = args.command[0]

    if command_name == 'move-2d-processed-images':
      common.data_utils.move_2d_finished_images(*args.args)
    else:
      logger.error('Could not find command {}.\nYour options: {}'
                  .format(command_name, self.command_choices))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Perform all kinds of actions around this gesture synthesis method.')
  subparsers = parser.add_subparsers(help='category of the command to execute')

  dataset_parser = subparsers.add_parser('dataset')
  DatasetCommand(dataset_parser)

  visualize_parser = subparsers.add_parser('visualize')
  VisualizeCommand(visualize_parser)

  test_parser = subparsers.add_parser('test')
  TestCommand(test_parser)

  bot_parser = subparsers.add_parser('bot')
  BotCommand(bot_parser)

  survey_parser = subparsers.add_parser('survey')
  SurveyCommand(survey_parser)

  misc_parser = subparsers.add_parser('misc')
  MiscCommand(misc_parser)

  args = parser.parse_args()
  args.func(args)
