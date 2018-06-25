#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import naoqi  # Needs to be imported before random? Why? No clue.
from random import randint

import json
import jsonlines

import common.data_utils
import common.pose_utils
import common.bot
import common.visualize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_primitives():
  from SFA_Python.src.timeseries.TimeSeries import TimeSeries
  import numpy as np

  clips = common.data_utils.get_clips()

  dataset = {
      "type": "MV",
      "Samples": 50,
      "Dimensions": len(clips[0]["angles"][0].keys()),
      "Labels": [-1] * 50,
  }

  for i, clip in enumerate(clips):
    frames = np.array(
        list(map(common.pose_utils.get_angle_list, clip["angles"])))
    dataset[i] = {
        j: TimeSeries((frames[:, j]).tolist(), -1)
        for j in range(frames.shape[1])
    }
    if i >= 50:
      break

  from SFA_Python.src.classification.MUSEClassifier import MUSEClassifier
  classifier = MUSEClassifier(0)
  output = classifier.fit(dataset)
  print(output)
  return output


if __name__ == '__main__':
  command_choices = [
      'remove-duplicate-clips', 'add-clips-to', 'merge',
      'move-2d-finished_images', 'normalize', 'add-angles', 'sample',
      'test-angle-conversion', 'bot-play', 'bot-play-random-clip',
      'test-zero-pose', 'test-normalization', 'test-plot-angles',
      'get-poses-from-angle-files', 'bot-play-clusters', 'create-tfrecords',
      'create-primitives', 'create-vocabulary'
  ]

  parser = argparse.ArgumentParser(description='Manipulate clip data files.')
  parser.add_argument(
      'command',
      metavar='command',
      type=str,
      nargs=1,
      help="The command to execute. One of {}".format(command_choices),
      choices=command_choices)
  parser.add_argument(
      'args',
      metavar='args',
      type=str,
      nargs='*',
      help='Arguments for the command')

  args = parser.parse_args()

  command_name = args.command[0]
  if command_name == 'remove-duplicate-clips':
    common.data_utils.remove_duplicate_clips(*args.args)
  elif command_name == 'add-clips-to':
    common.data_utils.add_clips_to(*args.args)
  elif command_name == 'merge':
    common.data_utils.add_clips_to(*args.args)
  elif command_name == 'move-2d-finished-images':
    common.data_utils.move_2d_finished_images(*args.args)
  elif command_name == 'normalize':
    common.data_utils.normalize_clips(*args.args)
  elif command_name == 'add-angles':
    common.data_utils.add_clip_angles(*args.args)
  elif command_name == 'test-normalization':
    clip = common.data_utils.get_random_clip()
    poses = clip['points_3d']

    common.visualize.animate_3d_poses(poses)
    new_poses = common.data_utils.straighten_frames(poses)
    common.visualize.animate_3d_poses(list(new_poses))
  elif command_name == 'test-angle-conversion':
    clip = common.data_utils.get_random_clip()
    poses = clip['points_3d']

    angles = map(common.pose_utils.get_pose_angles, poses)
    new_poses = map(common.pose_utils.get_pose_from_angles, angles)
    h36m_poses = map(common.pose_utils.get_encoded_pose, new_poses)

    common.visualize.animate_3d_poses(poses)
    common.visualize.animate_3d_poses(list(h36m_poses))
  elif command_name == 'bot-play-random-clip':
    bot = common.bot.BotController(*args.args)
    clip = common.data_utils.get_random_clip()
    print(clip['subtitle'])
    bot.play_angles(clip['angles'])
  elif command_name == 'bot-play':
    bot = common.bot.BotController(port=args.args[1])
    angles = []
    with open(args.args[0], "r") as read_file:
      angles = json.load(read_file)['clip']

    bot.play_angles(angles)
  elif command_name == 'test-zero-pose':
    nao_h36m = [[0, 0, 0]] * 32
    for key, value in common.pose_utils.NAO_ZERO_POSE.iteritems():
      nao_h36m[common.pose_utils.H36M_NAMES.index(key)] = value
    common.visualize.show_3d_pose(nao_h36m)
  elif command_name == 'test-plot-angles':
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
  elif command_name == 'write-poses-from-angle-files':
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
  elif command_name == 'bot-play-clusters':
    clusters = []
    with jsonlines.open('stats/center-points.jsonl', 'r') as reader:
      for cluster in reader:
        clusters.append(cluster)

    all_angles = []
    for i in args.args[1:]:
      all_angles += clusters[int(i)]['angles']

    bot = common.bot.BotController(args.args[0])
    bot.play_angles(all_angles)
  elif command_name == 'create-tfrecords':
    import learning.data
    learning.data.create_tfrecords(*args.args)
  elif command_name == 'create-primitives':
    create_primitives()
  elif command_name == 'create-vocabulary':
    common.data_utils.create_vocabulary()
  else:
    logger.error("Command {} not found.".format(command_name))
