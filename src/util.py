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


def create_question(bot_port):
  import learning.model
  import common.watson

  # Write question data
  clip = common.data_utils.get_random_clip()
  video_order = [0, 1, 2]
  random.shuffle(video_order)
  with jsonlines.open('questions.jsonl', mode='a') as writer:
    question = dict(
        id=clip['id'],
        subtitle=clip['subtitle'],
        angles_expected=clip['angles'],
        class_expected=clip['class'],
        video_order=video_order)
    writer.write(question)

  # Generate TTS audio clip
  file_name_speech = os.path.join(common.data_utils.DEFAULT_TTS_PATH, clip['id'] + '.wav')
  common.watson.write_tts_clip(file_name_speech, clip['subtitle'])

  # Record clips
  # Define some functions
  from subprocess import Popen
  from common.bot import BotController
  bot = BotController(bot_port)

  def record_screen(tag, time):
    file_name = os.path.join(common.data_utils.DEFAULT_VIDEO_PATH, clip['id'] + '--' + tag + '.mp4')
    p_recording = Popen([
        'ffmpeg',
        '-framerate', '25',
        '-f', 'x11grab',
        '-s', '560x752',
        '-i', ':1+1280,270',
        '-t', str(time),
        file_name
    ])

    return p_recording, file_name

  def record_pose_animation(frames, tag):
    time = int(math.ceil(len(frames) / 25)) + 1  # Extra second for margin
    proc, file_name = record_screen(tag, time)
    bot.play_angles(frames)
    output = proc.wait()
    print(output)

    return file_name

  # Now, use them
  bot.reset_pose()
  file_name_expected = record_pose_animation(clip['angles'], 'expected')

  bot.reset_pose()
  predicted_class = learning.model.predict_class(clip['subtitle'])
  clusters = common.data_utils.get_clusters()
  cluster = common.data_utils.get_clusters()[predicted_class]
  file_name_predicted = record_pose_animation(cluster, 'predicted')

  bot.reset_pose()
  time_expected = int(math.ceil(len(clip['angles']) / 25)) + 1
  proc, file_name_nao = record_screen('nao', time_expected)
  bot.say(clip['subtitle'])
  output = proc.wait()
  print(output)

  # Merge it all
  video_file_names = [file_name_expected, file_name_predicted, file_name_nao]
  proc = Popen([
      'ffmpeg',
      '-i', video_file_names[video_order[0]],
      '-i', video_file_names[video_order[1]],
      '-i', video_file_names[video_order[2]],
      '-i', file_name_speech,
      '-filter_complex', ('[0:v][1:v][2:v]hstack=inputs=3[v];' +
                          "[v]drawtext=text=" + clip['subtitle'] + ":fontfile=DejaVuSans\\\:style=Bold" + 
                          ":x=(main_w/2-text_w/2)" +
                          ":y=(main_h-(text_h*2)):fontsize=48:fontcolor=white" +
                          ":borderw=2 [v]"),
      '-map', '[v]',
      '-map', '3:a:0',
      os.path.join(common.data_utils.DEFAULT_VIDEO_PATH, clip['id'] + '--merged.mp4')
  ])
  proc.wait()
  print("Recorded. Saved merged video file.")


if __name__ == '__main__':
  command_choices = [
      'remove-duplicate-clips', 'add-clips-to', 'merge',
      'move-2d-finished_images', 'normalize', 'add-angles', 'sample',
      'test-angle-conversion', 'bot-play', 'bot-play-random-clip',
      'test-zero-pose', 'test-normalization', 'test-plot-angles',
      'get-poses-from-angle-files', 'bot-play-clusters', 'create-tfrecords',
      'create-primitives', 'create-vocabulary', 'create-question',
      'create-sfa-dataset'
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
  elif command_name == 'bot-play-clusters':
    clusters = common.data_utils.get_clusters()

    all_angles = []
    for i in args.args[1:]:
      all_angles += clusters[int(i)]

    bot = common.bot.BotController(args.args[0])
    bot.play_angles(all_angles)
  elif command_name == 'create-tfrecords':
    import learning.data
    learning.data.create_tfrecords(*args.args)
  elif command_name == 'create-primitives':
    create_primitives()
  elif command_name == 'create-vocabulary':
    common.data_utils.create_vocabulary()
  elif command_name == 'create-question':
    create_question(args.args[0]);
  elif command_name == 'create-sfa-dataset':
    common.data_utils.create_sfa_dataset(*args.args)
  else:
    logger.error("Command {} not found.".format(command_name))
