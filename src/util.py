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
import itertools

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
  video_order = [0, 1, 2, 3]
  # random.shuffle(video_order)

  clips = [common.data_utils.get_random_clip() for _ in range(3)]
  subtitle = ''
  subtitles = []
  with jsonlines.open('questions.jsonl', mode='a') as writer:
    question = dict(
        ids=[],
        subtitle='',
        angles_expected=[],
        classes_expected=[],
        n_frames_expected=[],
        video_order=video_order)

    for clip in clips:
      question['ids'] += [clip['id']]
      subtitle += clip['subtitle'] + ' '
      subtitles += [str(clip['subtitle'])]
      question['angles_expected'] += clip['angles']
      question['classes_expected'] += [clip['class']]
      question['n_frames_expected'] += [len(clip['angles'])]

    question['subtitle'] = subtitle
    writer.write(question)

  print(subtitles)
  question_id = '.'.join(question['ids'])

  # Generate TTS audio clip
  file_names_speech = []
  for i, sub in enumerate(subtitles):
    file_name_speech = os.path.join(common.data_utils.DEFAULT_TTS_PATH, question['ids'][i] + '.wav')
    common.watson.write_tts_clip(file_name_speech, subtitle)
    file_names_speech += [file_name_speech]

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
        str(file_name)
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
  file_name_expected = record_pose_animation(question['angles_expected'], 'expected')

  bot.reset_pose()
  clusters = common.data_utils.get_clusters()
  predictions = learning.model.predict_classes(subtitles)
  animation = list(itertools.chain.from_iterable(clusters[x] for x in predictions))
  file_name_cluster = record_pose_animation(animation, 'cluster')

  bot.reset_pose()
  animation = list(itertools.chain.from_iterable(learning.model.predict_sequences(subtitles)))
  file_name_predicted = record_pose_animation(animation, 'predicted')

  bot.reset_pose()
  time_expected = int(math.ceil(len(question['angles_expected']) / 25)) + 1
  proc, file_name_nao = record_screen('nao', time_expected)
  bot.say(subtitle)
  output = proc.wait()

  def text_filter(text, start, end):
    text = text.replace('\'', '\\\'')
    return (';[v]drawtext=text=\'' + text + '\'' +
            ":enable='between(t,"+str(int(math.ceil(start)))+","+str(int(math.floor(end)))+")'"
            ":fontfile=DejaVuSans\\\:style=Bold" +
            ":x=(main_w/2-text_w/2)" +
            ":y=(main_h-(text_h*2)):fontsize=48:fontcolor=white" +
            ":borderw=2[v]")

  # Merge it all
  text_filters = ''
  start = 0
  for i, sub in enumerate(subtitles):
    end = start + float(question['n_frames_expected'][i]) / 25
    print(start)
    print(end)
    text_filters += text_filter(sub, start, end)
    start = end
    print(start)
    print(end)

  print(text_filters)

  subtitles = ''.join(text_filter(sub, question['n_frames_expected'][i], end) for i, sub in enumerate(subtitles))
  print(subtitles)
  video_file_names = [file_name_expected, file_name_cluster, file_name_predicted, file_name_nao]
  proc = Popen([
      'ffmpeg',
      '-i', video_file_names[video_order[0]],
      '-i', video_file_names[video_order[1]],
      '-i', video_file_names[video_order[2]],
      '-i', video_file_names[video_order[3]],
      '-i', file_names_speech[0], '-itsoffset', str(math.ceil(question['n_frames_expected'][0] / 25)),
      '-i', file_names_speech[1], '-itsoffset', str(math.ceil((question['n_frames_expected'][0] + question['n_frames_expected'][1]) / 25)),
      '-i', file_names_speech[2],
      '-filter_complex', ('[0:v][1:v][2:v][3:v]hstack=inputs=4[v]' +
                          text_filters +
                          ';[4:a][5:a][6:a]amix=3[a]'),
      '-map', '[v]',
      '-map', '[a]',
      # '-map', '4:a:0',
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
      'write-poses-from-angle-files', 'bot-play-clusters', 'create-tfrecords',
      'create-primitives', 'create-vocabulary', 'create-question',
      'create-sfa-dataset', 'create-sanity-check-2d', 'create-sanity-check-2d-3d',
      'create-sanity-check-pipeline', 'create-pose-vector-plot',
      'create-sanity-check-cluster-centers', 'create-sanity-check-cluster-samples',
      'count-tfrecords'
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
  elif command_name == 'create-sanity-check-2d':
    clips = common.data_utils.get_clips()
    for clip in clips:
      try:
        common.visualize.create_sanity_check_2d(clip)
        break
      except (ValueError, IOError) as e:
        print(e)
  elif command_name == 'create-sanity-check-2d-3d':
    clips = common.data_utils.get_clips()
    for clip in clips[5:]:
      try:
        common.visualize.create_sanity_check_2d_3d(clip)
        break
      except (ValueError, IOError) as e:
        print(e)
  elif command_name == 'create-sanity-check-pipeline':
    clips = common.data_utils.get_clips()
    for clip in clips[5:]:
      try:
        common.visualize.create_sanity_check_pipeline(clip, openpose_output_dir='./output/')
        break
      except (ValueError, IOError) as e:
        print(e)
  elif command_name == 'create-pose-vector-plot':
    clips = common.data_utils.get_clips()
    for clip in clips[7:]:
      try:
        common.visualize.create_pose_vector_plot(clip)
        break
      except (ValueError, IOError) as e:
        print(e)
  elif command_name == 'create-sanity-check-cluster-centers':
    with open('./cluster-centers.json') as centers_file:
      centers = json.load(centers_file)['clusters']
    common.visualize.create_sanity_check_gesture_grid_animation(centers, 'cluster-centers')
  elif command_name == 'create-sanity-check-cluster-samples':
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
  elif command_name == 'count-tfrecords':
    import learning.data
    print(learning.data.count_tfrecords(args.args[0]))
  else:
    logger.error("Command {} not found.".format(command_name))
