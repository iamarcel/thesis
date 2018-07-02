import logging
import os
import os.path
import shutil
import numpy as np
import scipy
import scipy.linalg
import math
import re
import json
import jsonlines
import collections
from random import randint

from . import pose_utils, vector

logger = logging.getLogger(__name__)

DEFAULT_CLIPS_PATH = 'clips.jsonl'
DEFAULT_CONFIG_PATH = 'config.json'
DEFAULT_IMAGES_PATH = 'images/'
DEFAULT_TTS_PATH = os.path.join('tts-clips')
DEFAULT_VIDEO_PATH = os.path.join('video-clips')

# Joints in H3.6M -- data has 32 joints, but only 17 that move;
# these are the indices.
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

DIRECTION_NAMES = [
    'RThigh', 'RShin', 'LThigh', 'LShin', 'Breast', 'Backbone', 'Neck', 'Head',
    'LCollar', 'LUpperArm', 'LLowerArm', 'RCollar', 'RUpperArm', 'RLowerArm'
]

UPPER_BODY_PARTS = [
    'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LElbow',
    'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]


class ClipWriter():

  def __init__(self, path=DEFAULT_CLIPS_PATH, mode='a'):
    if mode == 'w':
      logger.warn("Overwriting any clips in {}".format(path))

    self.path = path
    self.writer = jsonlines.open(path, mode=mode)

  def send(self, clip):
    logger.debug('Writing clip {} to {}'.format(clip['id'], self.path))
    self.writer.write(clip)

  def close(self):
    self.writer.close()


def get_clips(path=DEFAULT_CLIPS_PATH):
  with jsonlines.open(path, mode='r') as clips:
    return list(clips)


def get_clip_ids(path=DEFAULT_CLIPS_PATH):
  with jsonlines.open(path, mode='r') as reader:
    return list(map(lambda c: c['id'], reader))


def remove_duplicate_clips(clips_path=DEFAULT_CLIPS_PATH,
                           output_path='clips-deduped.jsonl'):
  writer = ClipWriter(output_path)

  with jsonlines.open(clips_path, 'r') as reader:
    ids = []
    n_total_clips = 0

    for clip in reader:
      n_total_clips += 1
      clip_id = clip['id']
      if clip_id in ids:
        continue

      writer.send(clip)
      ids.append(clip_id)

  writer.close()
  logger.info("Wrote {} unique out of {} clips.".format(
      len(ids), n_total_clips))


def add_clips_to(clips_path_a, clips_path_b=DEFAULT_CLIPS_PATH):
  logger.info("Adding clips from {} to {}.".format(clips_path_a, clips_path_b))

  writer = ClipWriter(clips_path_b)
  with jsonlines.open(clips_path_a, 'r') as reader:
    for clip in reader:
      writer.send(clip)

  writer.close()


def get_clip_image_filenames(clip, images_path=DEFAULT_IMAGES_PATH):
  filenames = os.listdir(images_path)
  return filter(lambda f: f.startswith(clip['id'] + '-'), filenames)


def move_2d_finished_images(clips_path=DEFAULT_CLIPS_PATH,
                            images_path=DEFAULT_IMAGES_PATH,
                            images_path_done=None):
  if images_path_done is None:
    images_path_done = os.path.join(images_path, 'done')

  clips_path = os.path.realpath(clips_path)
  images_path = os.path.realpath(images_path)
  images_path_done = os.path.realpath(images_path_done)

  logger.debug("Reading clips from {}.".format(clips_path))
  logger.debug("Reading images from {}.".format(images_path))
  logger.debug("Moving images of finished clips to {}."
               .format(images_path_done))

  if not os.path.exists(images_path_done):
    os.makedirs(images_path_done)

  n_clips_done = 0
  n_clips_all = 0
  all_filenames = list(pose_utils.get_outputs())
  with jsonlines.open(clips_path, 'r') as reader:
    for clip in reader:
      n_clips_all += 1

      filenames = pose_utils.get_clip_files(clip, filenames=all_filenames)
      has_detections = any(True for _ in filenames)

      if not has_detections:
        continue

      n_clips_done += 1
      image_files = list(
          get_clip_image_filenames(clip, images_path=images_path))
      logging.debug("Moving {} image files for clip {}.".format(
          len(image_files), clip['id']))
      for filename in image_files:
        in_path = os.path.join(images_path, filename)
        out_path = os.path.join(images_path_done, os.path.basename(filename))
        logger.debug("Moving {} to {}".format(filename, out_path))
        shutil.move(in_path, out_path)

  logging.info("Moved {} out of {} clips' images.".format(
      n_clips_done, n_clips_all))


def get_clip_stats(clips_path=DEFAULT_CLIPS_PATH):
  import itertools

  clips = get_clips(clips_path)
  angles = map(lambda clip: clip['angles'], clips)
  all_angles = itertools.chain.from_iterable(angles)
  all_angles = map(pose_utils.get_angle_list, all_angles)
  all_angles = np.array(list(all_angles))

  mean = np.mean(all_angles, axis=0)
  std = np.std(all_angles, axis=0)

  print("Mean: {}".format(mean))
  print("STD:  {}".format(std))

  return mean, std


def add_clip_angles(read_path=DEFAULT_CLIPS_PATH,
                    write_path='clips-angles.jsonl'):

  assert read_path != write_path

  writer = ClipWriter(write_path, mode='w')
  clips = get_clips(read_path)

  n_clips_in = 0
  n_clips_out = 0

  for clip in clips:
    n_clips_in += 1
    try:
      points = np.asarray(clip['points_3d'])
      angles = list(map(pose_utils.get_pose_angles, points))
      clip['angles'] = angles
      writer.send(clip)
      n_clips_out += 1
    except ValueError as e:
      logger.warn(e)

  writer.close()

  logger.info("Wrote {} out of {} clips.".format(n_clips_out, n_clips_in))


def normalize_clips(read_path=DEFAULT_CLIPS_PATH,
                    write_path='clips-normalized.jsonl'):

  assert read_path != write_path

  writer = ClipWriter(write_path, mode='w')
  clips = get_clips(read_path)

  n_clips_in = 0
  n_clips_out = 0

  all_angles = None

  for clip in clips:
    n_clips_in += 1
    try:
      if 'points_3d' not in clip:
        raise ValueError('No 3D points in clip')

      points = np.asarray(clip['points_3d'])
      if len(points.shape) < 2:
        raise ValueError('Badly shaped 3D points, has shape {}'.format(
            points.shape))

      points = list(map(straighten_pose, points))
      points = patch_poses(points)
      points = straighten_frames(points)
      clip['points_3d'] = points.tolist()
      clip['angles'] = list(map(pose_utils.get_pose_angles, points.tolist()))

      angle_list = np.array(
          list(map(pose_utils.get_angle_list, clip['angles'])))
      if all_angles is None:
        all_angles = angle_list
      else:
        all_angles = np.vstack((all_angles, angle_list))

      writer.send(clip)
      n_clips_out += 1
    except ValueError as e:
      logger.warn(e)

  writer.close()

  logger.info("Wrote {} out of {} clips.".format(n_clips_out, n_clips_in))

  config_path = DEFAULT_CONFIG_PATH
  if os.path.isfile(config_path):
    with open(config_path) as config_file:
      config = json.load(config_file)
  else:
    config = {}

  config['angle_stats'] = {
      'mean': np.mean(all_angles, axis=0).tolist(),
      'std': np.std(all_angles, axis=0).tolist()
  }

  with open(config_path, 'w') as config_file:
    json.dump(config, config_file)

  logger.info("Saved statistics to {}".format(config_path))


def oneliner_rotation_matrix(axis, theta):
  return scipy.linalg.expm(
      np.cross(np.eye(3), axis / scipy.linalg.norm(axis) * theta))


def straighten_pose(points_3d):
  points_3d = np.array(points_3d)

  # Figure out the person's orientation from hip position
  # Note that the hip center is [0, 0, 0]
  rhip = points_3d[1, :]
  rhip = rhip / np.linalg.norm(rhip)
  alpha = np.arcsin(rhip[2])
  if rhip[0] > 0.0:
    alpha = np.pi - alpha

  # Straighten the person's orientation so he looks towards +z
  # Rotate alpha degrees around the y axis, some transformation matrix thing
  M = rotation_matrix([0, 1, 0], alpha)
  assert points_3d.shape[1] == 3
  normal_points = np.matmul(points_3d, M)

  normal_points = normalize_pose_scale(normal_points)

  if np.any(np.abs(normal_points) > 1.0):
    raise ValueError("At least one point is larger than 1.0")

  if not abs(normal_points[1, 2]) < 1.0e-2:
    logging.warn("Right hip z is large: {}".format(normal_points[1, 2]))

  return normal_points


def straighten_frames(frames, epsilon=np.pi / 12):
  """Straightens a pose by correcting for a forward lean.
    """
  frames = np.asarray(frames)

  spine = frames[:, H36M_NAMES.index(
      'Thorax'), :] - frames[:, H36M_NAMES.index('Hip'), :]
  alphas = vector.multi_angle_between(spine, [0, -1, 0], [1, 0, 0])
  alphas_clamped = np.clip(alphas, -epsilon, epsilon)

  patch_alphas = alphas_clamped - alphas
  patch_alpha = np.mean(patch_alphas)
  M = rotation_matrix([1, 0, 0], -patch_alpha)

  upper_body_indices = [H36M_NAMES.index(i) for i in UPPER_BODY_PARTS]
  frames[:, upper_body_indices] = np.matmul(frames[:, upper_body_indices], M)

  return frames


def rotation_matrix(axis, theta):
  """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
  axis = np.asarray(axis)
  axis = axis / math.sqrt(np.dot(axis, axis))
  a = math.cos(theta / 2.0)
  b, c, d = -axis * math.sin(theta / 2.0)
  aa, bb, cc, dd = a * a, b * b, c * c, d * d
  bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
  return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                   [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                   [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def normalize_pose_scale(pose, to_height=1.0):
  """Scales a pose so its height is 1
    """
  head_index = pose_utils.H36M_NAMES.index('Head')
  foot_index = pose_utils.H36M_NAMES.index('LFoot')
  head_y = pose[head_index, 1]
  foot_y = pose[foot_index, 1]

  height = abs(head_y - foot_y)
  scale = to_height / height
  pose = pose * scale

  return pose


def patch_poses(poses, max_out_of_bounds_joints=4.0 / 32, max_distance=0.3):
  """Fills in points that moved too much with the previous frame's point.

    Args:
        poses: ndarray (n_frames, n_joints, n_dims)
        max_out_of_bounds_joints: float Fraction of out of bounds allowed
        max_distance: maximum travel distance between frames
            (a person's height is 1)
    Returns:
        poses
    Raises:
        ValueError: if too many joints are out of bounds (specified by
            max_out_of_bounds_joints)
    """
  max_distance_sq = max_distance**2

  poses = np.asarray(poses)

  for i, pose in enumerate(poses):
    # Skip first frame
    if i == 0:
      continue

    # Calculate distance to previous frame
    previous_pose = poses[i - 1]
    distance_sq = np.sum((pose - previous_pose)**2, axis=1)
    assert distance_sq.shape == (pose.shape[0],), \
        "Should sum for every joint, instead got shape {}" \
        .format(distance_sq.shape)
    out_of_bounds_joints = np.where(distance_sq > max_distance_sq)

    if len(out_of_bounds_joints[0]) > \
       max_out_of_bounds_joints * pose.shape[0]:
      raise ValueError("Too many ({} out of {}) joints out of bounds.".format(
          len(out_of_bounds_joints[0]), pose.shape[0]))

    # Fill in with previous frame's points
    pose[out_of_bounds_joints, :] = previous_pose[out_of_bounds_joints, :]

  return poses


def clean_word(word):
  return re.sub('[ 0123456789\.\,;:\?!\(\)\[\]\{\}"\'\<\>%]', '',
                word.strip().lower())


def create_vocabulary(vocab_size=512, clips_path=DEFAULT_CLIPS_PATH, vocab_path='vocab.txt'):
  vocab = collections.Counter()
  for clip in get_clips(clips_path):
    words = clip['subtitle'].split(' ')
    words = list(filter(lambda x: len(x.strip()) > 0, map(clean_word, words)))
    vocab.update(words)

  vocab_file = open(vocab_path, 'w')
  for i, (word, cnt) in enumerate(vocab.most_common(vocab_size)):
    vocab_file.write('{}\n'.format(str(word), i + 1))
  vocab_file.close()


def get_random_clip():
  clips = get_clips()
  clip = None
  while clip is None or 'points_3d' not in clip or len(clip['points_3d']) == 0:
    logger.info('Getting a random clip')
    clip = clips[randint(0, len(clips))]

  return clip


def get_clusters():
  with open("cluster-centers.json") as centers_file:
    centers = json.load(centers_file)['clusters']

  return centers


def create_sfa_dataset(file_name='clips_sfa.txt'):
  clips = get_clips();
  with open(file_name, 'w') as f:
    for i, clip in enumerate(clips):
      for j, frame in enumerate(clip['angles']):
        f.write(str(i) + " ")  # Example ID
        f.write(str(j) + " ")  # Time index
        f.write("1 ")          # Label

        frame_arr = pose_utils.get_angle_list(frame)
        f.write(" ".join(str(x) for x in frame_arr))
        f.write("\n")
