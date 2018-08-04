import os
import json
import logging
import numpy as np
import math

from . import vector

logger = logging.getLogger(__name__)

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
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

COCO_BODY_PARTS = [
    'Neck/Nose', 'Thorax', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder',
    'LElbow', 'LWrist', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
    'REye', 'LEye', 'REar', 'LEar'
]

# To convert the dictionary to a list with consistent ordering
ANGLE_NAMES_ORDER = [
    'HipRoll', 'HipPitch', 'RShoulderPitch', 'RShoulderRoll', 'LShoulderPitch',
    'LShoulderRoll', 'RElbowRoll', 'LElbowRoll', 'HeadPitch', 'HeadYaw'
]

JOINTS = np.array([
  ['Hip', 'RHip'], ['RHip', 'RKnee'], ['RKnee', 'RFoot'],
  ['Hip', 'LHip'], ['LHip', 'LKnee'], ['LKnee', 'LFoot'],
  ['Hip', 'Spine'], ['Spine', 'Thorax'], ['Thorax', 'Neck/Nose'],
  ['Neck/Nose', 'Head'], ['Thorax', 'LShoulder'], ['LShoulder', 'LElbow'],
  ['LElbow', 'LWrist'], ['Thorax', 'RShoulder'], ['RShoulder', 'RElbow'],
  ['RElbow', 'RWrist']])

COCO_JOINTS = np.array([
  ['RHip', 'RKnee'], ['RKnee', 'RFoot'],
  ['LHip', 'LKnee'], ['LKnee', 'LFoot'], ['RHip', 'Thorax'],
  ['LHip', 'Thorax'], ['Thorax', 'Neck/Nose'],
  ['Thorax', 'LShoulder'], ['LShoulder', 'LElbow'],
  ['LElbow', 'LWrist'], ['Thorax', 'RShoulder'], ['RShoulder', 'RElbow'],
  ['RElbow', 'RWrist'], ['Neck/Nose', 'LEye'], ['Neck/Nose', 'REye'],
  ['LEye', 'LEar'], ['REye', 'REar']])

LEFT_FRONT_BODY_PARTS = [
  'Hip',
  'LHip',
  'LKnee',
  'LFoot',
  'Spine',
  'Thorax',
  'Neck/Nose',
  'Head',
  'LShoulder',
  'LElbow',
  'LWrist'
]

KEY_OP_PEOPLE = 'people'
KEY_OP_KEYPOINTS = 'pose_keypoints_2d'

DEFAULT_OPENPOSE_OUTPUT_PATH = '../output/'

X_AXIS = np.array([1., 0., 0.])
Y_AXIS = np.array([0., 1., 0.])
Z_AXIS = np.array([0., 0., 1.])

# In NAO's frame of reference, in meters
# [[file:~/org/thesis.org::*NAO Skeleton]]
NAO_ZERO_POSE = {
    'Hip': [0., 0., -0.085],
    'RHip': [0., -0.05, -0.085],
    'RKnee': [0., -0.05, -0.185],
    'RFoot': [0., -0.05, -0.333],
    'LHip': [0., 0.05, -0.085],
    'LKnee': [0., 0.05, -0.185],
    'LFoot': [0., 0.05, -0.333],
    'Spine': [0., 0., 0.05],
    'Thorax': [0., 0., 0.1265],
    'Neck/Nose': [.05, 0., 0.1765],
    'Head': [0., 0., 0.2265],
    'LShoulder': [0., 0.098, 0.1265],
    'LElbow': [0.105, 0.113, 0.1265],
    'LWrist': [0.2187, 0.113, 0.1265],
    'RShoulder': [0., -0.098, 0.1265],
    'RElbow': [0.105, -0.113, 0.1265],
    'RWrist': [0.2187, -0.113, 0.1265],
}

# In NAO's reference frame
ROLL_AXIS = [1., 0., 0.]
PITCH_AXIS = [0., 1., 0.]
YAW_AXIS = [0., 0., 1.]


def load_clip_keypoints(clip,
                        openpose_output_dir=DEFAULT_OPENPOSE_OUTPUT_PATH,
                        min_confidence=0.6,
                        filenames=None):
  if filenames is None:
    filenames = get_outputs(openpose_output_dir)
  keypoints = []
  clip_files = list(get_clip_files(
    clip,
    filenames=filenames,
    openpose_output_dir=openpose_output_dir))
  if len(clip_files) == 0:
    raise ValueError("No keypoint data found")

  for full_name in clip_files:
    with open(full_name) as keypoint_file:
      frame_keypoint_data = json.load(keypoint_file)
      people = frame_keypoint_data[KEY_OP_PEOPLE]
      if len(people) == 0:
        raise ValueError("Clip has frame without people detected")

      person = closest_to_center_person(people, clip['center'])
      if len(person[KEY_OP_KEYPOINTS]) == 0:
        raise ValueError("No keypoints available")

      mean_confidence = np.mean(get_confidences(person[KEY_OP_KEYPOINTS]))
      if np.any(mean_confidence < min_confidence):
        raise ValueError("Clip has pose with confidence score" +
                         " lower than {}".format(min_confidence))

      keypoints.append([person[KEY_OP_KEYPOINTS]])

  return keypoints


def get_clip_files(clip,
                   filenames=None,
                   openpose_output_dir=DEFAULT_OPENPOSE_OUTPUT_PATH):

  if filenames is None:
    filenames = get_outputs(openpose_output_dir)

  return map(
      lambda f: os.path.join(openpose_output_dir, f),
      sorted(filter(lambda f: f.startswith(clip['id'] + '-'), filenames)))


def get_outputs(path=DEFAULT_OPENPOSE_OUTPUT_PATH):
  return filter(lambda f: f.endswith('_keypoints.json'), os.listdir(path))


def closest_to_center_person(people, center):
  """Returns the person whose hip is closest to the given center

    input:
        people: OpenPose detection (dict)
        center: (x, y) tuple of center position
    output:
        person: OpenPose detection of the person that is closest to the center
    """

  best_person = people[0]
  best_person_distance = 100000
  for i, person in enumerate(people):
    points = person[KEY_OP_KEYPOINTS]
    hip = (float(points[COCO_BODY_PARTS.index('LHip') * 3] +
                 points[COCO_BODY_PARTS.index('RHip') * 3]) / 2,
           float(points[COCO_BODY_PARTS.index('LHip') * 3] +
                 points[COCO_BODY_PARTS.index('RHip') * 3]) / 2)

    distance = (hip[0] - center[0])**2 + (hip[1] - center[1])**2
    if distance < best_person_distance:
      best_person = person
      best_person_distance = distance

  return best_person


def get_positions(keypoints):
  xy = []
  # ignore confidence score
  for o in range(0, len(keypoints), 3):
    xy.append(keypoints[o])
    xy.append(keypoints[o + 1])

  return xy


def get_confidences(keypoints):
  confidences = []
  for o in range(0, len(keypoints), 3):
    confidences.append(keypoints[o + 2])

  return confidences


def get_all_positions(keypoints_arr):
  n_points = keypoints_arr.shape[1]
  indices = np.sort(
      np.hstack((np.arange(0, n_points, 3), np.arange(1, n_points, 3))))

  return np.array(keypoints_arr)[:, indices]


def get_all_confidences(keypoints_arr):
  return np.array(keypoints_arr)[:, 2::3]


def openpose_to_baseline(coco_frames):
  """Converts a list of OpenPose frames to Baseline-compatible format

    Args:
      coco_frames: ndarray (?x, 18*3) - for every body part xi, yi, ci
    Returns:
      h36m_frames: ndarray (?x, 32*3) - for every H36M body part xi, yi, ci
    """
  coco_frames = np.squeeze(np.asarray(coco_frames))
  if coco_frames.shape[1] != len(COCO_BODY_PARTS) * 3:
    raise ValueError(
        "Expected predictions to be in OpenPose format, i.e. of shape (?, " +
        str(len(COCO_BODY_PARTS) * 3) + "), but got " + str(coco_frames.shape))

  # Store in flattened 2D coordinate array
  h36m_frames = np.zeros((coco_frames.shape[0], len(H36M_NAMES) * 3))

  # Corresponsing destination indices to map OpenPose data into H36M data
  h36m_indices = [
      np.where(np.array(H36M_NAMES) == name)[0] for name in COCO_BODY_PARTS
  ]
  coco_indices = np.where([len(i) != 0 for i in h36m_indices])[0]
  h36m_indices = np.array(
      [x[0] for x in list(filter(lambda x: len(x) != 0, h36m_indices))])

  # OpenPose format: xi, yi, ci (confidence)
  h36m_frames[:, h36m_indices * 3] = coco_frames[:, coco_indices * 3]
  h36m_frames[:, h36m_indices * 3 + 1] = coco_frames[:, coco_indices * 3 + 1]
  h36m_frames[:, h36m_indices * 3 + 2] = coco_frames[:, coco_indices * 3 + 2]

  def add_computed_point(dest_name, src1_name, src2_name, fn):
    dest_index = np.where(np.array(H36M_NAMES) == dest_name)[0][0]
    src1_index = np.where(np.array(H36M_NAMES) == src1_name)[0][0]
    src2_index = np.where(np.array(H36M_NAMES) == src2_name)[0][0]

    for j in range(3):
      di = dest_index * 3 + j
      s1i = src1_index * 3 + j
      s2i = src2_index * 3 + j
      h36m_frames[:, di] = fn(h36m_frames[:, s1i], h36m_frames[:, s2i])

  # Hip is center of Left and Right Hip
  add_computed_point('Hip', 'LHip', 'RHip', lambda i, j: (i + j) / 2)

  # Take Head as half the distance between thorax & nose above the nose
  add_computed_point('Head', 'Neck/Nose', 'Thorax',
                     lambda i, j: i + (i - j) / 2)

  # Spine is nead the neck base, between neck and hip
  add_computed_point('Spine', 'Thorax', 'Hip', lambda i, j: i + (j - i) / 2)

  return h36m_frames


def get_named_pose(in_pose, fmt='h36m'):
  in_pose = np.asarray(in_pose)

  names = []
  if fmt == 'h36m':
    names = H36M_NAMES
  elif fmt == 'coco':
    names = COCO_BODY_PARTS
  elif fmt == 'angle':
    return {ANGLE_NAMES_ORDER[i]: v for i, v in enumerate(in_pose)}
  elif fmt == 'sh':
    raise NotImplementedError("Doesn't support SH yet")
  else:
    raise ValueError("Unrecognized pose format {}".format(fmt))

  out_pose = {}
  in_pose = np.reshape(in_pose, (len(names), -1))
  for i, name in enumerate(names):
    if name is None or name == '':
      continue
    out_pose[name] = in_pose[i, :]

  return out_pose


def get_pose_angles(pose, fmt='h36m'):
  """Returns angles as understood by NAOqi from a H36M-formatted pose.
    """
  pose = get_named_pose(pose, fmt=fmt)
  pose = {k: np.asarray(v) for k, v in pose.iteritems()}
  angles = {}

  def norm_joint(name_a, name_b):
    vec = pose[name_b] - pose[name_a]
    vec /= np.linalg.norm(vec)
    return vec

  chest = norm_joint('Hip', 'Thorax')

  # Only Pepper has a hip
  angles['HipRoll'] = vector.angle_between(chest, -Y_AXIS, -Z_AXIS)
  angles['HipPitch'] = vector.angle_between(chest, -Y_AXIS, X_AXIS)

  r_upper_arm = norm_joint('RShoulder', 'RElbow')
  angles['RShoulderPitch'] = vector.angle_between(r_upper_arm, chest,
                                                  X_AXIS) - np.pi / 2

  angles['RShoulderRoll'] = vector.angle_with_plane(r_upper_arm,
                                                    X_AXIS) + np.pi / 10

  l_upper_arm = norm_joint('LShoulder', 'LElbow')
  angles['LShoulderPitch'] = vector.angle_between(chest, l_upper_arm,
                                                  X_AXIS) - np.pi / 2

  angles['LShoulderRoll'] = vector.angle_with_plane(l_upper_arm,
                                                    X_AXIS) - np.pi / 10

  r_elbow = norm_joint('RElbow', 'RWrist')
  angles['RElbowRoll'] = vector.shortest_angle_between(r_upper_arm, r_elbow)

  l_elbow = norm_joint('LElbow', 'LWrist')
  angles['LElbowRoll'] = -(vector.shortest_angle_between(l_upper_arm, l_elbow))

  nose = norm_joint('Thorax', 'Neck/Nose')
  head = norm_joint('Thorax', 'Head')
  angles['HeadPitch'] = vector.angle_between(nose, head, X_AXIS) - np.pi / 4
  angles['HeadYaw'] = vector.angle_between(-Z_AXIS, nose, -Y_AXIS)

  return angles


def get_angle_list(pose_dict, fmt='angles'):
  if fmt != 'angles':
    angles = get_pose_angles(pose_dict, fmt)
  else:
    angles = pose_dict

  return [angles[k] for k in ANGLE_NAMES_ORDER]


def get_point_list(joint_dict, fmt='h36m'):
  format_template = None
  if fmt == 'h36m':
    format_template = H36M_NAMES
  elif fmt == 'coco':
    format_template = COCO_BODY_PARTS
  else:
    raise NotImplementedError('Cannot convert to format {}'.format(fmt))

  joint_shape = list(np.asarray(joint_dict.itervalues().next()).shape)

  pose = np.zeros([len(format_template)] + joint_shape)
  for i, joint_name in enumerate(format_template):
    if joint_name == '':
      continue
    if joint_name not in joint_dict:
      logger.warn('Joint {} is not in this dict'.format(joint_name))
    else:
      pose[i, :] = np.asarray(joint_dict[joint_name])

  if fmt == 'h36m' and joint_shape[0] == 3:
    # Reorder axes so they correspond with the H36M frame
    pose[:, 0], pose[:, 1], pose[:, 2] = pose[:, 1].copy(), -pose[:, 2].copy(), -pose[:, 0].copy()

  return pose


def get_pose_from_angles(angles, axis_frame='nao'):
  """Returns a dict of 3d joint positions, with NAO's skeleton and in `axis_frame` frame
    of reference.

    Params:
      angles: dict of angles
      axis_frame: either 'nao' (default) or 'h36m'
    Returns:
      pose: dict of np.array([x, y, z]) coordinates
    """

  pose = NAO_ZERO_POSE.copy()
  pose = {k: np.asarray(v) for k, v in pose.iteritems()}

  def rotate_joints(joint_names, origin, angles):
    for joint_name in joint_names:
      position = pose[joint_name]

      rel_position = position - origin
      for axis, angle in angles:
        rel_position = np.matmul(rel_position, rotation_matrix(axis, angle))

      pose[joint_name] = rel_position + origin

  # Hip
  if 'HipRoll' in angles and 'HipPitch' in angles:
    rotate_joints(
        joint_names=[
            'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow',
            'LWrist', 'RShoulder', 'RElbow', 'RWrist'
        ],
        origin=pose['Hip'],
        angles=[(ROLL_AXIS, -angles['HipRoll']), (PITCH_AXIS,
                                                  -angles['HipPitch'])])

  # Right Arm
  if 'RShoulderPitch' in angles:
    rotate_joints(
        joint_names=['RElbow', 'RWrist'],
        origin=pose['RShoulder'],
        angles=[(PITCH_AXIS, -angles['RShoulderPitch'])])

  right_shoulder_roll_axis = np.cross(pose['RShoulder'] - pose['Thorax'],
                                      pose['RElbow'] - pose['RShoulder'])
  if 'RShoulderRoll' in angles:
    rotate_joints(
        joint_names=['RElbow', 'RWrist'],
        origin=pose['RShoulder'],
        angles=[(right_shoulder_roll_axis, -angles['RShoulderRoll'])])

  # Right Wrist
  if 'RElbowRoll' in angles:
    rotate_joints(
        joint_names=['RWrist'],
        origin=pose['RElbow'],
        angles=[(right_shoulder_roll_axis, -angles['RElbowRoll'])])

  # Left Arm
  if 'LShoulderPitch' in angles:
    rotate_joints(
        joint_names=['LElbow', 'LWrist'],
        origin=pose['LShoulder'],
        angles=[(PITCH_AXIS, -angles['LShoulderPitch'])])

  left_shoulder_roll_axis = np.cross(pose['LElbow'] - pose['LShoulder'],
                                     pose['LShoulder'] - pose['Thorax'])
  if 'LShoulderRoll' in angles:
    rotate_joints(
        joint_names=['LElbow', 'LWrist'],
        origin=pose['LShoulder'],
        angles=[(left_shoulder_roll_axis, -angles['LShoulderRoll'])])

  # Left Wrist
  if 'LElbowRoll' in angles:
    rotate_joints(
        joint_names=['LWrist'],
        origin=pose['LElbow'],
        angles=[(left_shoulder_roll_axis, -angles['LElbowRoll'])])

  # Head
  if 'HeadPitch' in angles and 'HeadYaw' in angles:
    rotate_joints(
        joint_names=['Neck/Nose', 'Head'],
        origin=pose['Thorax'],
        angles=[(PITCH_AXIS, -angles['HeadPitch']), (YAW_AXIS,
                                                     -angles['HeadYaw'])])

  if axis_frame != 'nao':
    pose = {k: lambda v: np.stack([pose[1, :], -pose[2, :], -pose[0, :]]) for
            k, v in pose.iteritems()}

  return pose


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


class Pose(object):
  """A pose is a single frame representing how a person is standing.

  The internal representation for this object is a dictionary that maps either
  the angle names or the joint names to its angle or position, respectively.
  There are no manipulations done on the actual data (no conversion between
  angle and points) if it is not necessary.

  """

  def __init__(self, data, fmt='angle'):
    if isinstance(data, dict):
      self.data = data
    else:
      self.data = get_named_pose(data, fmt)

    self.fmt = fmt

  def as_list(self, fmt='h36m'):
    if fmt == 'angles':
      return get_angle_list(self.data, fmt=fmt)
    else:
      return get_point_list(self.data, fmt=fmt)

  def as_dict(self, fmt='h36m'):
    if fmt == self.fmt:
      return self.data
    elif self.fmt == 'angle':
      # Returns in the axis frame of reference for H36M points
      return get_pose_from_angles(self.data, axis_frame='h36m')
    elif fmt == 'angle':
      return get_pose_angles(self.data, fmt)
    else:
      raise ValueError('Cannot determine which pose conversion to do')


class Gesture(object):
  """A Gesture represents a movement, i.e., a sequence of frames, each a Pose.

  """

  def __init__(self, frames, fmt='angle'):
    self.frames = list(map(lambda x: Pose(x, fmt), frames))
    self.fmt = fmt

  def as_list(self, fmt='h36m'):
    return list(map(lambda x: x.as_list(fmt), self.frames))

  def as_dict(self, fmt='h36m'):
    return list(map(lambda x: x.as_dict(fmt), self.frames))
