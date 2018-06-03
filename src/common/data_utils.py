import argparse
import logging
import os
import os.path
import shutil
import numpy as np
import scipy
import scipy.linalg
import math
import json

import jsonlines

from . import config_utils, pose_utils, vector

logger = logging.getLogger(__name__)

DEFAULT_CLIPS_PATH = 'clips.jsonl'
DEFAULT_CONFIG_PATH = 'config.json'
DEFAULT_IMAGES_PATH = 'images/'

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

H36M_PARENT_NAMES = [''] * 32
H36M_PARENT_NAMES[2] = 'RHip'
H36M_PARENT_NAMES[3] = 'RKnee'
H36M_PARENT_NAMES[7] = 'LHip'
H36M_PARENT_NAMES[8] = 'LKnee'
H36M_PARENT_NAMES[12] = 'Thorax'
H36M_PARENT_NAMES[13] = 'Hip'
H36M_PARENT_NAMES[14] = 'Spine'
H36M_PARENT_NAMES[15] = 'Neck/Nose'
H36M_PARENT_NAMES[17] = 'Spine'
H36M_PARENT_NAMES[18] = 'LShoulder'
H36M_PARENT_NAMES[19] = 'LElbow'
H36M_PARENT_NAMES[25] = 'Spine'
H36M_PARENT_NAMES[26] = 'RShoulder'
H36M_PARENT_NAMES[27] = 'RElbow'

DIRECTION_NAMES = [
    'RThigh',
    'RShin',
    'LThigh',
    'LShin',
    'Breast',
    'Backbone',
    'Neck',
    'Head',
    'LCollar',
    'LUpperArm',
    'LLowerArm',
    'RCollar',
    'RUpperArm',
    'RLowerArm'
]

UPPER_BODY_PARTS = [
    'Spine',
    'Thorax',
    'Neck/Nose',
    'Head',
    'LShoulder',
    'LElbow',
    'LElbow',
    'LWrist',
    'RShoulder',
    'RElbow',
    'RWrist'
]

class ClipWriter():

    def __init__(self, path=DEFAULT_CLIPS_PATH, mode='a'):
        if mode == 'w':
            logger.warn("Overwriting existing clips file")

        self.path = path
        self.writer = jsonlines.open(path, mode=mode)

    def send(self, clip):
        logger.debug('Writing clip {} to {}'.format(clip['id'], self.path))
        self.writer.write(clip)

    def close(self):
        self.writer.close()


def get_clip_ids(path=DEFAULT_CLIPS_PATH):
    with jsonlines.open(path, mode='r') as reader:
        return list(map(lambda c: c['id'], reader))


def get_clips(path=DEFAULT_CLIPS_PATH):
    with jsonlines.open(path, mode='r') as clips:
        return list(clips)


def config_to_clips(
        config_path=DEFAULT_CONFIG_PATH,
        clips_path=DEFAULT_CLIPS_PATH):
    logging.info("Adding clips from {} to {}."
                 .format(config_path, clips_path))

    writer = ClipWriter(clips_path)
    clips = config_utils.load_config(config_path)['clips']
    logging.debug("Found {} clips.".format(len(clips)))

    for clip in clips:
        writer.send(clip)

    writer.close()


def remove_duplicate_clips(
        clips_path=DEFAULT_CLIPS_PATH,
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
    logger.info("Wrote {} unique out of {} clips."
                .format(len(ids), n_total_clips))


def add_clips_to(clips_path_a, clips_path_b=DEFAULT_CLIPS_PATH):
    logger.info("Adding clips from {} to {}."
                .format(clips_path_a, clips_path_b))

    writer = ClipWriter(clips_path_b)
    with jsonlines.open(clips_path_a, 'r') as reader:
        for clip in reader:
            writer.send(clip)

    writer.close()


def get_image_files(
        clip,
        images_path=DEFAULT_IMAGES_PATH):
    filenames = os.listdir(images_path)
    return filter(
        lambda f: f.startswith(clip['id'] + '-'),
        filenames)


def move_2d_finished_images(
        clips_path=DEFAULT_CLIPS_PATH,
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

            filenames = pose_utils.get_clip_files(
                clip,
                filenames=all_filenames)
            has_detections = any(True for _ in filenames)

            if not has_detections:
                continue

            n_clips_done += 1
            image_files = list(get_image_files(clip, images_path=images_path))
            logging.debug("Moving {} image files for clip {}."
                          .format(len(image_files), clip['id']))
            for filename in image_files:
                in_path = os.path.join(
                    images_path,
                    filename)
                out_path = os.path.join(
                    images_path_done,
                    os.path.basename(filename))
                logger.debug("Moving {} to {}"
                             .format(filename, out_path))
                shutil.move(in_path, out_path)

    logging.info("Moved {} out of {} clips' images."
                 .format(n_clips_done, n_clips_all))


def clip_stats(clips_path=DEFAULT_CLIPS_PATH):
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
                raise ValueError('Badly shaped 3D points, has shape {}'.format(points.shape))

            points = list(map(straighten_pose, points))
            points = patch_poses(points)
            points = straighten_frames(points)
            clip['points_3d'] = points.tolist()
            clip['angles'] = list(map(pose_utils.get_pose_angles, points.tolist()))

            angle_list = np.array(list(map(pose_utils.get_angle_list, clip['angles'])))
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
    return scipy.linalg.expm(np.cross(np.eye(3), axis / scipy.linalg.norm(axis) * theta))


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


def straighten_frames(frames, epsilon=np.pi/12):
    """Straightens a pose by correcting for a forward lean.
    """
    frames = np.asarray(frames)

    spine = frames[:, H36M_NAMES.index('Thorax'), :] - frames[:, H36M_NAMES.index('Hip'), :]
    alphas = vector.multi_angle_between(spine, [0, -1, 0], [1, 0, 0])
    alphas_clamped = np.clip(alphas, -epsilon, epsilon)

    patch_alphas = alphas_clamped - alphas
    patch_alpha = np.mean(patch_alphas)
    M = rotation_matrix([1, 0, 0], -patch_alpha)

    upper_body_indices = [H36M_NAMES.index(i) for i in UPPER_BODY_PARTS]
    frames[:, upper_body_indices] = np.matmul(frames[:, upper_body_indices], M)

    return frames


def pose_to_directions(pose):
    # For reconstruction: bone_lengths = calc_bone_lengths(REFERENCE_POSE)
    directions = []

    for i, joint in enumerate(pose):
        joint_name = H36M_NAMES[i]
        if joint_name in ['', 'Hip', 'LHip', 'RHip']:
            continue

        parent = get_joint_parent(pose, i)
        direction = joint - parent
        direction /= np.linalg.norm(direction)
        directions.append(direction)

    return np.asarray(directions)


def directions_to_pose(directions):
    pose = np.zeros((32, 3))
    #pose[H36M_NAMES.index('LHip')] = hip_length / 2

    # for i, parent_name in enumerate(H36M_PARENT_NAMES):


def get_joint_parent(pose, joint_i):
    parent_i = pose.index(H36M_PARENT_NAMES[joint_i])
    return pose[parent_i]

def calc_bone_lengths(pose):
    hip_length = np.linalg.norm(pose[H36M_NAMES.index('RHip')] -
                                pose[H36M_NAMES.index('LHip')])


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


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


def patch_poses(poses, max_out_of_bounds_joints=4.0/32, max_distance=0.3):
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
    max_distance_sq = max_distance ** 2

    poses = np.asarray(poses)

    for i, pose in enumerate(poses):
        # Skip first frame
        if i == 0:
            continue

        # Calculate distance to previous frame
        previous_pose = poses[i-1]
        distance_sq = np.sum((pose - previous_pose) ** 2, axis=1)
        assert distance_sq.shape == (pose.shape[0],), \
            "Should sum for every joint, instead got shape {}" \
            .format(distance_sq.shape)
        out_of_bounds_joints = np.where(distance_sq > max_distance_sq)

        if len(out_of_bounds_joints[0]) > \
           max_out_of_bounds_joints * pose.shape[0]:
            raise ValueError("Too many ({} out of {}) joints out of bounds."
                             .format(len(out_of_bounds_joints[0]), pose.shape[0]))

        # Fill in with previous frame's points
        pose[out_of_bounds_joints, :] = previous_pose[out_of_bounds_joints, :]

    return poses


def is_usable_example(example):
    """Checks whether this example is usable for training.

    Arguments:
      example: dict(feature: [Sparse]Tensor)
    Returns
      is_usable: boolean
    """
    pass


def tensor_rotate(tensor, axis, angle):
    """Rotates tensor by given angle along the axis.
    This is an implementation of Ridrigues' rotation formula.

    Arguments:
      tensor: Tensor, input vector
      axis: Tensor, axis around which to rotate
      angle: float, angle around which to rotate
    Returns:
      tensor: Tensor, rotated vector
    """
    axis = axis / tf.norm(axis)
    return (
        tensor * tf.cos(angle)
        + tf.cross(axis, tensor) * tf.sin(angle)
        + axis * tf.dot(axis, tensor) * (1 - tf.cos(angle)))


def preprocess_example(example):
    """Preprocess an example.

    Arguments:
      example: dict(feature: [Sparse]Tensor)
    Returns
      processed_example: None if it should not be used, same format
        as the example argument otherwise
    """
    try:
        frames = example['points_3d']
        frames = list(map(straighten_pose, frames))
        frames = patch_poses(frames)
        frames = straighten_frames(frames)
        clip['points_3d'] = frames
        writer.send(clip)
        n_clips_out += 1
    except ValueError as e:
        logger.warn(e)
