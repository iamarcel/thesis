import os
import json
import logging
import numpy as np
import scipy

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

KEY_OP_PEOPLE = 'people'
KEY_OP_KEYPOINTS = 'pose_keypoints_2d'

DEFAULT_OPENPOSE_OUTPUT_PATH = '/root/dev/output/'


def load_clip_keypoints(clip,
                        openpose_output_dir=DEFAULT_OPENPOSE_OUTPUT_PATH,
                        min_confidence=0.6,
                        filenames=None):
    if filenames is None:
        filenames = get_outputs(openpose_output_dir)
    keypoints = []
    clip_files = list(get_clip_files(clip, filenames=filenames))
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

            mean_confidence = np.mean(
                get_confidences(person[KEY_OP_KEYPOINTS]))
            if np.any(mean_confidence < min_confidence):
                raise ValueError("Clip has pose with confidence score" +
                                 " lower than {}".format(min_confidence))

            keypoints.append(person[KEY_OP_KEYPOINTS])

    return keypoints


def get_clip_files(clip,
                   filenames=None,
                   openpose_output_dir=DEFAULT_OPENPOSE_OUTPUT_PATH):

    if filenames is None:
        filenames = get_outputs(openpose_output_dir)

    return map(lambda f: os.path.join(openpose_output_dir, f),
               sorted(
                   filter(lambda f: f.startswith(clip['id'] + '-'),
                          filenames)))


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
      b36m_frames: ndarray (?x, 32*3) - for every H36M body part xi, yi, ci
    """
    if coco_frames.shape[1] != len(COCO_BODY_PARTS) * 3:
        raise ValueError(
            "Expected predictions to be in OpenPose format, i.e. of shape (?, "
            + str(len(COCO_BODY_PARTS) * 3) + "), but got " + str(
                coco_frames.shape))

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
    add_computed_point('Spine', 'Thorax', 'Hip', lambda i, j: i + (j - i) / 4)

    return h36m_frames


def get_named_pose(in_pose, fmt='h36m'):
    names = []
    if fmt == 'h36m':
        names = H36M_NAMES
    elif fmt == 'coco':
        names = COCO_BODY_PARTS
    elif fmt == 'sh':
        raise NotImplementedError("Doesn't support SH yet")
    else:
        raise ValueError("Unrecognized pose format {}".format(fmt))

    out_pose = {}
    for i, name in enumerate(names):
        if name is None or name == '':
            continue
        out_pose[name] = in_pose[i]

    return out_pose


def get_lines_3d(pose):
    pose = np.asarray(pose)
    assert pose.shape == (len(H36M_NAMES), 3), \
        ("pose should have shape ({}, 3), instead got {}"
         .format(len(H36M_NAMES), pose.shape))

    start_points = np.array([
        1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27
    ]) - 1  # start points
    end_points = np.array([
        2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28
    ]) - 1  # end points

    lines = []
    for i in np.arange(len(start_points)):
        x, y, z = [np.array([pose[start_points[i], j], pose[end_points[i], j]])
                   for j in range(3)]
        lines.append((x, y, z))

    return lines


def plot_lines_3d(lines, ax, lcolor="#3498db", rcolor="#e74c3c"):
    is_left = np.array(
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    ax.set_aspect(1)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.invert_zaxis()

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_zticklabels([])

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 0.1, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)

    plots = []
    for i, (x, y, z) in enumerate(lines):
        plot, = ax.plot(
            x,
            y,
            z,
            marker='o',
            markersize=2,
            lw=1,
            c=lcolor if is_left[i] else rcolor)
        plots.append(plot)

    return plots


def update_plots_3d(plots, lines):
    for plot, line in zip(plots, lines):
        plot.set_data(line[0], line[1])
        plot.set_3d_properties(line[2])


def plot_3d_pose(pose,
                 ax,
                 lcolor="#3498db",
                 rcolor="#e74c3c",
                 add_labels=False):
    """
    Visualize a 3d skeleton

    Args
        channels: 96x1 vector. The pose to plot.
        ax: matplotlib 3d axis to draw on
        lcolor: color for left part of the body
        rcolor: color for right part of the body
        add_labels: whether to add coordinate labels
    Returns
        Nothing. Draws on ax.
    """

    pose = np.asarray(pose)
    assert pose.shape == (len(H36M_NAMES), 3), \
        ("pose should have shape ({}, 3), instead got {}"
         .format(len(H36M_NAMES), pose.shape))

    start_points = np.array([
        1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27
    ]) - 1  # start points
    end_points = np.array([
        2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28
    ]) - 1  # end points
    is_left = np.array(
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection matrix
    for i in np.arange(len(start_points)):
        x, y, z = [np.array([pose[start_points[i], j], pose[end_points[i], j]])
                   for j in range(3)]
        ax.plot(
            x,
            y,
            z,
            marker='o',
            markersize=2,
            lw=1,
            c=lcolor if is_left[i] else rcolor)

    RADIUS = 0.5  # space around the subject
    xroot, yroot, zroot = pose[0, 0], pose[0, 1], pose[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    # Get rid of the ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_zticklabels([])
    ax.set_aspect('equal')

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 0.1, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)


def plot_3d_animation(poses, ax):
    if (poses.shape[1] != 32 or poses.shape[2] != 3):
        raise ValueError(
            "Expected poses.shape to be (?, 32, 3), got " + str(poses.shape))

    # Swap y and z axes because mpl shows z as height instead of depth
    poses[:, :, 1], poses[:, :, 2] = poses[:, :, 2].copy(), poses[:, :, 1].copy()

    ax.set_aspect(1)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.invert_zaxis()

    plot_3d_pose(poses[0], ax)
    def update(pose):
        ax.clear()
        plot_3d_pose(pose, ax)
        ax.invert_zaxis()

    return update, poses

    # ani = FuncAnimation(fig, update, frames=poses, interval=80)
    # ani.save('viz.mp4')


def get_pose_angles(pose):
    pose = get_named_pose(pose)
    angles = {}

    def norm_joint(name_a, name_b):
        vec = np.asarray(pose[name_b]) - np.asarray(pose[name_a])
        vec /= scipy.linalg.norm(vec)
        return vec

    def angle_between(vec_a, vec_b):
        return np.arccos(np.dot(np.asarray(vec_a), np.asarray(vec_b)))

    chest = norm_joint('Hip', 'Thorax')

    # Only Pepper has a hip
    angles['HipRoll'] = angle_between(chest, [-1., 0., 0.]) - np.pi / 2
    angles['HipPitch'] = angle_between(chest, [0., 0., -1.]) - np.pi / 2

    r_upper_arm = norm_joint('RShoulder', 'RElbow')
    angles['RShoulderPitch'] = angle_between(chest, r_upper_arm) - np.pi / 2
    angles['RShoulderRoll'] = angle_between([0.0, 0.0, -1.0], r_upper_arm)

    l_upper_arm = norm_joint('LShoulder', 'LElbow')
    angles['LShoulderPitch'] = angle_between(chest, l_upper_arm) - np.pi / 2
    angles['LShoulderRoll'] = - angle_between([0.0, 0.0, -1.0], l_upper_arm)

    r_elbow = norm_joint('RElbow', 'RWrist')
    angles['RElbowRoll'] = angle_between(r_upper_arm, r_elbow)

    l_elbow = norm_joint('LElbow', 'LWrist')
    angles['LElbowRoll'] = - angle_between(l_upper_arm, l_elbow)

    head = norm_joint('Thorax', 'Head')
    angles['HeadPitch'] = angle_between([0., 0., -1.], head) - np.pi / 2
    angles['HeadYaw'] = angle_between([-1., 0., 0.], head) - np.pi / 2
    print(angles)

    return angles
