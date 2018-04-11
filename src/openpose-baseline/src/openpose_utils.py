import os
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
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
    'Neck/Nose',
    'Thorax',
    'RShoulder',
    'RElbow',
    'RWrist',
    'LShoulder',
    'LElbow',
    'LWrist',
    'RHip',
    'RKnee',
    'RFoot',
    'LHip',
    'LKnee',
    'LFoot',
    'REye',
    'LEye',
    'REar',
    'LEar'
]

KEY_OP_PEOPLE = 'people'
KEY_OP_KEYPOINTS = 'pose_keypoints_2d'


def load_clip_keypoints(clip, openpose_output_dir='/root/dev/output/'):
    id = clip['id']
    json_files = os.listdir(openpose_output_dir)
    keypoints = []
    confidences = []
    clip_files = list(filter(lambda f: f.startswith(id + '-'), json_files))
    if len(clip_files) == 0:
        logger.warn("No keypoints found for " + str(id))

    for file_name in sorted(clip_files):
        full_name = os.path.join(openpose_output_dir, file_name)
        with open(full_name) as keypoint_file:
            frame_keypoint_data = json.load(keypoint_file)
            people = frame_keypoint_data[KEY_OP_PEOPLE]
            if len(people) == 0:
                continue

            person = closest_to_center_person(people, clip['center'])
            # person = most_confident_person(people)
            keypoints.append(person[KEY_OP_KEYPOINTS])

    return keypoints


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
        hip = (
            float(
                points[COCO_BODY_PARTS.index('LHip') * 3] +
                points[COCO_BODY_PARTS.index('RHip') * 3]) / 2,
            float(
                points[COCO_BODY_PARTS.index('LHip') * 3] +
                points[COCO_BODY_PARTS.index('RHip') * 3]) / 2
        )

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
        xy.append(keypoints[o+1])

    return xy


def get_confidences(keypoints):
    confidences = []
    for o in range(0, len(keypoints), 3):
        confidences.append(keypoints[o+2])

    return confidences


def get_all_positions(keypoints_arr):
    n_points = keypoints_arr.shape[1]
    indices = np.sort(np.hstack((
        np.arange(0, n_points, 3),
        np.arange(1, n_points, 3))))

    return keypoints_arr[:, indices]


def openpose_to_baseline(coco_frames):
    """Converts a list of OpenPose frames to Baseline-compatible format

    Args:
      coco_frames: ndarray (?x, 18*3) - for every body part xi, yi, ci
    Returns:
      b36m_frames: ndarray (?x, 32*3) - for every H36M body part xi, yi, ci
    """
    if coco_frames.shape[1] != len(COCO_BODY_PARTS) * 3:
        raise ValueError("Expected predictions to be in OpenPose format, i.e. of shape (?, " + str(len(COCO_BODY_PARTS)*3) + "), but got " + str(coco_frames.shape))

    # Store in flattened 2D coordinate array
    h36m_frames = np.zeros((coco_frames.shape[0], len(H36M_NAMES) * 3))

    # Corresponsing destination indices to map OpenPose data into H36M data
    h36m_indices = [np.where(np.array(H36M_NAMES) == name)[0] for name in COCO_BODY_PARTS]
    coco_indices = np.where([len(i) != 0 for i in h36m_indices])[0]
    h36m_indices = np.array([x[0] for x in list(filter(lambda x: len(x) != 0, h36m_indices))])

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
    add_computed_point('Spine', 'Thorax', 'Hip',
                       lambda i, j: i + (j - i) / 4)

    return h36m_frames
