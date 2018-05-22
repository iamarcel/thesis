import time
from naoqi import ALProxy
import almath
import motion
import numpy as np

from . import data_utils
from . import visualize

EFFECTOR_MAP = {
    'LArm': 'LWrist',
    'RArm': 'RWrist'
}


class BotController:
    def __init__(self, port=43155):
        host = 'localhost'
        self.tts = ALProxy('ALTextToSpeech', host, port)
        self.motion = ALProxy('ALMotion', host, port)
        self.posture = ALProxy('ALRobotPosture', host, port)

        # self.motion.setStiffnesses('Body', 1.0)
        self.motion.wakeUp()
        self.posture.goToPosture("StandInit", 0.5)

    def say(self, text):
        self.tts.say(text)

    def move(self, effector, position):
        frame = motion.FRAME_TORSO
        axisMask = almath.AXIS_MASK_VEL
        fractionMaxSpeed = 1.0
        position6d = position + [0, 0, 0]

        self.motion.setPositions(effector, frame, position6d, fractionMaxSpeed,
                                 axisMask)

    def test_move_arm(self):
        effector = 'LArm'
        position = [
            0.16368391960859299, 0.20231289386749268, 0.10050057619810104,
        ]

        self.move(effector, position)
        time.sleep(1)
        self.posture.goToPosture("StandInit", 0.5)

    def play_poses(self, named_poses, interval=0.167):
        first_pose = True
        for pose in named_poses:
            mapped_pose = {k: pose[v] for (k, v) in EFFECTOR_MAP.iteritems()}
            for effector, position in mapped_pose.iteritems():
                print(position)
                self.move(effector, position)
            if first_pose:
                time.sleep(10)
                first_pose = False
            time.sleep(interval)

        # time.sleep(5)
        # self.posture.goToPosture("StandZero", 1.0)

    def play_angles(self, angles_frames, interval=0.04):
        joint_names = self.motion.getBodyNames('Body')

        for frame in angles_frames:
            supported_joints = {k: v for k, v in frame.iteritems() if k in joint_names}
            names = supported_joints.keys()
            angles = supported_joints.values()
            self.motion.setAngles(names, angles, 0.5)
            time.sleep(interval)


def map_pose_to_nao_frame(pose):
    # Pose should be (32, 3)
    # NAO is 574mm high
    # Units should be meters
    # Reference point is torso, 85mm above the hip
    height = 0.574
    reference_delta = [0.0, 0.085, 0.0]

    pose = np.asarray(pose)
    pose = data_utils.normalize_pose_scale(pose, height)
    pose += np.tile(reference_delta, (pose.shape[0], 1))
    pose = np.vstack((-pose[:, 2], pose[:, 0], -pose[:, 1]))
    pose = np.swapaxes(pose, 0, 1)

    return pose.tolist()
