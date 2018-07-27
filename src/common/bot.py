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

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump


class BotController:
    def __init__(self, port=9559):
        host = '192.168.2.198'
        port = int(port)
        self.tts = ALProxy('ALAnimatedSpeech', host, port)
        self.motion = ALProxy('ALMotion', host, port)
        self.posture = ALProxy('ALRobotPosture', host, port)
        self.life = ALProxy('ALAutonomousLife', host, port)
        self.speaking_movement = ALProxy('ALSpeakingMovement', host, port)
        self.leds = ALProxy('ALLeds', host, port)

        # self.motion.setStiffnesses('Body', 1.0)
        self.motion.wakeUp()
        self.life.setAutonomousAbilityEnabled('BasicAwareness', False)
        self.speaking_movement.setEnabled(False)
        self.posture.goToPosture("StandInit", 0.5)

    def say(self, text):
        print(text)
        self.life.setAutonomousAbilityEnabled('BasicAwareness', True)
        self.speaking_movement.setEnabled(True)
        self.speaking_movement.setMode('contextual')
        configuration = {"bodyLanguageMode": "random"}
        self.tts.say(str(text), configuration)
        self.speaking_movement.setEnabled(True)
        self.life.setAutonomousAbilityEnabled('BasicAwareness', False)

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

    def play_angles_animation(self, angles_frames, interval=0.04):
        joint_names = self.motion.getBodyNames('Body')
        names = []
        angle_lists = []

        for frame in angles_frames:
            supported_joints = {str(k): v for k, v in frame.iteritems() if k in joint_names}

            for k, v in supported_joints.iteritems():
                if k not in names:
                    names.append(k)
                    angle_lists.append([])

                angle_lists[names.index(k)].append(v)

        time_lists = list(map(lambda x: list(frange(interval, (len(x)+1)*interval, interval)), angle_lists))
        print(time_lists)
        is_absolute = True

        self.motion.angleInterpolation(names, angle_lists, time_lists, is_absolute)

    def play_angles(self, angles_frames, interval=0.04):
        joint_names = self.motion.getBodyNames('Body')

        for frame in angles_frames:
            supported_joints = {str(k): v for k, v in frame.iteritems() if k in joint_names}
            names = supported_joints.keys()
            angles = supported_joints.values()
            self.motion.setAngles(names, angles, 0.5)
            time.sleep(interval)

    def reset_pose(self):
        self.motion.wakeUp()
        self.posture.goToPosture("StandInit", 0.5)

    def leds_off(self):
      self.leds.off('FaceLeds')

    def leds_on(self):
      self.leds.on('FaceLeds')

    def blink_leds(self):
      led_names = 'FaceLeds'
      self.leds.off(led_names)
      time.sleep(0.1)
      self.leds.on(led_names)


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
