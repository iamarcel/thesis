import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import data_utils
import viz
import re
import cameras
import json
import os
import cv2
import imageio
import logging

from predict_3dpose import create_model, train_dir
from openpose_utils import load_clip_keypoints, openpose_to_baseline
import openpose_utils
import linear_model

# import flags
FLAGS = tf.app.flags.FLAGS
summaries_dir = os.path.join(train_dir, "log")  # Directory for TB summaries

order = [15, 12, 25, 26, 27, 17, 18, 19, 1, 2, 3, 6, 7, 8]

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
    'Head',  # Actually Neck/Nose but there's no head so our people will have really tiny heads
    'Spine',
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KEY_OP_PEOPLE = 'people'
KEY_OP_KEYPOINTS = 'pose_keypoints_2d'

def show_anim_curves(anim_dict, _plt):
    val = np.array(list(anim_dict.values()))
    for o in range(0,36,2):
        x = val[:,o]
        y = val[:,o+1]
        _plt.plot(x, 'r--', linewidth=0.2)
        _plt.plot(y, 'g', linewidth=0.2)
    return _plt


def most_confident_person(people):
    best_person = 0
    best_person_confidence = 0
    for i, person in enumerate(people):
        # Sum the confidence scores
        # NOTE: Confidence is 0 for non-detected joints, which is awesome
        confidence = sum(person[KEY_OP_KEYPOINTS][2::3])
        if confidence > best_person_confidence:
            best_person = i
            best_person_confidence = confidence

    return people[best_person]


def read_clips(file_name='config.json'):
    config = {}
    with open(file_name) as config_file:
        config = json.load(config_file)

    return filter(lambda c: 'processed_2d' not in c or c['processed_2d'] == False, config['clips'])


def process_clips():
    clips = read_clips()

    input_indices = []
    input_points = np.empty((0, len(COCO_BODY_PARTS) * 3))
    input_keys = []

    for i, clip in enumerate(clips):
        print("Queueing 3D poses for " + str(clip['id']))
        keypoints = load_clip_keypoints(clip)
        if len(keypoints) == 0:
            continue

        input_keys.append((len(input_points), len(input_points) + len(keypoints)))
        input_points = np.append(input_points, keypoints, axis=0)
        input_indices.append(i)

    # TODO Try to implement smoothing

    h36m_points = openpose_utils.get_all_positions(openpose_to_baseline(input_points))
    poses_3d = predict_batch(h36m_points)

    config = {}
    with open('config.json') as config_file:
        config = json.load(config_file)

    for i, (start, end) in enumerate(input_keys):
        config['clips'][input_indices[i]]['points_3d'] = poses_3d[start:end,:].tolist()

    with open('config.json', 'w') as config_file:
        json.dump(config, config_file, indent=2)
        print("Wrote " + str('config.json'))


def normalize_batch(frames):
    actions = data_utils.define_actions(FLAGS.action)
    SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]

    # Get training data stats
    rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(
        actions, FLAGS.data_dir)
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
        actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14)

    mu = data_mean_2d[dim_to_use_2d]
    stddev = data_std_2d[dim_to_use_2d]

    # Normalize input
    enc_in = np.divide(frames[:, dim_to_use_2d] - np.tile(mu, (frames.shape[0], 1)), np.tile(stddev, (frames.shape[0], 1)))

    return enc_in, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, data_mean_2d, data_std_2d, dim_to_ignore_2d


def load_model(session, batch_size):
    model = linear_model.LinearModel(
        FLAGS.linear_size,
        FLAGS.num_layers,
        FLAGS.residual,
        FLAGS.batch_norm,
        FLAGS.max_norm,
        batch_size,
        FLAGS.learning_rate,
        summaries_dir,
        FLAGS.predict_14,
        dtype=tf.float16 if FLAGS.use_fp16 else tf.float32)

    # Load a previously saved model
    ckpt = tf.train.get_checkpoint_state(train_dir, latest_filename="checkpoint")

    if ckpt and ckpt.model_checkpoint_path:
        # Check if the specific checkpoint exists
        if os.path.isfile(os.path.join(train_dir,"checkpoint-{0}.index".format(FLAGS.load))):
            ckpt_name = os.path.join( os.path.join(train_dir,"checkpoint-{0}".format(FLAGS.load)) )
        else:
            raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))

        print("Loading model {0}".format(ckpt_name))
        model.saver.restore(session, ckpt.model_checkpoint_path)
        return model
    else:
        print("Could not find checkpoint. Aborting.")
        raise(ValueError, "Checkpoint {0} does not seem to exist".format( ckpt.model_checkpoint_path ) )

    return model


def predict_batch(data, batch_size=128):
    """
    Input:
    data: matrix with shape (#frames, 64)
    """

    data = np.array(data)

    # Wrap in another matrix if there's only a single clip
    if len(data.shape) == 1:
        data = np.array([data])

    if data.shape[1] != 64:
        raise ValueError("Expected data shape to be (?, 64), got " + str(data.shape))

    normalized_data, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, data_mean_2d, data_std_2d, dim_to_ignore_2d = normalize_batch(data)

    with tf.Session() as sess:
        model = load_model(sess, batch_size)
        dp = 1.0
        dec_out = np.zeros((normalized_data.shape[0], 48))
        _, _, points = model.step(sess, normalized_data, dec_out, dp, isTraining=False)

        points = data_utils.unNormalizeData(points, data_mean_3d, data_std_3d, dim_to_ignore_3d)

        points = np.reshape(points, (-1, 32, 3))
        points = np.apply_along_axis(savitzky_golay, 0, points, 5, 2)

        return points


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def read_openpose_json(smooth=True, *args):
    # openpose output format:
    # [x1,y1,c1,x2,y2,c2,...]
    # ignore confidence score, take x and y [x1,y1,x2,y2,...]

    logger.info("start reading data")
    #load json files
    json_files = os.listdir(openpose_output_dir)
    # check for other file types
    json_files = sorted([filename for filename in json_files if filename.endswith(".json")])
    cache = {}
    smoothed = {}
    ### extract x,y and ignore confidence score
    for file_name in json_files:
        logger.debug("reading {0}".format(file_name))
        _file = os.path.join(openpose_output_dir, file_name)
        if not os.path.isfile(_file): raise Exception("No file found!!, {0}".format(_file))
        data = json.load(open(_file))

        #take best person
        if len(data["people"]) == 0:
            continue
        best_person = 0
        best_person_confidence = 0
        for i, person in enumerate(data["people"]):
            # Sum the confidence Visualize Resultsscores
            # NOTE: Confidence is 0 for non-detected joints, which is awesome
            confidence = sum(person["pose_keypoints_2d"][2::3])
            if confidence > best_person_confidence:
                best_person = i
                best_person_confidence = confidence
        _data = data["people"][best_person]["pose_keypoints_2d"]
        xy = []

        #ignore confidence score
        for o in range(0,len(_data),3):
            xy.append(_data[o])
            xy.append(_data[o+1])

        # get frame index from openpose 12 padding
        frame_indx = re.findall("(\d+)", file_name)
        logger.debug("found {0} for frame {1}".format(xy, str(int(frame_indx[0]))))

        #add xy to frame
        cache[int(frame_indx[0])] = xy
    plt.figure(1)
    drop_curves_plot = show_anim_curves(cache, plt)
    pngName = 'png/dirty_plot.png'
    drop_curves_plot.savefig(pngName)

    # FIXME Try to get smoothing working again (doesn't work if frames are skipped because they don't contain people)
    return cache
    # exit if no smoothing
    if not smooth:
        # return frames cache incl. 18 joints (x,y)
        return cache

    if len(json_files) == 1:
        logger.info("found single json file")
        # return frames cache incl. 18 joints (x,y) on single image\json
        return cache

    if len(json_files) <= 8:
        raise Exception("need more frames, min 9 frames/json files for smoothing!!!")

    logger.info("start smoothing")

    # create frame blocks
    first_frame_block = [int(re.findall("(\d+)", o)[0]) for o in json_files[:4]]
    last_frame_block = [int(re.findall("(\d+)", o)[0]) for o in json_files[-4:]]

    ### smooth by median value, n frames
    for frame, xy in cache.items():

        # create neighbor array based on frame index
        forward, back = ([] for _ in range(2))

        # joints x,y array
        _len = len(xy) # 36

        # create array of parallel frames (-3<n>3)
        for neighbor in range(1,4):
            # first n frames, get value of xy in postive lookahead frames(current frame + 3)
            if frame in first_frame_block:
                forward += cache[frame+neighbor]
            # last n frames, get value of xy in negative lookahead frames(current frame - 3)
            elif frame in last_frame_block:
                back += cache[frame-neighbor]
            else:
                # between frames, get value of xy in bi-directional frames(current frame -+ 3)
                forward += cache[frame+neighbor]
                back += cache[frame-neighbor]

        # build frame range vector
        frames_joint_median = [0 for i in range(_len)]
        # more info about mapping in src/data_utils.py
        # for each 18joints*x,y  (x1,y1,x2,y2,...)~36
        for x in range(0,_len,2):
            # set x and y
            y = x+1
            if frame in first_frame_block:
                # get vector of n frames forward for x and y, incl. current frame
                x_v = [xy[x], forward[x], forward[x+_len], forward[x+_len*2]]
                y_v = [xy[y], forward[y], forward[y+_len], forward[y+_len*2]]
            elif frame in last_frame_block:
                # get vector of n frames back for x and y, incl. current frame
                x_v =[xy[x], back[x], back[x+_len], back[x+_len*2]]
                y_v =[xy[y], back[y], back[y+_len], back[y+_len*2]]
            else:
                # get vector of n frames forward/back for x and y, incl. current frame
                # median value calc: find neighbor frames joint value and sorted them, use numpy median module
                # frame[x1,y1,[x2,y2],..]frame[x1,y1,[x2,y2],...], frame[x1,y1,[x2,y2],..]
                #                 ^---------------------|-------------------------^
                x_v =[xy[x], forward[x], forward[x+_len], forward[x+_len*2],
                        back[x], back[x+_len], back[x+_len*2]]
                y_v =[xy[y], forward[y], forward[y+_len], forward[y+_len*2],
                        back[y], back[y+_len], back[y+_len*2]]

            # get median of vector
            x_med = np.median(sorted(x_v))
            y_med = np.median(sorted(y_v))

            # holding frame drops for joint
            if not x_med:
                # allow fix from first frame
                if frame:
                    # get x from last frame
                    x_med = smoothed[frame-1][x]
            # if joint is hidden y
            if not y_med:
                # allow fix from first frame
                if frame:
                    # get y from last frame
                    y_med = smoothed[frame-1][y]

            logger.debug("old X {0} sorted neighbor {1} new X {2}".format(xy[x],sorted(x_v), x_med))
            logger.debug("old Y {0} sorted neighbor {1} new Y {2}".format(xy[y],sorted(y_v), y_med))

            # build new array of joint x and y value
            frames_joint_median[x] = x_med
            frames_joint_median[x+1] = y_med

        smoothed[frame] = frames_joint_median

    # return frames cache incl. smooth 18 joints (x,y)
    return smoothed


def make_3d_prediction(poses_2d):
    enc_in = np.zeros((1, 64))
    enc_in[0] = [0 for i in range(64)]

    actions = data_utils.define_actions(FLAGS.action)

    SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
    rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(
        actions, FLAGS.data_dir)
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
        actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14)

    device_count = {"GPU": 1}
    png_lib = []
    with tf.Session(config=tf.ConfigProto(
            device_count=device_count,
            allow_soft_placement=True)) as sess:
        # plt.figure(3)
        batch_size = 128
        model = create_model(sess, actions, batch_size)
        all_poses_3d = []
        n = 0
        # for i in range(poses_2d.shape[0]):

        frames = openpose_to_baseline(np.array([p[1] for p in poses_2d.items()]))
        frames, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, data_mean_2d, data_std_2d, dim_to_ignore_2d = normalize_batch(frames)

        for i, (frame, xy) in enumerate(poses_2d.items()):
            logger.info("calc frame " + str(i))

            enc_in = np.array([frames[i]])

            # set spine
            spine_x = enc_in[0][24]
            spine_y = enc_in[0][25]

            dp = 1.0
            dec_out = np.zeros((1, 48))
            dec_out[0] = [0 for i in range(48)]
            _, _, poses3d = model.step(sess, enc_in, dec_out, dp, isTraining=False)
            all_poses_3d = []
            enc_in = data_utils.unNormalizeData(enc_in, data_mean_2d, data_std_2d, dim_to_ignore_2d)
            poses3d = data_utils.unNormalizeData(poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d)
            gs1 = gridspec.GridSpec(1, 1)
            gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
            plt.axis('off')
            all_poses_3d.append( poses3d )
            enc_in, poses3d = map( np.vstack, [enc_in, all_poses_3d] )
            subplot_idx, exidx = 1, 1
            max = 0
            min = 10000

            for i in range(poses3d.shape[0]):
                for j in range(32):
                    tmp = poses3d[i][j * 3 + 2]
                    poses3d[i][j * 3 + 2] = poses3d[i][j * 3 + 1]
                    poses3d[i][j * 3 + 1] = tmp
                    if poses3d[i][j * 3 + 2] > max:
                        max = poses3d[i][j * 3 + 2]
                    if poses3d[i][j * 3 + 2] < min:
                        min = poses3d[i][j * 3 + 2]

            for i in range(poses3d.shape[0]):
                for j in range(32):
                    poses3d[i][j * 3 + 2] = max - poses3d[i][j * 3 + 2] + min
                    poses3d[i][j * 3] += (spine_x - 630)
                    poses3d[i][j * 3 + 2] += (500 - spine_y)

            # Plot 3d predictions
            ax = plt.subplot(gs1[subplot_idx - 1], projection='3d')
            ax.view_init(18, -70)
            logger.debug(np.min(poses3d))

            p3d = poses3d
            logger.debug(poses3d)
            viz.show3Dpose(p3d, ax, lcolor="#9b59b6", rcolor="#2ecc71")

            pngName = 'png/test_{0}.png'.format(str(n))
            n += 1
            plt.savefig(pngName)
            png_lib.append(imageio.imread(pngName))

    logger.info("creating Gif png/movie_smoothing.gif, please Wait!")
    imageio.mimsave('png/movie_smoothing.gif', png_lib, fps=FLAGS.gif_fps)
    return np.array(all_poses_3d)


def read_specified_output():
    smoothed = read_openpose_json()
    logger.info("reading and smoothing done. start feeding 3d-pose-baseline")
    plt.figure(2)
    smooth_curves_plot = show_anim_curves(smoothed, plt)
    pngName = 'png/smooth_plot.png'
    smooth_curves_plot.savefig(pngName)

    make_3d_prediction(smoothed)


def main(_):
    process_clips()


if __name__ == "__main__":

    openpose_output_dir = FLAGS.openpose

    level = {0: logging.ERROR,
             1: logging.WARNING,
             2: logging.INFO,
             3: logging.DEBUG}

    logger.setLevel(level[FLAGS.verbose])

    tf.app.run()
