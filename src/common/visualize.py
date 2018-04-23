import os
import json
import random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as img
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from common.openpose_utils import load_clip_keypoints, openpose_to_baseline, get_confidences, get_positions
from . import data_utils, openpose_utils


def plot_skeleton(points, points_2d, image_paths, confidences=None):
    if (points.shape[1] != 32 or points.shape[2] != 3):
        raise ValueError(
            "Expected points.shape to be (?, 32, 3), got " + str(points.shape))

    # Swap y and z axes because mpl shows z as height instead of depth
    points[:, :, 1], points[:, :, 2] = points[:, :, 2].copy(), points[:, :, 1].copy()

    fig = plt.figure(figsize=(18, 5))
    ax_2d = fig.add_subplot(131)
    ax = fig.add_subplot(132, projection='3d')
    ax.view_init(18, -70)
    ax.set_aspect(1)
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_zlim(-500, 500)
    ax_img = fig.add_subplot(133)
    ax_img.axis('off')

    viz.show2Dpose(points_2d[0], ax_2d, confidences=confidences[0])
    # axes_3d = draw_3d_pose(points[0], ax=ax)
    viz.show3Dpose(points[0], ax)
    ax_img = plt.imshow(img.imread(image_paths[0]), animated=True)

    def update(frame_enumerated):
        img_i, frame = frame_enumerated
        ax_img.set_data(img.imread(image_paths[img_i]))

        ax_2d.clear()
        viz.show2Dpose(points_2d[img_i], ax_2d, confidences=confidences[img_i])
        ax_2d.invert_yaxis()

        ax.clear()
        viz.show3Dpose(frame, ax)
        ax.invert_zaxis()

    ani = FuncAnimation(fig, update, frames=enumerate(points), interval=80)
    # ani.save('viz.mp4')

    plt.show()


def image_path(id, i, root, ext):
    return os.path.join(root, id + "-" + ("{:06d}".format(i)) + ext)


def preview_clip(n=-1):
    clips = list(data_utils.get_clips(path='clips-with-3d.jsonl'))
    config = {}
    with open('config.json') as config_file:
        config = json.load(config_file)
    image_root = config['image_root']
    image_extension = config['image_extension']
    print(len(clips))

    if n == -1:
        n = random.randint(0, len(clips))

    while True:
        try:
            clip = clips[n]
            images = [image_path(clip['id'], i + 1, image_root, image_extension) for i in range(clip['end'] - clip['start'])]

            keypoints = openpose_to_baseline(np.array(load_clip_keypoints(clip)))
            points_2d = np.array(list(map(get_positions, keypoints)))

            plot_skeleton(np.array(clip['points_3d']),
                          points_2d,
                          images,
                          confidences=list(map(get_confidences, keypoints)))
            return
        except ValueError as e:
            print(e)
            print("Trying another clip...")
            n = random.randint(0, len(clips))


def animate_3d_poses(points):
    # Swap y and z axes because mpl shows z as height instead of depth
    # points[:, :, 1], points[:, :, 2] = (
    #     points[:, :, 2].copy(), points[:, :, 1].copy())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(18, -70)
    ax.set_aspect(1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    show3Dpose(points[0], ax)

    def update(frame_enumerated):
        img_i, frame = frame_enumerated

        ax.clear()
        show3Dpose(frame, ax)
        ax.invert_zaxis()

    ani = FuncAnimation(fig, update, frames=enumerate(points), interval=80)
    # ani.save('viz.mp4')

    plt.show()


def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False): # blue, orange
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

    assert channels.size == len(openpose_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (len(openpose_utils.H36M_NAMES), -1) )

    I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
    J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
    LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    # Make connection matrix
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, marker='o', markersize=2, lw=1, c=lcolor if LR[i] else rcolor)
        # print_line_lengths([x, y, z], I[i], J[i])

    RADIUS = 0.5 # space around the subject
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

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

