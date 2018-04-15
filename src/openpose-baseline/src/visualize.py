import os
import json
import random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as img
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import viz
from openpose_utils import load_clip_keypoints, openpose_to_baseline, get_confidences, get_positions


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
    config = {}
    with open('config.json') as config_file:
        config = json.load(config_file)
    image_root = config['image_root']
    image_extension = config['image_extension']
    print(len(config['clips']))

    while True:
        try:
            if n == -1:
                n = random.randint(0, len(config['clips']))
            clip = config['clips'][n]
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


preview_clip(200)
