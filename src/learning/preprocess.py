import random
import itertools
import logging

import numpy as np
import sklearn
import sklearn.cluster
import tensorflow as tf
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
from matplotlib.animation import FuncAnimation

import common.data_utils
import common.openpose_utils

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def cluster_poses(points_3d):
    """
    Args:
      points_3d: list of ndarray (time, dimension)
    """

    points_3d = np.asarray(points_3d)
    centroids = cluster_k_means(points_3d, 16, 4)

    fig = plt.figure(figsize=(18, 18))
    n_rows = int(np.sqrt(len(centroids)))
    n_cols = int(len(centroids) / n_rows)

    all_plots = []
    all_poses = []
    for i, center in enumerate(centroids):
        poses = center

        # Swap y and z axes because mpl shows z as height instead of depth
        poses[:, :, 1], poses[:, :, 2] = poses[:, :, 2].copy(), poses[:, :, 1].copy()
        all_poses.append(poses)

        ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
        lines = common.openpose_utils.get_lines_3d(poses[0])
        plots = common.openpose_utils.plot_lines_3d(lines, ax)
        all_plots.append(plots)

    def update_all(poses):
        for pose, plots in zip(poses, all_plots):
            lines = common.openpose_utils.get_lines_3d(pose)
            common.openpose_utils.update_plots_3d(plots, lines)

    logging.debug(np.asarray(all_poses).shape)

    # all_poses: 16 x (time x pose)
    pose_shape = all_poses[0].shape[1:]
    empty_pose = np.full(pose_shape, np.nan)
    all_poses = list(itertools.zip_longest(*all_poses, fillvalue=empty_pose))
    all_poses = np.asarray(all_poses).swapaxes(0, 1)
    logging.debug(np.asarray(all_poses).shape)

    ani = FuncAnimation(fig, update_all, frames=np.swapaxes(all_poses, 0, 1), interval=80)
    ani.save('clusters.mp4')
    plt.show()



def cluster_k_means(data, n_clusters, n_iterations, window_size=5, reach=5):
    data = list(map(lambda x: np.asarray(x), data))
    centroids = np.random.choice(data, n_clusters).tolist()

    for iteration in range(n_iterations):
        logging.debug("Iteration {}/{}".format(iteration+1, n_iterations))
        assignments = {}

        # Expectation
        for i, sample in enumerate(data):
            min_distance = float('inf')
            closest_cluster_index = None

            for j, centroid in enumerate(centroids):
                distance = dtak_distance(sample, centroid, window_size)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster_index = j

            if closest_cluster_index in assignments:
                assignments[closest_cluster_index].append(i)
            else:
                assignments[closest_cluster_index] = [i]

        # Maximization
        for cluster_index, data_indices in assignments.items():
            logging.debug("cluster {:<4}: {}".format(cluster_index, data_indices))
            cluster_samples = [np.asarray(data[i]) for i in data_indices]
            centroids[cluster_index] = filled_mean(cluster_samples)

    return centroids


def filled_mean(items):
    """Takes the mean of a list of variable-length data.

    Args:
       items: list(ndarray(x, ...))
    Returns:
       ndarray (mean(x), ...)
    """

    frame_shape = items[0].shape[1:]
    empty_frame = np.full(frame_shape, np.nan)

    filled_items = np.asarray(list(itertools.zip_longest(*items, fillvalue=empty_frame))).swapaxes(0, 1)
    mean = np.nanmean(filled_items, axis=0)

    mean_length = int(np.mean([x.shape[0] for x in items]))

    val = mean[:mean_length]
    return val


def dtw_distance(s1, s2, window_size=32):
    """
    Args:
      s1 ndarray (time1, dimension)
      s2 ndarray (time2, dimension)
    """

    n_s1 = s1.shape[0]
    n_s2 = s2.shape[0]
    assert s1.shape[1] == s2.shape[1]
    window_size = max(window_size, abs(n_s1 - n_s2))

    # Row 0 and column 0 are both -1, actually
    DTW = np.array((n_s1 + 1, n_s2 + 1)).fill(np.inf)
    DTW[0, 0] = 0.0

    for i in range(n_s1):
        for j in range(max(0, i - window_size), min(n_s2, i + window_size)):
            DTW[i + 1:j + 1] = (s1[i + 1] - s2[j + 1])**2
            DTW[i + 1:j + 1] += min(DTW[i, j + 1], DTW[i + 1, j], DTW[i, j])

    return np.sqrt(DTW[-1, -1])


def dtak_distance(s1, s2, window_size=32):
    """
    Args:
      s1 ndarray (time1, dimension)
      s2 ndarray (time2, dimension)
    """

    n_s1 = s1.shape[0]
    n_s2 = s2.shape[0]
    assert s1.shape[1] == s2.shape[1]
    window_size = max(window_size, abs(n_s1 - n_s2))

    # Row 0 and column 0 are both -1, actually
    P = np.full((n_s1 + 1, n_s2 + 1), 0.0)
    P[0, 0] = 0.0

    def kernel(X1, X2, i, j, sigma=1000):
        return np.exp(-1.0 / (2 * sigma**2) * np.sum((X1[i] - X2[j])**2))

    for i in range(n_s1 + 1):
        for j in range(max(0, i - window_size), min(n_s2 + 1, i + window_size)):
            k_ij = kernel(s1, s2, i - 1, j - 1)
            p1 = P[i - 1, j] + k_ij
            p2 = P[i - 1, j - 1] + 2 * k_ij
            p3 = P[i, j - 1] + k_ij

            P[i, j] = max(p1, p2, p3)

    return P[-1, -1]


def dtw_lower_bound(s1, s2, reach=32):
    s1 = np.asarray(s1)
    s2 = np.asarray(s2)
    bound = 0

    for i, val in enumerate(s1):
        lower_bound = np.amin(s2[max(0, i - reach):i + reach])
        upper_bound = np.amax(s2[max(0, i - reach):i + reach])

        if val > upper_bound:
            bound += (val - upper_bound)**2
        elif val < lower_bound:
            bound += (val - lower_bound)**2

    return np.sqrt(bound)


if __name__ == '__main__':
    clips = common.data_utils.get_clips()
    clips = list(
        map(lambda clip: clip['points_3d'],
            filter(
                lambda clip: 'points_3d' in clip and len(clip['points_3d']) > 0,
                clips)))
    cluster_poses(clips)
