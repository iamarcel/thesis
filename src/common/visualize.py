# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import json
import random
import itertools

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as img
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as PathEffects
from matplotlib import rcParams
import numpy as np
from adjustText import adjust_text

from common.pose_utils import load_clip_keypoints, openpose_to_baseline, get_confidences, get_positions, Pose
from . import data_utils, pose_utils

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['UGent Panno Text', 'Tahoma']

UGENT_BLUE = "#1E64C8"
UGENT_YELLOW = "#FFD200"
UGENT_EA = "#6F71B9"


flatten = lambda l: [item for sublist in l for item in sublist]

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def create_plot_grid(ny, nx, fig=None):
  if fig is None:
    fig = plt.figure()

  axes = []
  for y in range(ny):
    row_axes = []
    for x in range(nx):
      i = y * nx + x + 1
      row_axes.append(fig.add_subplot(ny, nx, i, projection='3d'))
    axes.append(row_axes)
  return axes


def plot_skeleton(points, points_2d, image_paths, confidences=None):
  if (points.shape[1] != 32 or points.shape[2] != 3):
    raise ValueError("Expected points.shape to be (?, 32, 3), got " +
                     str(points.shape))

  fig = plt.figure(figsize=(18, 5))
  ax_2d = fig.add_subplot(131)
  ax = fig.add_subplot(132, projection='3d')
  ax.view_init(18, -70)
  ax_img = fig.add_subplot(133)
  ax_img.axis('off')

  show2Dpose(points_2d[0], ax_2d, confidences=confidences[0])
  # axes_3d = draw_3d_pose(points[0], ax=ax)
  show3Dpose(points[0], ax)
  ax_img = plt.imshow(img.imread(image_paths[0]), animated=True)

  def update(frame_enumerated):
    img_i, frame = frame_enumerated
    ax_img.set_data(img.imread(image_paths[img_i]))

    ax_2d.clear()
    show2Dpose(points_2d[img_i], ax_2d, confidences=confidences[img_i])

    ax.clear()
    show3Dpose(frame, ax)

  ani = FuncAnimation(fig, update, frames=enumerate(points), interval=80)
  # ani.save('viz.mp4')

  plt.show()


def get_clip_image_paths(clip):
  config = {}
  with open('config.json') as config_file:
    config = json.load(config_file)
  image_root = config['image_root']
  image_extension = config['image_extension']
  images = [
      image_path(clip['id'], i + 1, image_root, image_extension)
      for i in range(clip['end'] - clip['start'])
  ]

  return images


def image_path(id, i, root, ext):
  return os.path.join(root, id + "-" + ("{:06d}".format(i)) + ext)


def preview_clip(n=-1, openpose_output_dir='../output/'):
  clips = list(data_utils.get_clips(path='clips.jsonl'))
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
      images = [
          image_path(clip['id'], i + 1, image_root, image_extension)
          for i in range(clip['end'] - clip['start'])
      ]

      keypoints = openpose_to_baseline(np.array(load_clip_keypoints(clip, openpose_output_dir=openpose_output_dir)))
      points_2d = np.array(list(map(get_positions, keypoints)))

      plot_skeleton(
          np.array(clip['points_3d']),
          points_2d,
          images,
          confidences=list(map(get_confidences, keypoints)))
      return
    except ValueError as e:
      print(e)
      print("Trying another clip...")
      n = random.randint(0, len(clips))


def create_pose_format_comparison(clip):
  poses = pose_utils.load_clip_keypoints(
      clip,
      openpose_output_dir='./openpose/src/output/')
  image = get_clip_image_paths(clip)[0]
  if not os.path.isfile(image):
    raise ValueError('No image for this frame present.')

  fig = plt.figure(figsize=(6, 3), dpi=300)
  ax_h36m = fig.add_subplot(121)
  ax_h36m.axis('off')
  ax_coco = fig.add_subplot(122)
  ax_coco.axis('off')

  poses = np.array(poses)
  pose = np.reshape(poses[0, 0, :], (-1, 3))
  plot_2d_pose(ax_coco, pose, fmt='coco', joint_labels=True, joint_label_type='index')

  pose = np.reshape(openpose_to_baseline(poses)[0, :], (-1, 3))
  plot_2d_pose(ax_h36m, pose, fmt='h36m', joint_labels=True, joint_label_type='index')

  _save_for_report(fig, 'pose-format-comparison')


def create_sanity_check_2d(clip):
  poses = pose_utils.load_clip_keypoints(
      clip,
      openpose_output_dir='./openpose/src/output/')
  image = get_clip_image_paths(clip)[0]
  if not os.path.isfile(image):
    raise ValueError('No image for this frame present.')

  fig = plt.figure(figsize=(10, 3), dpi=300)
  ax_img = fig.add_subplot(121)
  ax_img.axis('off')
  ax_img = plt.imshow(img.imread(image))

  ax_skel = fig.add_subplot(122)
  plot_2d_pose(ax_skel, poses[0], fmt='coco')

  _save_for_report(fig, 'sanity-check-openpose')


def create_crappy_3d_detection_plot(clips_path='clips.dirty.jsonl', n_poses=3):
  fig = plt.figure(figsize=(3 * n_poses, 3), dpi=300)

  for i in range(n_poses):
    points = data_utils.get_crappy_3d_prediction(clips_path)
    ax_3d = fig.add_subplot(1, n_poses, i + 1, projection='3d')
    ax_3d.invert_zaxis()
    show_3d_pose(points[0, :], ax=ax_3d, radius=None)

  _save_for_report(fig, 'crappy-3d-detection')


def create_crappy_3d_detection_animation(clips_path='clips.dirty.jsonl', n_gestures=3):
  gestures = [data_utils.get_crappy_3d_prediction(clips_path) for _ in range(n_gestures)]
  return create_sanity_check_gesture_grid_animation(
    gestures,
    'crappy-3d-detections',
    fmt='position',
    figsize=(n_gestures * 3, 3),
    n_rows=1,
    radius=None)


def create_sanity_check_2d_3d(clip):
  poses = pose_utils.load_clip_keypoints(
      clip,
      openpose_output_dir='./openpose/src/output/')

  fig = plt.figure(figsize=(10, 3), dpi=300)
  ax_2d = fig.add_subplot(121)
  ax_2d.axis('off')
  ax_3d = fig.add_subplot(122, projection='3d')
  ax_3d.axis('off')

  pose_2d = poses[0][0]
  if 0 in pose_2d:
    raise ValueError('There\'s a zero, not cool.')
  pose_3d = np.array(clip['points_3d'])[0, :]

  plot_2d_pose(ax_2d, poses[0], fmt='coco')
  show_3d_pose(pose_3d, ax=ax_3d, radius=0.3)
  ax_3d.invert_zaxis()

  _save_for_report(fig, 'sanity-check-3d')


def create_sanity_check_pipeline(clip, openpose_output_dir='../output/'):
  images = get_clip_image_paths(clip)
  points_2d = pose_utils.load_clip_keypoints(
      clip,
      openpose_output_dir=openpose_output_dir)
  # points_2d = openpose_to_baseline(points_2d)

  image = get_clip_image_paths(clip)[0]
  if not os.path.isfile(image):
    raise ValueError('No image for this frame present.')

  fig = plt.figure(figsize=(10, 5), dpi=300)
  ax_img = fig.add_subplot(131)
  ax_img.axis('off')
  ax_img = plt.imshow(img.imread(image))

  ax_2d = fig.add_subplot(132)
  ax_2d.axis('off')
  ax_3d = fig.add_subplot(133, projection='3d')
  ax_3d.axis('off')

  points_3d = np.array(clip['points_3d'])[0, :]

  show_2d_pose(openpose_to_baseline(points_2d), ax=ax_2d)
  show_3d_pose(points_3d, ax=ax_3d, radius=0.3)
  ax_3d.invert_zaxis()

  fig.suptitle(u"“" + clip['subtitle'] + u"”", fontsize=16)

  fig.tight_layout(pad=0)
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
  plt.axis('off')
  plt.savefig('../img/sanity-check-pipeline.png', pad_inches=0)
  plt.savefig('../img/sanity-check-pipeline.pgf', pad_inches=0)

  print("Don't forget to check the image include path in the generated .pgf file")


def create_sanity_check_pipeline_animation(clip, openpose_output_dir='../output/'):
  # Set up the data
  name = 'sanity-check-pipeline'
  images = get_clip_image_paths(clip)
  points_2d = pose_utils.load_clip_keypoints(
      clip,
      openpose_output_dir=openpose_output_dir)

  images = get_clip_image_paths(clip)
  print(images)
  if not os.path.isfile(images[0]):
    raise ValueError('No image for this clip present.')

  # Set up the plot
  fig = plt.figure(figsize=(10, 5), dpi=300)
  ax_img = fig.add_subplot(131)
  ax_img.axis('off')
  ax_img = plt.imshow(img.imread(images[0]), animated=True)

  ax_2d = fig.add_subplot(132)
  ax_2d.axis('off')
  ax_3d = fig.add_subplot(133, projection='3d')
  ax_3d.axis('off')

  fig.suptitle(u"“" + clip['subtitle'] + u"”", fontsize=16)

  # Prepare the data
  points_3d = np.array(clip['points_3d'])
  # points_2d = pose_utils.get_all_positions(openpose_to_baseline(points_2d))
  # points_2d = [get_positions(Pose(x, fmt='coco').as_list(fmt='h36m')) for x in points_2d]

  # Create the animation
  spec_2d = get_2d_pose_animation_spec(points_2d, fmt='coco')
  setup_2d, update_2d = spec_2d
  setup_2d(ax_2d)

  spec_3d = get_pose_animation_spec(points_3d, radius=0.3)
  setup_3d, update_3d = spec_3d
  setup_3d(ax_3d)

  def update(i):
    ax_img.set_data(img.imread(images[i]))
    update_2d(ax_2d, i)
    update_3d(ax_3d, i)

    if i == 1:
      _save_for_report(fig, name, do_close=False)

  animation = FuncAnimation(
    fig,
    update,
    interval=40,
    save_count=len(images))

  fig.tight_layout(pad=0)
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
  animation.save('../img/{}.mp4'.format(name))



def create_pose_vector_plot(clip):
  poses = pose_utils.load_clip_keypoints(
      clip,
      openpose_output_dir='./openpose/src/output/')

  fig = plt.figure(figsize=(5, 5), dpi=300)
  ax = fig.add_subplot(111, projection='3d')
  ax.axis('off')

  points_3d = np.array(clip['points_3d'])[0, :]
  points_3d = _mpl_reorder_pose(points_3d)
  vals = pose_utils.get_named_pose(points_3d)
  _mpl_setup_ax_3d(ax)

  # Make connection matrix
  joints = pose_utils.JOINTS
  for i in range(joints.shape[0]):
    x, y, z = [np.array([vals[joints[i, 0]][j], vals[joints[i, 1]][j]]) for j in range(3)]
    a = Arrow3D(x, y, z, mutation_scale=10,
                lw=2, arrowstyle="->", color=UGENT_BLUE,
                alpha=0.9 if joints[i, 1] in pose_utils.LEFT_FRONT_BODY_PARTS else 0.2)
    ax.add_artist(a)

  ax.add_artist(
    Arrow3D([0, 0.2], [0, 0], [0, 0],
            mutation_scale=10,
            lw=1, arrowstyle="->", color="red", alpha=0.5))
  ax.add_artist(
    Arrow3D([0, 0], [0, 0.2], [0, 0],
            mutation_scale=10,
            lw=1, arrowstyle="->", color="blue", alpha=0.5))
  ax.add_artist(
    Arrow3D([0, 0], [0, 0], [0, 0.2],
            mutation_scale=10,
            lw=1, arrowstyle="->", color="green", alpha=0.5))

  ax.invert_zaxis()
  fig.tight_layout(pad=0)
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
  plt.axis('off')
  plt.savefig('../img/pose-vectors.png', pad_inches=0)
  # plt.savefig('../img/pose-vectors.pgf', pad_inches=0)  # Huge file size


def create_sanity_check_gesture_grid(gestures, name, figsize=(6, 3)):
  """Creates a grid of frames of the given gesture (one grid point per gesture,
showing the first frame)"""

  n_gestures = len(gestures)

  fig = plt.figure(figsize=figsize, dpi=300)
  axes = create_plot_grid(2, n_gestures / 2, fig=fig)
  axes = flatten(axes)

  for i, gesture in enumerate(gestures):
    ax = axes[i]
    # Pick random frame from the gesture
    points = Pose(random.choice(gesture)).as_list(fmt='h36m')
    show_3d_pose(points, ax=ax, radius=0.3)
    ax.invert_zaxis()

  _save_for_report(fig, name)


def create_sanity_check_gesture_grid_animation(
    gestures,
    name,
    fmt='angles',
    figsize=(6, 3),
    n_rows=2,
    radius=0.3):
  """Creates a grid of frames of the given gesture (one grid point per gesture,
showing the first frame)"""

  n_gestures = len(gestures)

  fig = plt.figure(figsize=figsize, dpi=300)
  axes = create_plot_grid(n_rows, n_gestures / n_rows, fig=fig)
  axes = flatten(axes)

  animation_specs = []
  max_gesture_length = 0

  for i, gesture in enumerate(gestures):
    ax = axes[i]
    if fmt == 'angles':
      gesture = [Pose(frame).as_list('h36m') for frame in gesture]
    max_gesture_length = max(max_gesture_length, len(gesture))

    spec = get_pose_animation_spec(gesture, radius=radius)
    animation_specs.append(spec)
    setup, update = spec
    setup(ax)

  def update(i):
    for j, spec in enumerate(animation_specs):
      setup, update = spec
      update(axes[j], i)

    if i == 1:
      _save_for_report(fig, name, do_close=False)

  animation = FuncAnimation(
    fig,
    update,
    interval=40,
    save_count=max_gesture_length)

  fig.tight_layout(pad=0)
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
  animation.save('../img/{}.mp4'.format(name))


def animate_3d_poses(points, add_labels=False, ax=None, save=False):
  # Swap y and z axes because mpl shows z as height instead of depth
  # points[:, :, 1], points[:, :, 2] = (
  #     points[:, :, 2].copy(), points[:, :, 1].copy())
  points = np.asarray(points)
  # points = _mpl_reorder_poses(points)

  if ax is None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

  fig = ax.get_figure()

  show3Dpose(points[0], ax, add_labels=add_labels)

  def update(frame_enumerated):
    img_i, frame = frame_enumerated

    ax.clear()
    show3Dpose(frame, ax, add_labels=add_labels)

  ani = FuncAnimation(
      fig,
      update,
      frames=itertools.cycle(enumerate(points)),
      interval=40,
      repeat=True,
      save_count=points.shape[0])

  if save:
    ani.save('viz.mp4')

  plt.show()


def get_pose_animation_spec(gesture, add_labels=False, radius=0.3):
  gesture = np.asarray(gesture)

  def setup(ax):
    points = _mpl_reorder_pose(gesture[0, :, :])
    show3Dpose(points, ax, add_labels=add_labels, radius=radius)
    ax.invert_zaxis()

  def update(ax, i):
    if i >= gesture.shape[0]:
      return

    ax.clear()
    points = _mpl_reorder_pose(gesture[i, :, :])
    show3Dpose(points, ax, add_labels=add_labels, radius=radius)
    ax.invert_zaxis()

  return (setup, update)


def get_2d_pose_animation_spec(gesture, fmt):
  gesture = np.asarray(gesture)

  def setup(ax):
    points = gesture[0, :]
    plot_2d_pose(ax, points, fmt=fmt)
    ax.axis('off')

  def update(ax, i):
    if i >= gesture.shape[0]:
      return

    ax.clear()
    points = gesture[i, :]
    plot_2d_pose(ax, points, fmt=fmt)
    ax.axis('off')

  return (setup, update)


def show_3d_pose(points, ax=None, add_labels=False, radius=0.5):
  points = np.asarray(points)
  points = _mpl_reorder_pose(points)

  if ax is None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
  show3Dpose(points, ax, add_labels=add_labels, radius=radius)


def show3Dpose(channels,
               ax,
               lcolor=UGENT_BLUE,
               rcolor=UGENT_EA,
               add_labels=False,
               radius=0.5):  # blue, orange
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

  assert channels.size == len(
      pose_utils.H36M_NAMES
  ) * 3, "channels should have 96 entries, it has %d instead" % channels.size
  vals = np.reshape(channels, (len(pose_utils.H36M_NAMES), -1))

  _mpl_setup_ax_3d(ax, add_labels=add_labels, radius=radius)

  I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27
               ]) - 1  # start points
  J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28
               ]) - 1  # end points
  LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange(len(I)):
    x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
    ax.plot(
        x, y, z, lw=2, marker='o', c=lcolor if LR[i] else rcolor)
    # print_line_lengths([x, y, z], I[i], J[i])


def show2Dpose(channels,
               ax,
               lcolor=UGENT_BLUE,
               rcolor=UGENT_EA,
               add_labels=False,
               confidences=None,
               radius=0.3):
  """
    Visualize a 2d skeleton

    Args
        channels: 64x1 vector. The pose to plot.
        ax: matplotlib axis to draw on
        lcolor: color for left part of the body
        rcolor: color for right part of the body
        add_labels: whether to add coordinate labels
    Returns
        Nothing. Draws on ax.
    """

  assert channels.shape[0] == len(
      pose_utils.H36M_NAMES
  ) * 2, "channels should have 64 entries, it has %d instead" % channels.size
  vals = np.reshape(channels, (len(pose_utils.H36M_NAMES), -1))

  _mpl_setup_ax_2d(ax, radius=radius)

  I = np.array(
      [1, 2, 3, 1, 7, 8, 1, 13, 14, 14, 18, 19, 14, 26, 27],
      dtype=int) - 1  # start points
  J = np.array(
      [2, 3, 4, 7, 8, 9, 13, 14, 16, 18, 19, 20, 26, 27, 28],

    dtype=int) - 1  # end points
  LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange(len(I)):
    x, y = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(2)]

    color = lcolor if LR[i] else rcolor
    if confidences is not None:
      color = matplotlib.cm.viridis(confidences[I[i]])

    ax.plot(x, y, lw=2, marker='o', c=color)

  if confidences is not None:
    indices = np.sort(np.unique(np.hstack((I, J))))
    confidences = np.array(confidences)
    plt.title("Average confidence: {:.3f}".format(
        np.mean(confidences[indices])))


def show_2d_pose(points, ax=None, add_labels=False):
  points = np.array(list(map(get_positions, points)))
  points = np.reshape(points[1, :], (-1, 2))
  points = points - points[0, :]  # Use hip as center
  height = abs(points[15, 1] - points[8, 1])
  points /= (height * 1.2)
  points = points.flatten()

  if ax is None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
  show2Dpose(points, ax, add_labels=add_labels)


def plot_2d_pose(ax, point_list, fmt='h36m', joint_labels=False, joint_label_type='name'):
  pose = pose_utils.get_named_pose(point_list, fmt)
  joints = pose_utils.JOINTS if fmt == 'h36m' else pose_utils.COCO_JOINTS
  point_names = pose_utils.H36M_NAMES if fmt == 'h36m' else pose_utils.COCO_BODY_PARTS

  start_names = joints[:, 0]
  start_points = [pose[joint_name] for joint_name in start_names]

  end_names = joints[:, 1]
  end_points = [pose[joint_name] for joint_name in end_names]

  values = np.asarray(pose.values())
  radius_x = np.max(values[:, 0]) - np.min(values[:, 0])
  radius_y = np.max(values[:, 1]) - np.min(values[:, 1])
  radius = max(radius_x, radius_y) * 0.5 * 1.3
  center = np.mean([pose['LHip'], pose['RHip']], axis=0)

  _mpl_setup_ax_2d(ax, radius=radius, center=center)

  for start, end in zip(start_points, end_points):
    x = [start[0], end[0]]
    y = [start[1], end[1]]

    ax.plot(x, y, lw=2, marker='o', color=UGENT_BLUE)

  # Annotate
  if joint_labels:
    texts = []
    for k, v in pose.items():
      texts.append(ax.text(
        v[0],
        v[1],
        k if joint_label_type == 'name' else point_names.index(k),
        path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")]
      ))
    adjust_text(texts, ax=ax)


def _mpl_reorder_poses(points):
  """Reorders points so Matplotlib shows them in the orientation we expect"""
  # Swap y and z axes because mpl shows z as height instead of depth
  points[:, :, 1], points[:, :, 2] = points[:, :, 2].copy(), points[:, :,
                                                                    1].copy()
  return points


def _mpl_reorder_pose(points):
  """Reorders points so Matplotlib shows them in the orientation we expect"""
  # Swap y and z axes because mpl shows z as height instead of depth
  points[:, 1], points[:, 2] = points[:, 2].copy(), points[:, 1].copy()
  return points


def _mpl_setup_ax_2d(ax, radius=0.5, add_labels=False, center=[0., 0.]):

  ax.set_aspect('equal', 'box')

  if radius is not None:
    ax.set_xlim([center[0] - radius, center[0] + radius])
    ax.set_ylim([center[1] - radius, center[1] + radius])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
  else:
    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Get rid of tick labels
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

  ax.invert_yaxis()


def _mpl_setup_ax_3d(ax, radius=0.5, add_labels=False):
  _mpl_setup_ax_2d(ax, radius, add_labels)

  if radius is not None:
    ax.set_xlim3d([-radius, radius])
    ax.set_zlim3d([-radius, radius])
    ax.set_ylim3d([-radius, radius])

  if add_labels:
    ax.set_zlabel("z")
  else:
    ax.set_zticks([])
    ax.set_zticklabels([])

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 0.1, 0.0)
  ax.w_xaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
  ax.w_xaxis.line.set_color(white)
  ax.w_yaxis.line.set_color(white)
  ax.w_zaxis.line.set_color(white)

  ax.set_aspect(1)

  # ax.invert_zaxis()


def _save_for_report(fig, name, do_close=True):
  fig.tight_layout(pad=0)
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
  plt.savefig('../img/{}.png'.format(name), pad_inches=0)
  plt.savefig('../img/{}.pgf'.format(name), pad_inches=0)

  if do_close:
    plt.close()

  eprint("Wrote ./img/{}.{{png,pgf}}.".format(name))
  eprint("If your plot includes images, don't forget to update its path in the .pgf file.")
  print(name)


class Arrow3D(FancyArrowPatch):
  def __init__(self, xs, ys, zs, *args, **kwargs):
    FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
    self._verts3d = xs, ys, zs

  def draw(self, renderer):
    xs3d, ys3d, zs3d = self._verts3d
    xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
    self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
    FancyArrowPatch.draw(self, renderer)
