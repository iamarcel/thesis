import argparse
import logging
import os
import shutil

import jsonlines

from common import config_utils, openpose_utils

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_CLIPS_PATH = 'clips.jsonl'
DEFAULT_CONFIG_PATH = 'config.json'
DEFAULT_IMAGES_PATH = 'images/'


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
    all_filenames = list(openpose_utils.get_outputs())
    with jsonlines.open(clips_path, 'r') as reader:
        for clip in reader:
            n_clips_all += 1

            filenames = openpose_utils.get_clip_files(
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


if __name__ == '__main__':
    command_choices = [
        'config_to_clips',
        'remove_duplicate_clips',
        'add_clips_to', 'merge',
        'move_2d_finished_images'
    ]

    parser = argparse.ArgumentParser(description='Manipulate clip data files.')
    parser.add_argument('command', metavar='command', type=str, nargs=1,
                        help="The command to execute. One of {}"
                        .format(command_choices),
                        choices=command_choices)
    parser.add_argument('args', metavar='args', type=str, nargs='*',
                        help='Arguments for the command')

    args = parser.parse_args()

    command_name = args.command[0]
    if command_name == 'config_to_clips':
        config_to_clips(*args.args)
    elif command_name == 'remove_duplicate_clips':
        remove_duplicate_clips(*args.args)
    elif command_name == 'add_clips_to':
        add_clips_to(*args.args)
    elif command_name == 'merge':
        add_clips_to(*args.args)
    elif command_name == 'move_2d_finished_images':
        move_2d_finished_images(*args.args)
    else:
        logger.error("Command {} not found.".format(command_name))
