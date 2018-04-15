import argparse
import logging

import jsonlines

import config_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_CLIPS_PATH = 'clips.jsonl'
DEFAULT_CONFIG_PATH = 'config.json'


class ClipWriter():

    def __init__(self, path=DEFAULT_CLIPS_PATH):
        self.path = path
        self.writer = jsonlines.open(path, mode='a')

    def send(self, clip):
        logger.debug('Writing clip {} to {}'.format(clip['id'], self.path))
        self.writer.write(clip)

    def close(self):
        self.writer.close()


def get_clip_ids(path=DEFAULT_CLIPS_PATH):
    with jsonlines.open(path, mode='r') as reader:
        return list(map(lambda c: c['id'], reader))


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manipulate clip data files.')
    parser.add_argument('command', metavar='command', type=str, nargs=1,
                        help='The command to execute',
                        choices=[
                            'config_to_clips',
                            'remove_duplicate_clips',
                            'add_clips_to', 'merge'
                        ])
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
    else:
        logger.error("Command {} not found.".format(command_name))
