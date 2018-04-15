import logging

import jsonlines

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_CLIPS_PATH = 'clips.jsonl'


class ClipWriter():

    def __init__(self, path=DEFAULT_CLIPS_PATH):
        self.path = path
        self.writer = jsonlines.open(path, mode='a')

    def send(self, clip):
        logger.info('Writing clip {} to {}'.format(clip['id'], self.path))
        self.writer.write(clip)

    def close(self):
        self.writer.close()


def get_clip_ids(path=DEFAULT_CLIPS_PATH):
    with jsonlines.open(path, mode='r') as reader:
        return list(map(lambda c: c['id'], reader))
