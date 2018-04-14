import logging

import jsonlines

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ClipWriter():

    def __init__(self, path='clips.jsonl'):
        self.path = path
        self.writer = jsonlines.open(path, mode='a')

    def send(self, clip):
        logger.info('Writing clip {} to {}'.format(clip['id'], self.path))
        self.writer.write(clip)

    def close(self):
        self.writer.close()
