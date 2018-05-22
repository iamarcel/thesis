#!/usr/bin/env python3
import json


def load_config(path='config.json'):
    with open(path) as config_file:
        return json.load(config_file)


def load_3d_poses(config=None, path='config.json'):
    if config is None:
        config = load_config(path)

    return [c['points_3d'] for c in config['clips']]
