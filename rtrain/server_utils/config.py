#!/usr/bin/env python3
"""Configuration parser for rtraind."""

import configparser


class RTrainConfig(object):
    def __init__(self, data):
        self.config = configparser.ConfigParser()
        self.config.read_string(data)
        if 'rtraind' not in self.config:
            self.config['rtraind'] = {}

    @property
    def db_string(self):
        return self.config['rtraind'].get('Database', 'sqlite:///:memory:')

    @property
    def password(self):
        return self.config['rtraind'].get('Password', '')
