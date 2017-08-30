#!/usr/bin/env python3
"""Database setup tool for rtraind."""

import argparse
import sys

import sqlalchemy

import rtrain.server_utils.config
import rtrain.server_utils.model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        default='/etc/rtraind.conf',
        help='Path to configuration file.')

    args = parser.parse_args()
    try:
        with open(args.config) as config_fh:
            config = rtrain.server_utils.config.RTrainConfig(config_fh.read())
    except FileNotFoundError:
        config = rtrain.server_utils.config.RTrainConfig('')
    except IOError:
        print("Failed to load config file.")
        sys.exit(1)

    engine = sqlalchemy.create_engine(config.db_string)
    rtrain.server_utils.model.Base.metadata.create_all(engine)


if __name__ == '__main__':
    main()
