#!/usr/bin/env python3

import sqlalchemy
import sys

import pkg_resources

import rtrain.server_utils.model
import rtrain.server_utils.model.database_operations as db

database_path = db.get_database_location()
engine = sqlalchemy.create_engine(database_path)


def main():
    rtrain.server_utils.model.Base.metadata.create_all(engine)


if __name__ == '__main__':
    main()
