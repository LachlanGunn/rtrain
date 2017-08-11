#!/usr/bin/env python3

import pkg_resources
import sqlite3
import sys
import rtrain.server_utils.database_operations as db

database_path = db.get_database_location()

if __name__ == '__main__':
    connection = sqlite3.connect(database_path)
    if connection is None:
        print("Could not connect to database.")
        sys.exit(1)

    query = str(pkg_resources.resource_string(__name__, "schema.sql"), 'utf-8')
    if query is None:
        print("Could not load database schema.")
        sys.exit(1)

    cursor = connection.cursor()
    cursor.executescript(query)
    cursor.close()
