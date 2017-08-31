#!/usr/bin/env python3

import rtrain.server_utils.config


def make_config_file(root_name, database_name, password):
    return """[%s]
Database=%s
Password=%s""" % (root_name, database_name, password)


def test_config():
    config_string = make_config_file("rtraind", "some://database/name",
                                     "the password")
    config = rtrain.server_utils.config.RTrainConfig(config_string)
    assert config.db_string == "some://database/name"
    assert config.password == "the password"


def test_config_empty_has_no_password():
    config_string = make_config_file("foo", "some://database/name",
                                     "the password")
    config = rtrain.server_utils.config.RTrainConfig(config_string)
    assert config.password == ""
