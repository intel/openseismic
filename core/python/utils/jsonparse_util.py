#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import json


def get_params_string(filename):
    """
    A simple function that takes in a configuration file path ``file_path``
    and parses it to convert the keys to command line tags and the values to
    their respective command line tag values.

    :param file_path: Path to JSON file.
    :return: String representing arguments in a command line.
    """
    with open(filename) as f:
        data = json.load(f)
        return " ".join([f"-{k} {data[k]}" for k in data.keys()])
