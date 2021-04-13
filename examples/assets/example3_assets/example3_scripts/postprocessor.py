#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import numpy as np


def postprocess(output_val):
    """
    This postprocess simply returns the input ``output_val``.

    :param output_val: dictionary mapping output_data to output_layers
    :return: ``output_val``
    """
    return output_val
