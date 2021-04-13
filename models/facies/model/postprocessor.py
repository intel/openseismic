#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import numpy as np
import torch
import torch.nn.functional as F


def postprocess(output_val, orig_shape):
    """
    This postprocess function first checks if the output contained within the
    ``output_val`` dictionary matches the specified ``orig_shape`` (a 4
    element list/tuple, but the last two elements are the ones that are
    checked). If the shapes are different, then the output is cropped to
    match the ``orig_shape``. Then, the output is ran through a softmax layer.
    The maximum probabilistic class is assigned to each pixel in the section,
    and this output is returned in the form ``{output_layer:prediction}``.

    :param output_val: The output value dictionary, structured as ``{output_layer:output}``.
    :param orig_shape: The original shape of the input data.
    :return: A dictionary mapping output layer to maximum class prediction per pixel.
    """

    """
    Params:
    output_val - dictionary mapping output_data to output_layers
    
    Return:
    outputs - dictionary mapping transformed_output_data to output_layers
    """
    ou_layer = list(output_val.keys())[0]
    output = output_val[ou_layer]

    if output.shape[2:] != orig_shape[2:]:
        output = output[:, :, :orig_shape[2], :orig_shape[3]]

    out = F.softmax(torch.from_numpy(output), dim=1)
    prediction = out.max(1)[1].cpu().numpy()[0]

    return {ou_layer: prediction}
