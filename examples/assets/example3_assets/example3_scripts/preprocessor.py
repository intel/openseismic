#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import numpy as np
import segyio


def preprocess(cube, input_layers):
    """
    Takes the cube input and stores it in a dictionary where each key represents a layer name.
    For clarity, the structure will be ``{..., input_layer_name_i:cube, ...}``. For the Salt
    model, there is only one layer, so the output will be ``{first_input_layer_name:cube}``.

    :param cube: The original data input.
    :param input_layers: A list of input layers. The Salt model will only contain one layer.
    :return: n input dictionary structured as ``{..., input_layer_name_i:cube, ...}``. In this case, the returned value will be ``{first_input_layer_name:cube}``.
    """

    """
    Params: 
    cube - 3d array with seismic data 
    input_layers - a list of the input layers (fseg should only have one input)
    
    Return:
    input_dict - dictionary mapping transformed inputs to proper input layers
    """
    input_dict = {}
    for input_layer in input_layers:
        input_dict[input_layer] = cube
    return input_dict
