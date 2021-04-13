#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import numpy as np


def preprocess(data, input_layers, n1n2n3=(128, 128, 128), model=None):
    """
    The preprocessing scripts follows a usual centering and normalizing of the input,
    followed by a transpose and addition of superfluous axes for inference purposes.
    If the input values are not the same shape as the FaultSeg input layer, then they
    will be inserted into a zero-tensor that is the same shape as the input layer.
    Therefore, we must guarantee that the input values are smaller than or equal to
    the input layer shape.

    :param data: The original data input.
    :param input_layers: A list of input layers. The FaultSeg model will only contain one layer.
    :param n1n2n3: (Optional) The expected shape to mold flat input values.
    :param model: (Optional) The model used for inference.
    :return: An input dictionary structured as ``{..., input_layer_name_i:input_val_i, ...}``. In this case, the returned value will be ``{first_input_layer_name:input_val}``.
    """

    n1, n2, n3 = n1n2n3
    gx = np.reshape(data, (n3, n2, n1)) if len(data.shape) == 1 else data
    gm = np.mean(gx)
    gs = np.std(gx)
    gx = gx-gm
    gx = gx/gs
    gx = np.transpose(gx)
    gx = gx[np.newaxis, np.newaxis, :, :, :]
    
    # Storing data in appropriate layers
    input_dict = {}
    for input_layer in input_layers:
        # Check gx shape
        edited_shape = False
        if model != None:
            input_shape = model.get_input_shape_dict()[input_layer]
            if list(input_shape) != list(gx.shape):
                orig_shape = gx.shape
                input_cube = np.zeros(input_shape)
                input_cube[0, 0, :orig_shape[2], :orig_shape[3], :orig_shape[4]] = gx
                input_dict[input_layer] = input_cube
                edited_shape = True
            
        if not edited_shape:
            input_dict[input_layer] = gx

    return input_dict
