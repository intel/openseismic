#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import numpy as np


def postprocess(output_val, output_shape=None):
    """
    The postprocess script for the FaultSeg model squeezes the output to get rid of extraneous
    axes. Then, if the model's output shape is different than the original output shape specified
    in ``output_shape`` (if given), then the model output is cropped to match ``output_shape``.
    Therefore, we must guarantee that the model output will be greater than or equal to the
    original shape.

    :param output_val: An output dictionary structured as ``{..., output_layer_i:output_val, ...}``.
    :param output_shape: (Optional) Tuple/list representing the original output shape.
    :return: An output dictionary structured the same as ``output_val`` except contained cropped output values.
    """
    for layers in output_val.keys():
        output_val_i = output_val[layers][0, 0, :, :, :]
        if output_shape != None:
            # Expect output_val_i.shape = T of output_shape (original input shape)
            output_val_i = output_val_i[
                :output_shape[2], :output_shape[1], :output_shape[0]]
            output_val_i = np.transpose(output_val_i)
        
        output_val[layers] = output_val_i
    return output_val
