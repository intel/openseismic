#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import numpy as np
import segyio


def preprocess(sheet, input_layers, model):
    """
    This preprocessing function takes the section input ``sheet`` and crops it
    to match the Facies input layer shape, if they are different. Then, an
    input dictionary is returned where each input layer is mapped to the sheet
    data. For the Facies model, there is only one input layer.

    :param sheet: Section input data.
    :param input_layers: A list of input layer names.
    :param model: The model object that is using this preprocessing function.
    :return: An input dictionary structured as ``{..., input_layer_i:input, ...}``. Since there is only one layer in the Facies model, the returned dictionary will be structured as ``{input_layer:input}``.
    """

    """
    Params: 
    sheet - seismic data
    input_layers - a list of the input layers (fseg should only have one input)
    model - Model's class object
    
    Return:
    input_dict - dictionary mapping transformed inputs to proper input layers
    """
    
    assert model != None, "[ERROR] Cannot accept 'None' for model."
    assert model.name == 'facies_model', "[ERROR] Cannot use model with preprocess."
    
    net = model.facies_net
    in_shape = net.input_info[list(net.input_info.keys())[0]].input_data.shape
    
    input_dict = {}
    for input_layer in input_layers:
        sheet_pad = np.zeros(shape=in_shape)
        sheet_pad[:, :, :sheet.shape[2], :sheet.shape[3]] = sheet
        input_dict[input_layer] = sheet_pad
    return input_dict
