#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import os
import shutil
import warnings
import itertools
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from scipy.interpolate import interpn
from utils.infer_util import InferRequestsQueue, loader

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

OUTPUT, SWAP, INFER, ARG = 5, 6, 7, 8

def infer_coarse_cubed_sync(arg_obj, logger, get_functions):
    """
    Infer on 3D seismic data synchronously with coarse cube output.
    In order to use this in your configuration, specify ``infer_type`` as
    ``coarse_cube_sync``. Fine cubed inference requires that additional
    parameters such as ``im_size`` and ``window`` be specified in the JSON
    configuration file. This function's specific arguments will be filled
    according to your configuration inputs.

    :param arg_obj: Arguments object that holds parameters needed for inference.
    :param logger: Common logger object for logging coherence.
    :param get_functions: Functions associated with parameters given.
    :return: None
    """
    logger.setLevel(OUTPUT)
    output_folder = arg_obj.output
    if not os.path.exists(output_folder):
        logger.output('Making new folder for output storage.')
        os.mkdir(output_folder)
    else:
        logger.output('Output folder already exists. Deleting...')
        shutil.rmtree(output_folder)
        logger.output('Making new folder for output storage.')
        os.mkdir(output_folder)

    logger.setLevel(INFER)
    logger.infer('Setting up inference...')
    preprocess, postprocess, model = get_functions(
        arg_obj.model, arg_obj.given_model_name)
    in_shape = model.get_input_shape()
    logger.infer('Using model: {}'.format(model.name))

    # Expects one input and one output layer
    assert (len(model.get_inputs()) <
            2), "[ERROR] Expects model with one input layer."
    assert (len(model.get_outputs()) <
            2), "[ERROR] Expects model with one output layer."
    
    # Getting relevant argument variables
    im_size = arg_obj.im_size
    wd_list = arg_obj.window
    sep = os.path.sep
    
    # Get data and store in data_arr
    data_arr = []
    assert (os.path.isdir(arg_obj.data) or os.path.isfile(
        arg_obj.data)), "[ERROR] Unexpected data input."
    if os.path.isdir(arg_obj.data):
        for i, data_file_name in enumerate(os.listdir(arg_obj.data)):
            path_to_file = arg_obj.data + sep + data_file_name
            data, data_info = loader(path_to_file)
            wd_index = i if len(os.listdir(arg_obj.data)) == len(wd_list) else 0
            wd_i = []
            if len(wd_list) > wd_index:
                wd_i = wd_list[wd_index]
            data_arr.append({'name': data_file_name, 'data': data, 'window': wd_i})
    if os.path.isfile(arg_obj.data):
        data, data_info = loader(arg_obj.data)
        wd_i = []
        if len(wd_list) > 0:
            wd_i = wd_list[0]
        data_arr.append({
            'name': arg_obj.data.replace(sep, "-").replace(".", "-"), 
            'data': data, 'window': wd_i
        })
    
    logger.infer('Conducting inference...')
    for data_dict in data_arr:
        input_name = data_dict['name']
        data = data_dict['data']
        assert (len(data.shape) == 3
            ), f"[ERROR] Expects 3D input. Dimension of {input_name} is {len(data.shape)}"
        start_i, finish_i = 0, data.shape[0]
        start_j, finish_j = 0, data.shape[1]
        start_k, finish_k = 0, data.shape[2]
        
        # Using window, if available
        window = data_dict['window']
        if len(window) == 3:
            start_i, finish_i = window[0]
            start_j, finish_j = window[1]
            start_k, finish_k = window[2]
            
        # Setting step size
        i_len = j_len = k_len = im_size
        
        # Checking input size
        try:
            if list(in_shape) != [im_size, im_size, im_size]:
                logger.infer('Reshaping inference input...')
                model.reshape_input([im_size, im_size, im_size])
                in_shape = model.get_input_shape()
                logger.infer(f'Reshaped inference input to: {in_shape}')
        except:
            assert (len(data.shape) == 3
                ), f"[ERROR] Cannot reshape input to size: {data.shape}"
        
        logger.infer('Conducting inference on input: {}...'.format(input_name))
        
        copy_cube = np.zeros(data.shape)
        i_range = range(start_i, finish_i, i_len)
        j_range = range(start_j, finish_j, j_len)
        k_range = range(start_k, finish_k, k_len)
        for i, j, k in tqdm(itertools.product(i_range, j_range, k_range)):
            begin_i, end_i = i, i + i_len if i + i_len < finish_i else finish_i - 1
            begin_j, end_j = j, j + j_len if j + j_len < finish_j else finish_j - 1
            begin_k, end_k = k, k + k_len if k + k_len < finish_k else finish_k - 1
            mini_cube = data[begin_i:end_i, begin_j:end_j, begin_k:end_k]

            input_dict = preprocess(mini_cube, model.get_inputs(), model=model)
            output_dict, latency = model.infer(input_dict)
            output_dict = postprocess(output_dict, output_shape=mini_cube.shape)
            out_data = output_dict[next(iter(output_dict.keys()))]
            copy_cube[begin_i:end_i, begin_j:end_j, begin_k:end_k] = out_data
        
        input_ref = input_name + "-input"
        save_path = output_folder + sep + input_ref
        logger.infer('Saving output to output path: {}'.format(
            save_path + sep + "out.npy"
        ))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        np.save(save_path + sep + "out", copy_cube)
        logger.infer('Complete!')

def infer_coarse_cubed_async(arg_obj, logger, get_functions):
    """
    Infer on 3D seismic data asynchronously with coarse cube output.
    In order to use this in your configuration, specify ``infer_type`` as
    ``coarse_cube_async``. Fine cubed inference requires that additional
    parameters such as ``im_size`` and ``window`` be specified in the JSON
    configuration file. This function's specific arguments will be filled
    according to your configuration inputs.

    :param arg_obj: Arguments object that holds parameters needed for inference.
    :param logger: Common logger object for logging coherence.
    :param get_functions: Functions associated with parameters given.
    :return: None
    """
    logger.setLevel(OUTPUT)
    output_folder = arg_obj.output
    if not os.path.exists(output_folder):
        logger.output('Making new folder for output storage.')
        os.mkdir(output_folder)
    else:
        logger.output('Output folder already exists. Deleting...')
        shutil.rmtree(output_folder)
        logger.output('Making new folder for output storage.')
        os.mkdir(output_folder)

    logger.setLevel(INFER)
    logger.infer('Setting up inference...')
    preprocess, postprocess, model = get_functions(
        arg_obj.model, arg_obj.given_model_name)
    logger.infer('Using model: {}'.format(model.name))

    # Expects one input and one output layer
    assert (len(model.get_inputs()) <
            2), "[ERROR] Expects model with one input layer."
    assert (len(model.get_outputs()) <
            2), "[ERROR] Expects model with one output layer."
    
    # Getting relevant argument variables
    im_size = arg_obj.im_size
    wd_list = arg_obj.window
    sep = os.path.sep
    
    # Get data and store in data_arr
    data_arr = []
    assert (os.path.isdir(arg_obj.data) or os.path.isfile(arg_obj.data)
           ), "[ERROR] Unexpected data input."
    if os.path.isdir(arg_obj.data):
        for i, data_file_name in enumerate(os.listdir(arg_obj.data)):
            path_to_file = arg_obj.data + sep + data_file_name
            data, data_info = loader(path_to_file)
            wd_index = i if len(os.listdir(arg_obj.data)) == len(wd_list) else 0
            wd_i = []
            if len(wd_list) > wd_index:
                wd_i = wd_list[wd_index]
            data_arr.append({'name': data_file_name, 'data': data, 'window': wd_i})
    if os.path.isfile(arg_obj.data):
        data, data_info = loader(arg_obj.data)
        wd_i = []
        if len(wd_list) > 0:
            wd_i = wd_list[0]
        data_arr.append({
            'name': arg_obj.data.replace(sep, "-").replace(".", "-"), 
            'data': data, 'window': wd_i
        })
        
        
    def async_callback(param_dict):
        """
        Params:
        param_dict - dictionary which holds:
            (1) request
            (2) postprocess
            (3) file_name
        """
        request = param_dict['request']
        postprocess = param_dict['postprocess']
        order_dict = param_dict['order_dict']
        i = param_dict['order']
        output_dict = request.outputs
        
        output_dict = postprocess(output_dict, output_shape=mini_cube.shape)
        out_data = output_dict[next(iter(output_dict.keys()))]
        copy_cube[begin_i:end_i, begin_j:end_j, begin_k:end_k] = out_data
        
        order_dict[i] = {
            'x0x1x2': param_dict['x0x1x2'], 'out': out
        }
        
        return out
    
    requests = model.get_requests()
    request_queue = InferRequestsQueue(requests, async_callback, postprocess)
    
    logger.infer('Conducting inference...')
    for data_dict in data_arr:
        input_name = data_dict['name']
        data = data_dict['data']
        assert (len(data.shape) == 3
            ), f"[ERROR] Expects 3D input. Dimension of {input_name} is {len(data.shape)}"
        start_i, finish_i = 0, data.shape[0]
        start_j, finish_j = 0, data.shape[1]
        start_k, finish_k = 0, data.shape[2]
        
        # Using window, if available
        window = data_dict['window']
        if len(window) == 3:
            start_i, finish_i = window[0]
            start_j, finish_j = window[1]
            start_k, finish_k = window[2]
            
        # Setting step size
        i_len = j_len = k_len = im_size
        
        # Checking input size
        try:
            if list(in_shape) != [im_size, im_size, im_size]:
                logger.infer('Reshaping inference input...')
                model.reshape_input([im_size, im_size, im_size])
                in_shape = model.get_input_shape()
                logger.infer(f'Reshaped inference input to: {in_shape}')
        except:
            assert (len(data.shape) == 3
                ), f"[ERROR] Cannot reshape input to size: {data.shape}"
        
        logger.infer('Conducting inference on input: {}...'.format(input_name))
        
        copy_cube = np.zeros(data.shape)
        i_range = range(start_i, finish_i, i_len)
        j_range = range(start_j, finish_j, j_len)
        k_range = range(start_k, finish_k, k_len)
        for i, j, k in tqdm(itertools.product(i_range, j_range, k_range)):
            begin_i, end_i = i, i + i_len if i + i_len < finish_i else finish_i - 1
            begin_j, end_j = j, j + j_len if j + j_len < finish_j else finish_j - 1
            begin_k, end_k = k, k + k_len if k + k_len < finish_k else finish_k - 1
            mini_cube = data[begin_i:end_i, begin_j:end_j, begin_k:end_k]

            input_dict = preprocess(mini_cube, model.get_inputs(), model=model)
            
            # Inference! input_dict => {output_layer: output_data}, latency
            infer_request = request_queue.get_idle_request()
            infer_request.start_async(input_dict, input_name, {
                'x0x1x2': ((begin_i, end_i), (begin_j, end_j), (begin_k, end_k)), 
                'order': i, 'order_dict': order_dict
            })
            
            output_dict, latency = model.infer(input_dict)
            output_dict = postprocess(output_dict, output_shape=mini_cube.shape)
            out_data = output_dict[next(iter(output_dict.keys()))]
            copy_cube[begin_i:end_i, begin_j:end_j, begin_k:end_k] = out_data
        
        input_ref = input_name + "-input"
        save_path = output_folder + sep + input_ref
        logger.infer('Saving output to output path: {}'.format(
            save_path + sep + "out.npy"
        ))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        np.save(save_path + sep + "out", copy_cube)
        logger.infer('Complete!')
