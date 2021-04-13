#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import os
import shutil
import warnings
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from utils.infer_util import InferRequestsQueue, loader

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

OUTPUT, SWAP, INFER, ARG = 5, 6, 7, 8


def infer_sync(arg_obj, logger, get_functions):
    """
    Synchronously infer as normal. In order to use this in your configuration,
    specify ``infer_type`` as ``sync``.

    :param arg_obj: Arguments object that holds parameters needed for inference.
    :param logger: Common logger object for logging coherence.
    :param get_functions: Functions associated with parameters given.
    :return: None
    """
    sep = os.path.sep

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

    preprocess, postprocess, model = get_functions(
        arg_obj.model, arg_obj.given_model_name)

    logger.setLevel(INFER)
    logger.infer('Using model: {}'.format(model.name))
    logger.infer('Loading datafile names for inference...')

    data_arr = []
    assert (os.path.isdir(arg_obj.data) or os.path.isfile(
        arg_obj.data)), "[ERROR] Unexpected data input."
    if os.path.isdir(arg_obj.data):
        for data_file_name in os.listdir(arg_obj.data):
            path_to_file = arg_obj.data + sep + data_file_name
            data, data_info = loader(path_to_file)
            data_arr.append(
                {'name': data_file_name, 'path': path_to_file, 'data': data})
    if os.path.isfile(arg_obj.data):
        data, data_info = loader(arg_obj.data)
        data_arr.append({'name': arg_obj.data.replace(
            "/", "-"), 'path': arg_obj.data, 'data': data})

    logger.infer('Conducting inference...')
    for data_obj in tqdm(data_arr):
        data_file_name = data_obj['name']
        data_file_path = data_obj['path']
        data = data_obj['data']

        # Preprocess: data_file_name => {input_layer: input_data}
        inputs = model.get_inputs()
        input_dict = preprocess(data, inputs)

        # Inference! input_dict => {output_layer: output_data}, latency
        output_dict, latency = model.infer(input_dict)

        # Postprocess: output_dict => {output_layer: transformed_output_data}
        output_dict = postprocess(output_dict)

        # Writing Prediction: output_folder/input_file_name/layer/output
        for layer in output_dict.keys():
            output_of_layer = output_dict[layer]
            input_ref = data_file_name + "-input"
            save_path = output_folder + sep + \
                input_ref + sep + layer.replace('/', '-')

            if not os.path.exists(output_folder + sep + input_ref):
                os.mkdir(output_folder + sep + input_ref)
            os.mkdir(save_path)

            np.save(save_path + sep + "out", output_of_layer)
    logger.infer('Complete!')


def infer_async(arg_obj, logger, get_functions):
    """
    Asynchronously infer as normal. In order to use this in your configuration,
    specify ``infer_type`` as ``async``.

    :param arg_obj: Arguments object that holds parameters needed for inference.
    :param logger: Common logger object for logging coherence.
    :param get_functions: Functions associated with parameters given.
    :return:
    """
    sep = os.path.sep

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
    logger.infer('Loading datafile names for inference...')

    data_arr = []
    assert (os.path.isdir(arg_obj.data) or os.path.isfile(
        arg_obj.data)), "[ERROR] Unexpected data input."
    if os.path.isdir(arg_obj.data):
        for data_file_name in os.listdir(arg_obj.data):
            path_to_file = arg_obj.data + sep + data_file_name
            data, data_info = loader(path_to_file)
            data_arr.append(
                {'name': data_file_name, 'path': path_to_file, 'data': data})
    if os.path.isfile(arg_obj.data):
        data, data_info = loader(arg_obj.data)
        data_arr.append({'name': arg_obj.data.replace(
            "/", "-"), 'path': arg_obj.data, 'data': data})

    logger.infer('Setting up inference queues and requests...')
    preprocess, postprocess, model = get_functions(
        arg_obj.model, arg_obj.given_model_name, arg_obj.infer_type, arg_obj.streams)
    logger.infer('Using model: {}'.format(model.name))

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
        data_file_name = param_dict['file_name']
        output_names = param_dict['output_names']
        output_dict = postprocess({
            layer: request.output_blobs[layer].buffer for layer in output_names
        })

        for layer in output_names:
            output_of_layer = output_dict[layer]
            input_ref = data_file_name + "-input"
            save_path = output_folder + sep + \
                input_ref + sep + layer.replace('/', '-')

            if not os.path.exists(output_folder + sep + input_ref):
                os.mkdir(output_folder + sep + input_ref)
            os.mkdir(save_path)

            np.save(save_path + sep + "out", output_of_layer)

        return output_dict

    requests = model.get_requests()
    request_queue = InferRequestsQueue(requests, async_callback, postprocess)

    logger.infer('Conducting inference...')
    for data_obj in tqdm(data_arr):
        data_file_name = data_obj['name']
        data_file_path = data_obj['path']
        data = data_obj['data']

        # Preprocess: data_file_name => {input_layer: input_data}
        inputs = model.get_inputs()
        outputs = model.get_outputs()
        input_dict = preprocess(data, inputs)

        # Inference! input_dict => {output_layer: output_data}, latency
        infer_request = request_queue.get_idle_request()
        infer_request.start_async(input_dict, data_file_name, {
            'output_names': outputs
        })

    logger.infer('Cleaning up requests...')
    request_queue.wait_all()

    logger.infer('Complete!')
