#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import os
import math
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


def infer_section_sync(arg_obj, logger, get_functions):
    """
    Infer on section data synchronously. In order to use this in your
    configuration, specify ``infer_type`` as ``section_sync``. Section inference
    requires that additional parameters such as ``slice``, ``subsampl``, and
    ``slice_no`` be specified in the JSON configuration file. This function's
    specific arguments will be filled according to your configuration inputs.

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

    slice_type = arg_obj.slice
    subsampl = 1  # arg_obj.subsampl
    im_size = arg_obj.im_size
    slice_no = arg_obj.slice_no
    return_full_size = arg_obj.return_to_fullsize
    sep = os.path.sep

    data_arr = []
    assert (os.path.isdir(arg_obj.data) or os.path.isfile(
        arg_obj.data)), "[ERROR] Unexpected data input."
    if os.path.isdir(arg_obj.data):
        for data_file_name in os.listdir(arg_obj.data):
            path_to_file = arg_obj.data + sep + data_file_name
            data, data_info = loader(path_to_file)
            data_arr.append(
                {'name': data_file_name, 'data': data, 'data_info': data_info})
    if os.path.isfile(arg_obj.data):
        data, data_info = loader(arg_obj.data)
        data_arr.append({'name': arg_obj.data.replace(
            "/", "-"), 'data': data, 'data_info': data_info})

    logger.infer('Conducting inference...')
    def ls(N): return np.linspace(0, N - 1, N, dtype='int')
    for data_dict in data_arr:
        input_name = data_dict['name']
        data = data_dict['data']
        data_info = data_dict['data_info']

        logger.infer('Conducting inference on input: {}...'.format(input_name))
        logger.infer(
            'Inference Config - Slice Type {} on Slice No. {}...'.format(slice_type, slice_no))

        N0, N1, N2 = data.shape
        x0_range = ls(N0)
        x1_range = ls(N1)
        x2_range = ls(N2)

        check_slice_type = slice_type == 'inline' or slice_type == 'crossline' or slice_type == 'timeslice'
        assert check_slice_type, "[ERROR] Invalid slice_type: {}".format(
            slice_type)

        if slice_type == 'inline':
            slice_no = slice_no - data_info['inline_start']
            class_cube = data[::subsampl, 0:1, ::subsampl] * 0
            x1_range = np.array([slice_no])

        elif slice_type == 'crossline':
            slice_no = slice_no - data_info['crossline_start']
            class_cube = data[::subsampl, ::subsampl, 0:1, ] * 0
            x2_range = np.array([slice_no])

        elif slice_type == 'timeslice':
            slice_no = slice_no - data_info['timeslice_start']
            class_cube = data[0:1, ::subsampl, ::subsampl] * 0
            x0_range = np.array([slice_no])

        assert slice_no > - \
            1, "[ERROR] Invalid slice_no. For {}, refer to: {}".format(
                input_name, data_info)

        n0, n1, n2 = class_cube.shape
        # x0_grid, x1_grid, x2_grid = np.meshgrid(ls(n0,), ls(n1), ls(n2), indexing='ij')
        X0_grid, X1_grid, X2_grid = np.meshgrid(
            x0_range, x1_range, x2_range, indexing='ij')
        X0_grid_sub = X0_grid[::subsampl, ::subsampl, ::subsampl]
        X1_grid_sub = X1_grid[::subsampl, ::subsampl, ::subsampl]
        X2_grid_sub = X2_grid[::subsampl, ::subsampl, ::subsampl]

        w = in_shape[2] // 2
        h = in_shape[3] // 2

        # Decide iterator axis
        # X0_grid_sub.size / w * w
        iter_axis = range(
            math.ceil(n1 / in_shape[2]) * math.ceil(n2 / in_shape[2]))
        if slice_type == 'inline':
            # X1_grid_sub.size / w * w
            iter_axis = range(
                math.ceil(n0 / in_shape[2]) * math.ceil(n2 / in_shape[2]))
        if slice_type == 'crossline':
            # X2_grid_sub.size / w * w
            iter_axis = range(
                math.ceil(n0 / in_shape[2]) * math.ceil(n1 / in_shape[2]))

        iter_axis = tqdm(iter_axis)

        next_X0_1, next_X0_2 = 0, in_shape[2]
        next_X1_1, next_X1_2 = 0, in_shape[2]
        next_X2_1, next_X2_2 = 0, in_shape[2]

        for i in iter_axis:
            X0 = X0_grid_sub.ravel()[
                i] if slice_type == 'timeslice' else next_X0_1 + w
            X1 = X1_grid_sub.ravel()[
                i] if slice_type == 'inline' else next_X1_1 + w
            X2 = X2_grid_sub.ravel()[
                i] if slice_type == 'crossline' else next_X2_1 + w

            mini_sheet = np.zeros(in_shape)
            found_mini_sheet = False
            if slice_type == 'inline' and next_X0_1 == X0 - w and next_X2_1 == X2 - w:
                # X1 out
                end_X0 = min(next_X0_2, X0 + w + 1)
                end_X2 = min(next_X2_2, X2 + w + 1)

                mini_sheet = data[X0-w:end_X0, X1, X2-w:end_X2]
                mini_sheet = mini_sheet[np.newaxis, np.newaxis, :, :]
                found_mini_sheet = True

            if slice_type == 'crossline' and next_X0_1 == X0 - w and next_X1_1 == X1 - w:
                # X2 out
                end_X0 = min(next_X0_2, X0 + w + 1)
                end_X1 = min(next_X1_2, X1 + w + 1)

                mini_sheet = data[X0-w: end_X0, X1-w: end_X1, X2]
                mini_sheet = mini_sheet[np.newaxis, np.newaxis, :, :]
                found_mini_sheet = True

            if slice_type == 'timeslice' and next_X1_1 == X1 - w and next_X2_1 == X2 - w:
                # X0 out
                end_X1 = min(next_X1_2, X1 + w + 1)
                end_X2 = min(next_X2_2, X2 + w + 1)

                mini_sheet = data[X0, X1-w: end_X1, X2-w: end_X2]
                mini_sheet = mini_sheet[np.newaxis, np.newaxis, :, :]
                found_mini_sheet = True

            if found_mini_sheet:
                orig_shape = mini_sheet.shape
                input_dict = preprocess(mini_sheet, model.get_inputs(), model)

                output_dict, latency = model.infer(input_dict)
                output_dict = postprocess(output_dict, orig_shape)
                out = output_dict[list(output_dict.keys())[0]]
                out = np.squeeze(out)

                if slice_type == 'inline':
                    # X1 out
                    class_cube[X0 - w: X0 + w + 1, 0, X2 - w: X2 + w + 1] = out

                    if next_X2_2 == n2:
                        next_X0_1, next_X0_2 = X0 + w, min(X0 + 3*w + 1, n0)
                        next_X2_1, next_X2_2 = 0, in_shape[2]
                    else:
                        next_X2_1, next_X2_2 = X2 + w, min(X2 + 3*w + 1, n2)

                if slice_type == 'crossline':
                    # X2 out
                    class_cube[X0 - w: X0 + w + 1, X1 - w: X1 + w + 1, 0] = out

                    if next_X1_2 == n1:
                        next_X0_1, next_X0_2 = X0 + w, min(X0 + 3*w + 1, n0)
                        next_X1_1, next_X1_2 = 0, in_shape[2]
                    else:
                        next_X1_1, next_X1_2 = X1 + w, min(X1 + 3*w + 1, n1)

                if slice_type == 'timeslice':
                    # X0 out
                    class_cube[0, X1 - w: X1 + w + 1, X2 - w: X2 + w + 1] = out

                    if next_X2_2 == n2:
                        next_X1_1, next_X1_2 = X1 + w, min(X1 + 3*w + 1, n1)
                        next_X2_1, next_X2_2 = 0, in_shape[2]
                    else:
                        next_X2_1, next_X2_2 = X2 + w, min(X2 + 3*w + 1, n2)

        input_ref = input_name + "-input"
        save_path = output_folder + sep + input_ref
        logger.infer('Saving output to output path: {}'.format(
            save_path + sep + "out.npy"
        ))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        np.save(save_path + sep + "out", class_cube)
        logger.infer('Complete!')


def infer_section_async(arg_obj, logger, get_functions):
    """
    Infer on section data asynchronously. In order to use this in your
    configuration, specify ``infer_type`` as ``section_async``. Section inference
    requires that additional parameters such as ``slice``, ``subsampl``, and
    ``slice_no`` be specified in the JSON configuration file. This function's
    specific arguments will be filled according to your configuration inputs.

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
    logger.infer('Setting up inference queues and requests...')
    preprocess, postprocess, model = get_functions(
        arg_obj.model, arg_obj.given_model_name, arg_obj.infer_type, arg_obj.streams)
    in_shape = model.get_input_shape()
    logger.infer('Using model: {}'.format(model.name))

    # Expects one input and one output layer
    assert (len(model.get_inputs()) <
            2), "[ERROR] Expects model with one input layer."
    assert (len(model.get_outputs()) <
            2), "[ERROR] Expects model with one output layer."

    slice_type = arg_obj.slice
    subsampl = 1  # arg_obj.subsampl
    im_size = arg_obj.im_size
    slice_no = arg_obj.slice_no
    return_full_size = arg_obj.return_to_fullsize
    sep = os.path.sep

    data_arr = []
    assert (os.path.isdir(arg_obj.data) or os.path.isfile(
        arg_obj.data)), "[ERROR] Unexpected data input."
    if os.path.isdir(arg_obj.data):
        for data_file_name in os.listdir(arg_obj.data):
            path_to_file = arg_obj.data + sep + data_file_name
            data, data_info = loader(path_to_file)
            data_arr.append({'name': data_file_name, 'data': data})
    if os.path.isfile(arg_obj.data):
        data, data_info = loader(arg_obj.data)
        data_arr.append({'name': arg_obj.data.replace("/", "-"), 'data': data})

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
        orig_shape = param_dict['orig_shape']
        slice_type = param_dict['slice_type']
        i = param_dict['order']

        output_blobs = request.output_blobs
        out_layer = list(output_blobs.keys())[0]
        output_dict = {out_layer: output_blobs[out_layer].buffer}

        output_dict = postprocess(output_dict, orig_shape)

        out = output_dict[list(output_dict.keys())[0]]
        out = np.squeeze(out)
        if slice_type == 'inline':
            out = out[:, np.newaxis, :]
        if slice_type == 'crossline':
            out = out[:, :, np.newaxis]
        if slice_type == 'timeslice':
            out = out[np.newaxis, :, :]

        order_dict[i] = {
            'x0x1x2': param_dict['x0x1x2'], 'out': out
        }

        return out

    requests = model.get_requests()
    request_queue = InferRequestsQueue(requests, async_callback, postprocess)

    logger.infer('Conducting inference...')
    def ls(N): return np.linspace(0, N - 1, N, dtype='int')
    for data_dict in data_arr:
        input_name = data_dict['name']
        data = data_dict['data']

        logger.infer('Conducting inference on input: {}...'.format(input_name))

        N0, N1, N2 = data.shape
        x0_range = ls(N0)
        x1_range = ls(N1)
        x2_range = ls(N2)
        pred_points = (x0_range[::subsampl],
                       x1_range[::subsampl], x2_range[::subsampl])

        check_slice_type = slice_type == 'inline' or slice_type == 'crossline' or slice_type == 'timeslice'
        assert check_slice_type, "[ERROR] Invalid slice_type: {}".format(
            slice_type)

        if slice_type == 'inline':
            slice_no = slice_no - data_info['inline_start']
            class_cube = data[::subsampl, 0:1, ::subsampl] * 0
            x1_range = np.array([slice_no])

        elif slice_type == 'crossline':
            slice_no = slice_no - data_info['crossline_start']
            class_cube = data[::subsampl, ::subsampl, 0:1, ] * 0
            x2_range = np.array([slice_no])

        elif slice_type == 'timeslice':
            slice_no = slice_no - data_info['timeslice_start']
            class_cube = data[0:1, ::subsampl, ::subsampl] * 0
            x0_range = np.array([slice_no])

        assert slice_no > - \
            1, f"[ERROR] Invalid slice_no. For {input_name}, refer to: {data_info}"

        n0, n1, n2 = class_cube.shape
        x0_grid, x1_grid, x2_grid = np.meshgrid(
            ls(n0,), ls(n1), ls(n2), indexing='ij')
        X0_grid, X1_grid, X2_grid = np.meshgrid(
            x0_range, x1_range, x2_range, indexing='ij')
        X0_grid_sub = X0_grid[::subsampl, ::subsampl, ::subsampl]
        X1_grid_sub = X1_grid[::subsampl, ::subsampl, ::subsampl]
        X2_grid_sub = X2_grid[::subsampl, ::subsampl, ::subsampl]

        w = in_shape[2] // 2
        h = in_shape[3] // 2
        order_dict = {}

        # Decide iterator axis
        # X0_grid_sub.size / w * w
        iter_axis = range(
            math.ceil(n1 / in_shape[2]) * math.ceil(n2 / in_shape[2]))
        if slice_type == 'inline':
            # X1_grid_sub.size / w * w
            iter_axis = range(
                math.ceil(n0 / in_shape[2]) * math.ceil(n2 / in_shape[2]))
        if slice_type == 'crossline':
            # X2_grid_sub.size / w * w
            iter_axis = range(
                math.ceil(n0 / in_shape[2]) * math.ceil(n1 / in_shape[2]))

        iter_axis = tqdm(iter_axis)

        next_X0_1, next_X0_2 = 0, in_shape[2]
        next_X1_1, next_X1_2 = 0, in_shape[2]
        next_X2_1, next_X2_2 = 0, in_shape[2]

        for i in iter_axis:
            X0 = X0_grid_sub.ravel()[
                i] if slice_type == 'timeslice' else next_X0_1 + w
            X1 = X1_grid_sub.ravel()[
                i] if slice_type == 'inline' else next_X1_1 + w
            X2 = X2_grid_sub.ravel()[
                i] if slice_type == 'crossline' else next_X2_1 + w

            mini_sheet = np.zeros(in_shape)
            found_mini_sheet = False
            end_X0 = end_X1 = end_X2 = 1
            if slice_type == 'inline' and next_X0_1 == X0 - w and next_X2_1 == X2 - w:
                # X1 out
                end_X0 = min(next_X0_2, X0 + w + 1)
                end_X2 = min(next_X2_2, X2 + w + 1)

                mini_sheet = data[X0-w:end_X0, X1, X2-w:end_X2]
                mini_sheet = mini_sheet[np.newaxis, np.newaxis, :, :]
                found_mini_sheet = True

            if slice_type == 'crossline' and next_X0_1 == X0 - w and next_X1_1 == X1 - w:
                # X2 out
                end_X0 = min(next_X0_2, X0 + w + 1)
                end_X1 = min(next_X1_2, X1 + w + 1)

                mini_sheet = data[X0-w:end_X0, X1-w:end_X1, X2]
                mini_sheet = mini_sheet[np.newaxis, np.newaxis, :, :]
                found_mini_sheet = True

            if slice_type == 'timeslice' and next_X1_1 == X1 - w and next_X2_1 == X2 - w:
                # X0 out
                end_X1 = min(next_X1_2, X1 + w + 1)
                end_X2 = min(next_X2_2, X2 + w + 1)

                mini_sheet = data[X0, X1-w: end_X1, X2-w: end_X2]
                mini_sheet = mini_sheet[np.newaxis, np.newaxis, :, :]
                found_mini_sheet = True

            if found_mini_sheet:
                orig_shape = mini_sheet.shape
                input_dict = preprocess(mini_sheet, model.get_inputs(), model)

                # Inference! input_dict => {output_layer: output_data}, latency
                infer_request = request_queue.get_idle_request()
                infer_request.start_async(input_dict, input_name, {
                    'x0x1x2': [
                        (0, 1) if slice_type == 'timeslice' else (X0 - w, end_X0),
                        (0, 1) if slice_type == 'inline' else (X1 - w, end_X1),
                        (0, 1) if slice_type == 'crossline' else (X2 - w, end_X2)
                    ], 'order': i, 'order_dict': order_dict, 'orig_shape': orig_shape,
                    'slice_type': slice_type
                })

                if slice_type == 'inline':
                    # X1 out

                    if next_X2_2 == n2:
                        next_X0_1, next_X0_2 = X0 + w, min(X0 + 3*w + 1, n0)
                        next_X2_1, next_X2_2 = 0, in_shape[2]
                    else:
                        next_X2_1, next_X2_2 = X2 + w, min(X2 + 3*w + 1, n2)

                if slice_type == 'crossline':
                    # X2 out

                    if next_X1_2 == n1:
                        next_X0_1, next_X0_2 = X0 + w, min(X0 + 3*w + 1, n0)
                        next_X1_1, next_X1_2 = 0, in_shape[2]
                    else:
                        next_X1_1, next_X1_2 = X1 + w, min(X1 + 3*w + 1, n1)

                if slice_type == 'timeslice':
                    # X0 out

                    if next_X2_2 == n2:
                        next_X1_1, next_X1_2 = X1 + w, min(X1 + 3*w + 1, n1)
                        next_X2_1, next_X2_2 = 0, in_shape[2]
                    else:
                        next_X2_1, next_X2_2 = X2 + w, min(X2 + 3*w + 1, n2)

        logger.infer('Cleaning up requests...')
        request_queue.wait_all()

        logger.infer('Placing prediction in proper cube spot...')
        available_keys = set(list(order_dict.keys()))

        # Decide iterator axis
        # X0_grid_sub.size / w * w
        iter_axis = range(
            math.ceil(n1 / in_shape[2]) * math.ceil(n2 / in_shape[2]))
        if slice_type == 'inline':
            # X1_grid_sub.size / w * w
            iter_axis = range(
                math.ceil(n0 / in_shape[2]) * math.ceil(n2 / in_shape[2]))
        if slice_type == 'crossline':
            # X2_grid_sub.size / w * w
            iter_axis = range(
                math.ceil(n0 / in_shape[2]) * math.ceil(n1 / in_shape[2]))

        iter_axis = tqdm(iter_axis)

        for i in iter_axis:
            if i in available_keys:
                out_w_param = order_dict[i]
                out = out_w_param['out']
                x0, x1, x2 = out_w_param['x0x1x2']
                x0_1, x0_2 = x0
                x1_1, x1_2 = x1
                x2_1, x2_2 = x2

                class_cube[x0_1:x0_2, x1_1:x1_2, x2_1:x2_2] = out

        input_ref = input_name + "-input"
        save_path = output_folder + sep + input_ref
        logger.infer('Saving output to output path: {}'.format(
            save_path + sep + "out.npy"))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        np.save(save_path + sep + "out", class_cube)
        logger.infer('Complete!')
