#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import os
import shutil
import warnings
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from scipy.interpolate import interpn
from utils.infer_util import InferRequestsQueue, loader

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

OUTPUT, SWAP, INFER, ARG = 5, 6, 7, 8


def infer_fine_cubed_sync(arg_obj, logger, get_functions):
    """
    Infer on 3D seismic data synchronously with fine cube output.
    In order to use this in your configuration, specify ``infer_type`` as
    ``fine_cube_sync``. Fine cubed inference requires that additional
    parameters such as ``slice``, ``subsampl``, ``im_size``, ``slice_no``,
    and ``return_to_fullsize`` be specified in the JSON configuration file.
    This function's specific arguments will be filled according to your
    configuration inputs.

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

    slice_type = arg_obj.slice
    subsampl = arg_obj.subsampl
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

        if slice_type == 'full':
            class_cube = data[::subsampl, ::subsampl, ::subsampl] * 0

        elif slice_type == 'inline':
            slice_no = slice_no - data_info['inline_start']
            class_cube = data[::subsampl, 0:1, ::subsampl] * 0
            x1_range = np.array([slice_no])
            pred_points = (pred_points[0], pred_points[2])

        elif slice_type == 'crossline':
            slice_no = slice_no - data_info['crossline_start']
            class_cube = data[::subsampl, ::subsampl, 0:1, ] * 0
            x2_range = np.array([slice_no])
            pred_points = (pred_points[0], pred_points[1])

        elif slice_type == 'timeslice':
            slice_no = slice_no - data_info['timeslice_start']
            class_cube = data[0:1, ::subsampl, ::subsampl] * 0
            x0_range = np.array([slice_no])
            pred_points = (pred_points[1], pred_points[2])

        n0, n1, n2 = class_cube.shape
        x0_grid, x1_grid, x2_grid = np.meshgrid(
            ls(n0,), ls(n1), ls(n2), indexing='ij')
        X0_grid, X1_grid, X2_grid = np.meshgrid(
            x0_range, x1_range, x2_range, indexing='ij')
        X0_grid_sub = X0_grid[::subsampl, ::subsampl, ::subsampl]
        X1_grid_sub = X1_grid[::subsampl, ::subsampl, ::subsampl]
        X2_grid_sub = X2_grid[::subsampl, ::subsampl, ::subsampl]

        w = im_size//2
        for i in tqdm(range(X0_grid_sub.size)):
            x0 = x0_grid.ravel()[i]
            x1 = x1_grid.ravel()[i]
            x2 = x2_grid.ravel()[i]

            X0 = X0_grid_sub.ravel()[i]
            X1 = X1_grid_sub.ravel()[i]
            X2 = X2_grid_sub.ravel()[i]

            if X0 > w and X1 > w and X2 > w and X0 < N0 - w + 1 and X1 < N1 - w + 1 and X2 < N2 - w + 1:
                mini_cube = data[X0 - w: X0 + w + 1, X1 -
                                 w: X1 + w + 1, X2 - w: X2 + w + 1]
                mini_cube = mini_cube[np.newaxis, np.newaxis, :, :, :]

                input_dict = preprocess(mini_cube, model.get_inputs())
                output_dict, latency = model.infer(input_dict)
                output_dict = postprocess(output_dict)
                out = output_dict[list(output_dict.keys())[0]]

                out = out[:, :, out.shape[2] // 2,
                          out.shape[3] // 2, out.shape[4] // 2]
                out = np.squeeze(out)

                # Make one output pr output channel
                if not isinstance(class_cube, list):
                    class_cube = np.split(
                        np.repeat(
                            class_cube[:, :, :, np.newaxis], out.size, 3),
                        out.size,
                        axis=3
                    )

                # Insert into output
                if out.size == 1:
                    class_cube[0][x0, x1, x2] = out
                else:
                    for j in range(out.size):
                        class_cube[j][x0, x1, x2] = out[j]

        # Resize to input size
        if return_full_size:
            logger.infer('Resizing output to input size...')
            N = X0_grid.size

            if slice_type == 'full':
                grid_output_cube = np.concatenate(
                    [X0_grid.reshape([N, 1]), X1_grid.reshape(
                        [N, 1]), X2_grid.reshape([N, 1])], 1
                )
            elif slice_type == 'inline':
                grid_output_cube = np.concatenate(
                    [X0_grid.reshape([N, 1]), X2_grid.reshape([N, 1])], 1
                )
            elif slice_type == 'crossline':
                grid_output_cube = np.concatenate(
                    [X0_grid.reshape([N, 1]), X1_grid.reshape([N, 1])], 1
                )
            elif slice_type == 'timeslice':
                grid_output_cube = np.concatenate(
                    [X1_grid.reshape([N, 1]), X2_grid.reshape([N, 1])], 1
                )

            for i in tqdm(range(len(class_cube))):
                is_int = np.sum(
                    np.unique(class_cube[i]).astype('float') -
                    np.unique(class_cube[i]).astype('int32').astype('float')) == 0
                class_cube[i] = interpn(
                    pred_points, class_cube[i].astype(
                        'float').squeeze(), grid_output_cube,
                    method='linear', fill_value=0, bounds_error=False)
                class_cube[i] = class_cube[i].reshape(
                    [x0_range.size, x1_range.size, x2_range.size]
                )

                if is_int:
                    class_cube[i] = class_cube[i].astype('int32')

        input_ref = input_name + "-input"
        save_path = output_folder + sep + input_ref
        logger.infer('Saving output to output path: {}'.format(
            save_path + sep + "out.npy"
        ))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        np.save(save_path + sep + "out", class_cube)
        logger.infer('Complete!')


def infer_fine_cubed_async(arg_obj, logger, get_functions):
    """
    Infer on 3D seismic data asynchronously with fine cube output.
    In order to use this in your configuration, specify ``infer_type`` as
    ``fine_cube_async``. Fine cubed inference requires that additional
    parameters such as ``slice``, ``subsampl``, ``im_size``, ``slice_no``,
    and ``return_to_fullsize`` be specified in the JSON configuration file.
    This function's specific arguments will be filled according to your
    configuration inputs.

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
    logger.infer('Using model: {}'.format(model.name))

    # Expects one input and one output layer
    assert (len(model.get_inputs()) <
            2), "[ERROR] Expects model with one input layer."
    assert (len(model.get_outputs()) <
            2), "[ERROR] Expects model with one output layer."

    slice_type = arg_obj.slice
    subsampl = arg_obj.subsampl
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
        i = param_dict['order']
        output_dict = postprocess(request.outputs)

        out = output_dict[list(output_dict.keys())[0]]
        out = out[:, :, out.shape[2]//2, out.shape[3] // 2, out.shape[4] // 2]
        out = np.squeeze(out)

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

        if slice_type == 'full':
            class_cube = data[::subsampl, ::subsampl, ::subsampl] * 0

        elif slice_type == 'inline':
            slice_no = slice_no - data_info['inline_start']
            class_cube = data[::subsampl, 0:1, ::subsampl] * 0
            x1_range = np.array([slice_no])
            pred_points = (pred_points[0], pred_points[2])

        elif slice_type == 'crossline':
            slice_no = slice_no - data_info['crossline_start']
            class_cube = data[::subsampl, ::subsampl, 0:1, ] * 0
            x2_range = np.array([slice_no])
            pred_points = (pred_points[0], pred_points[1])

        elif slice_type == 'timeslice':
            slice_no = slice_no - data_info['timeslice_start']
            class_cube = data[0:1, ::subsampl, ::subsampl] * 0
            x0_range = np.array([slice_no])
            pred_points = (pred_points[1], pred_points[2])

        n0, n1, n2 = class_cube.shape
        x0_grid, x1_grid, x2_grid = np.meshgrid(
            ls(n0,), ls(n1), ls(n2), indexing='ij')
        X0_grid, X1_grid, X2_grid = np.meshgrid(
            x0_range, x1_range, x2_range, indexing='ij')
        X0_grid_sub = X0_grid[::subsampl, ::subsampl, ::subsampl]
        X1_grid_sub = X1_grid[::subsampl, ::subsampl, ::subsampl]
        X2_grid_sub = X2_grid[::subsampl, ::subsampl, ::subsampl]

        w = im_size//2
        order_dict = {}
        for i in tqdm(range(X0_grid_sub.size)):
            x0 = x0_grid.ravel()[i]
            x1 = x1_grid.ravel()[i]
            x2 = x2_grid.ravel()[i]

            X0 = X0_grid_sub.ravel()[i]
            X1 = X1_grid_sub.ravel()[i]
            X2 = X2_grid_sub.ravel()[i]

            if X0 > w and X1 > w and X2 > w and X0 < N0 - w + 1 and X1 < N1 - w + 1 and X2 < N2 - w + 1:
                mini_cube = data[X0 - w: X0 + w + 1, X1 -
                                 w: X1 + w + 1, X2 - w: X2 + w + 1]
                mini_cube = mini_cube[np.newaxis, np.newaxis, :, :, :]

                input_dict = preprocess(mini_cube, model.get_inputs())

                # Inference! input_dict => {output_layer: output_data}, latency
                infer_request = request_queue.get_idle_request()
                infer_request.start_async(input_dict, input_name, {
                    'x0x1x2': (x0, x1, x2), 'order': i, 'order_dict': order_dict
                })

        logger.infer('Cleaning up requests...')
        request_queue.wait_all()

        logger.infer('Placing prediction in proper cube spot...')
        available_keys = set(list(order_dict.keys()))
        for i in tqdm(range(X0_grid_sub.size)):
            if i in available_keys:
                out_w_param = order_dict[i]
                out = out_w_param['out']
                x0, x1, x2 = out_w_param['x0x1x2']

                # Make one output pr output channel
                if not isinstance(class_cube, list):
                    class_cube = np.split(
                        np.repeat(
                            class_cube[:, :, :, np.newaxis], out.size, 3),
                        out.size,
                        axis=3
                    )

                # Insert into output
                if out.size == 1:
                    class_cube[0][x0, x1, x2] = out
                else:
                    for i in range(out.size):
                        class_cube[i][x0, x1, x2] = out[i]

        # Resize to input size
        if return_full_size:
            logger.infer('Resizing output to input size...')
            N = X0_grid.size

            if slice_type == 'full':
                grid_output_cube = np.concatenate(
                    [X0_grid.reshape([N, 1]), X1_grid.reshape(
                        [N, 1]), X2_grid.reshape([N, 1])], 1
                )
            elif slice_type == 'inline':
                grid_output_cube = np.concatenate(
                    [X0_grid.reshape([N, 1]), X2_grid.reshape([N, 1])], 1
                )
            elif slice_type == 'crossline':
                grid_output_cube = np.concatenate(
                    [X0_grid.reshape([N, 1]), X1_grid.reshape([N, 1])], 1
                )
            elif slice_type == 'timeslice':
                grid_output_cube = np.concatenate(
                    [X1_grid.reshape([N, 1]), X2_grid.reshape([N, 1])], 1
                )

            for i in tqdm(range(len(class_cube))):
                is_int = np.sum(
                    np.unique(class_cube[i]).astype('float') -
                    np.unique(class_cube[i]).astype('int32').astype('float')) == 0
                class_cube[i] = interpn(
                    pred_points, class_cube[i].astype(
                        'float').squeeze(), grid_output_cube,
                    method='linear', fill_value=0, bounds_error=False)
                class_cube[i] = class_cube[i].reshape(
                    [x0_range.size, x1_range.size, x2_range.size]
                )

                if is_int:
                    class_cube[i] = class_cube[i].astype('int32')

        logger.infer('Squeezing outputs...')
        for i in tqdm(range(len(class_cube))):
            class_cube[i] = class_cube[i].squeeze()

        input_ref = input_name + "-input"
        save_path = output_folder + sep + input_ref
        logger.infer('Saving output to output path: {}'.format(
            save_path + sep + "out.npy"
        ))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        np.save(save_path + sep + "out", class_cube)
        logger.infer('Complete!')
