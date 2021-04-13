#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import os
import threading

import segyio
import numpy as np

from datetime import datetime
from openvino.inference_engine import IECore


class OV_simulator(object):
    """
    The ``OV_simulator`` class is a bare-bones wrapper for doing inference in Open Seismic.
    """
    
    def __init__(self, path_to_xml,
                 path_to_bin, requests=1,
                 streams='CPU_THROUGHPUT_AUTO'):
        """
        Initialization of model.

        :param path_to_xml: Path to xml file of model's OpenVINO IR.
        :param path_to_bin: Path to bin file of model's OpenVINO IR.
        :param requests: Number of requests to use during inference.
        :param streams: Number of streams or a specification of stream usage.
        :param n1n2n3: Tuple representing size reference for input layer. Shape will be ``(1, 1, n1, n2, n3)``.
        """
        self.ie = IECore()
        self.ie.set_config({'CPU_THROUGHPUT_STREAM': str(streams)}, 'CPU')
        self.net = self.ie.read_network(model=path_to_xml, weights=path_to_bin)
        self.input_layers = list(self.net.inputs.keys())
        self.output_layers = list(self.net.outputs.keys())
        self.exec_net = self.ie.load_network(
            network=self.net, device_name="CPU", num_requests=requests
        )
        self.name = "OV_simulator"
        self.infer_requests = self.exec_net.requests

        # warm session
        self.infer_requests[0].infer()

    def get_inputs(self):
        """
        Get input layer name(s).

        :return: List of strings representing the names of the input layers.
        """
        return self.input_layers

    def get_outputs(self):
        """
        Get output layer name(s).

        :return: List of strings representing the names of the output layers.
        """
        return self.output_layers

    def get_requests(self):
        """
        Get requests object of model. The ``get_requests`` and ``get_idle_request_id`` methods
        are used for asynchronous inference.

        :return: Requests object of model.
        """
        return self.exec_net.requests
    
    def get_input_shape(self):
        """
        Get shape of each input layer. If there is only one layer, then a tuple will be returned. If there are
        multiple layers, then a list of tuples will be returned.

        :return: Tuple representing input layer shape or a dictionary mapping layer to shape if multiple layers
        """
        if len(list(self.self.net.input_info.keys())) < 2:
            return self.net.input_info[self.input_layers[0]].input_data.shape
        else:
            return self.get_input_shape_dict()

    def get_input_shape_dict(self):
        """
        Get input shape dictionary where the return value is ``{..., layer_i:layer_i_shape, ...}``.

        :return: Layer-to-shape dictionary
        """
        ret_dict = {}
        for layer_name, layer_info_ptr in self.net.input_info.items():
            ret_dict[layer_name] = layer_info_ptr.input_data.shape
        return ret_dict

    def get_idle_request_id(self):
        """
        Get an idle request id. The ``get_requests`` and ``get_idle_request_id`` methods
        are used for asynchronous inference.

        :return: Idle request id (int)
        """
        return self.fseg_exec.get_idle_request_id()

    def infer(self, input_val):
        """
        Conduct simple inference.

        :param input_dict: Input dictionary containing ``{..., layer_name:input_val, ...}``.
        :return: Tuple containing (1) an output dictionary mapping output layer to output value and (2) latency in milliseconds.
        """
        self.infer_requests[0].infer(input_val)
        latency = self.infer_requests[0].latency
        ret_val = self.infer_requests[0].outputs
        return ret_val, latency
    
    def reshape_input(self, shape):
        """
        Change the shape of the input layers. Shape may be a tuple/list or it may be a dictionary mapping
        layers to their new shapes.

        :param shape: Tuple/list representing the new desired input shape.
        :return: None
        """
        if type(shape) == dict:
            self.net.reshape(shape)
        else:
            self.net.reshape({layers: shape for layers in self.input_layers})
        self.exec_net = self.ie.load_network(
            network=self.net, device_name="CPU", num_requests=self.requests
        )
        self.infer_requests = self.exec_net.requests


class InferReqWrap:
    """
    The ``InferReqWrap`` class offers additional logic on top of the OpenVINO requests API in order to conduct
    and store inference per request.
    """
    
    def __init__(self, request,
                 req_id, custom_callback,
                 callback_args, callback_queue):
        """
        Initialization of the ``InferReqWrap`` class.

        :param request: Request from model paired with ``req_id``.
        :param req_id: The request id associated with ``request``.
        :param custom_callback: A custom callback function to execute after inference is completed.
        :param callback_args: Arguments to be passed into the custom_callback function.
        :param callback_queue: A callback function from an ``InferRequestsQueue`` object.
        """
        self.req_id = req_id
        self.request = request
        self.custom_callback = custom_callback
        self.request.set_completion_callback(self.callback, callback_args)
        self.callbackQueue = callback_queue
        self.extra_params = {}
        self.file_name = None
        self.__preds = []

    def callback(self, status_code, user_data):
        """
        This callback function will be executed once an inference request is complete. First, the
        custom callback is called with its appropriate arguments. Then, the ``InferRequestQueue`` callback
        will be called to clean up the inference request and set the request associated with this object as
        idle.

        :param status_code: The status code of the inference request.
        :param user_data: Arguments and other data used for the custom callback function.
        :return: None
        """
        if user_data['req_id'] != self.req_id:
            print(
                f"\n[WARNING] Request ID {self.req_id} does not correspond to user data {user_data['req_id']}")
        elif status_code:
            print(
                f"\n[ERROR] Request {self.req_id} failed with status code {status_code}")
        user_data['request'] = self.request
        user_data['file_name'] = self.file_name

        for key in self.extra_params.keys():
            user_data[key] = self.extra_params[key]

        out_w_params = self.custom_callback(user_data)
        self.__preds.append(out_w_params)
        self.callbackQueue(self.req_id, self.request.latency)

    def start_async(self, input_data, file_name, extra_params={}):
        """
        Starts asynchronous inference request.

        :param input_data: A dictionary mapping layers to their input data.
        :param file_name: Name of the file associated with the input data.
        :param extra_params: Params that might need to be used in the callback.
        :return: None
        """
        self.file_name = file_name
        self.extra_params = extra_params
        self.request.async_infer(input_data)

    def get_preds(self):
        """
        Gets the predictions that were computed by ``self.request``.

        :return: List of dictionaries that map output layers to their output values.
        """
        return self.__preds

    def infer(self, input_data, ground_truth=None, extra_params={}):
        """
        Starts synchronous inference request.

        :param input_data: A dictionary mapping layers to their input data.
        :param ground_truth: Label for the associated input data
        :param extra_params: Params that might need to be used in the callback.
        :return: None
        """
        self.request.infer(input_data)
        self.extra_params = extra_params
        self.callbackQueue(self.req_id, self.request.latency)


class InferRequestsQueue:
    """
    The ``InferRequestQueue`` manages the execution of requests that are wrapped in ``InferReqWrap`` objects.
    """
    
    def __init__(self, requests, callback, postprocess):
        """
        Initialization of the ``InferRequestsQueue``.

        :param requests: A requests object from the execution network from ``IECore``.
        :param callback: A custom callback that will be executed when an inference request is finished.
        :param postprocess: A postprocess function that may or may not be in the custom callback function.
        """
        self.idleIds = []
        self.requests = []
        self.times = []
        for req_id in range(len(requests)):
            self.requests.append(InferReqWrap(
                requests[req_id], req_id, callback,
                {'postprocess': postprocess, 'req_id': req_id},
                self.put_idle_request
            ))
            self.idleIds.append(req_id)
        self.startTime = datetime.max
        self.endTime = datetime.min
        self.cv = threading.Condition()

    def reset_times(self):
        """
        Resets the inference times.

        :return: None
        """
        self.times.clear()

    def get_duration_in_seconds(self):
        """
        Gets the total duration of inference in terms of seconds.

        :return: total duration in seconds
        """
        return (self.endTime - self.startTime).total_seconds()

    def put_idle_request(self, req_id, latency):
        """
        Sets the request associated with ``req_id`` to idle.

        :param req_id: A request id associated with a request.
        :param latency: The latency of the inference request.
        :return: None
        """
        self.cv.acquire()
        self.times.append(latency)
        self.idleIds.append(req_id)
        self.endTime = max(self.endTime, datetime.now())
        self.cv.notify()
        self.cv.release()

    def get_idle_request(self):
        """
        Gets an idle request.

        :return: A request object that is idle.
        """
        self.cv.acquire()
        while len(self.idleIds) == 0:
            self.cv.wait()
        req_id = self.idleIds.pop()
        self.startTime = min(datetime.now(), self.startTime)
        self.cv.release()
        return self.requests[req_id]

    def wait_all(self):
        """
        Waits for all inference requests to finish. This is particularly useful when all inputs
        have been put on the queue for execution. If called, this function will block the main thread until
        the requests have finished.

        :return: None
        """
        self.cv.acquire()
        while len(self.idleIds) != len(self.requests):
            self.cv.wait()
        self.cv.release()


def loader(file):
    """
    Given a path ``file``, this function converts this file type to a file
    type known by Open Seismic. The currently supported file types are
    ``.segy``, ``.npy``, and ``.dat``.

    :param file: Path to the file to be converted.
    :return: The converted file
    """
    print("[LOADER] Loading file: {}".format(file))
    
    file_name, file_ext = os.path.splitext(file)
    if file_ext.lower() == '.segy':
        return segy_to_np(file)
    elif file_ext.lower() == '.npy':
        return np.load(file), None
    elif file_ext.lower() == '.dat':
        return np.fromfile(file, dtype=np.single), None
    else:
        assert False, "[ERROR] Unsupported file type: {}".format(file_ext)


def segy_to_np(filename):
    """
    Given a ``file_path`` of a segy file, this function converts the segy
    trace data to a numpy data representation.

    :param file_path: Path to the segy data.
    :return: A tuple containing (1) NumPy data and (2) a dictionary with information from segy file.
    """
    data = segyio.tools.cube(filename)
    data = np.moveaxis(data, -1, 0)
    data = np.ascontiguousarray(data, 'float32')
    segyfile = segyio.open(filename, "r")

    data_info = {}
    data_info['crossline_start'] = segyfile.xlines[0]
    data_info['inline_start'] = segyfile.ilines[0]
    data_info['timeslice_start'] = 1  # Todo: read this from segy
    data_info['shape'] = data.shape

    return data, data_info
