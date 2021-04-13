#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import numpy as np
from openvino.inference_engine import IECore


class Model(object):
    """
    The FaultSeg model class contains methods important to regular inference and
    coarse cube inference. If you would like to make classes that utilize these
    inference tasks, treat the FaultSeg model class as the interface to implement.
    """
    
    def __init__(self, path_to_xml, path_to_bin, requests=1):
        """
        Initialization of FaultSeg model.

        :param path_to_xml: Path to xml file of model's OpenVINO IR.
        :param path_to_bin: Path to bin file of model's OpenVINO IR.
        :param requests: Number of requests to use during inference.
        :param n1n2n3: Tuple representing size reference for input layer. Shape will be ``(1, 1, n1, n2, n3)``.
        """
        self.ie = IECore()
        self.requests = requests
        self.fseg_net = self.ie.read_network(
            model=path_to_xml, weights=path_to_bin
        )
        self.input_layers = list(self.fseg_net.input_info.keys())
        self.output_layers = list(self.fseg_net.outputs.keys())
        self.fseg_exec = self.ie.load_network(
            network=self.fseg_net, device_name="CPU", num_requests=requests
        )
        self.name = "fseg_model"

        # warm session
        self.fseg_exec.requests[0].infer()

    def infer(self, input_dict, flexible_infer=False):
        """
        Conduct simple inference.

        :param input_dict: Input dictionary containing ``{..., layer_name:input_val, ...}``.
        :param flexible_infer: Boolean for specifying shape change mid-inference.
        :return: output_dict, latency (ms)
        """
        input_val = input_dict[self.input_layers[0]]
        input_val = input_dict[self.input_layers[0]]
        if flexible_infer and not np.all(input_val.shape == self.input_layer.shape):
            self.fseg_net.reshape({self.input_layer: input_val.shape})
            self.fseg_exec = self.ie.load_network(
                network=self.fseg_net, device_name="CPU"
            )
            self.infer_requests = self.fseg_exec.requests

        infer_requests = self.fseg_exec.requests
        infer_requests[0].infer(input_dict)
        latency = infer_requests[0].latency
        output = infer_requests[0].output_blobs[self.output_layers[0]].buffer
        output_dict = {self.output_layers[0]: output}
        return output_dict, latency  # latency in milliseconds

    def reshape_input(self, shape):
        """
        Change the shape of the input layer to ``shape``. In the case of FaultSeg, the first
        and only layer will change shape.

        :param shape: Tuple/list representing the new desired input shape.
        :return: None
        """
        shape = [1, 1, shape[0], shape[1], shape[2]]
        self.fseg_net.reshape({self.input_layers[0]: shape})
        self.fseg_exec = self.ie.load_network(
            network=self.fseg_net, device_name="CPU", num_requests=self.requests
        )
        self.infer_requests = self.fseg_exec.requests
        
    def get_input_shape(self):
        """
        Get shape of each input layer. There is only one input layer for the FaultSeg model.

        :return: Tuple representing input layer shape.
        """
        return self.fseg_net.input_info[self.input_layers[0]].input_data.shape
        
    def get_input_shape_dict(self):
        """
        Get input shape dictionary where the return value is ``{..., layer_i:layer_i_shape, ...}``.
        Since the FaultSeg model only has one layer, the dictionary returned is just
        ``{first_layer_name:first_layer_shape}``.

        :return: Layer-to-shape dictionary
        """
        ret_dict = {}
        for layer_name, layer_info_ptr in self.fseg_net.input_info.items():
            ret_dict[layer_name] = layer_info_ptr.input_data.shape
        return ret_dict

    # Base Functions Below

    def get_inputs(self):
        """
        Get input layer name(s). Only one layer name will be returned for the FaultSeg model.

        :return: List of strings representing the names of the input layers.
        """
        return self.input_layers
    
    def get_outputs(self):
        """
        Get output layer name(s). Only one layer name will be returned for the FaultSeg model

        :return: List of strings representing the names of the output layers.
        """
        return self.output_layers
    
    def get_requests(self):
        """
        Get requests object of model. The ``get_requests`` and ``get_idle_request_id`` methods
        are used for asynchronous inference.

        :return: Requests object of model.
        """
        return self.fseg_exec.requests
    
    def get_idle_request_id(self):
        """
        Get an idle request id. The ``get_requests`` and ``get_idle_request_id`` methods
        are used for asynchronous inference.

        :return: Idle request id (int)
        """
        return self.fseg_exec.get_idle_request_id()
