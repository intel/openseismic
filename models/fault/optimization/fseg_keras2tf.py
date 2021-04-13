#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import argparse
from unet3 import cross_entropy_balanced
import tensorflow.keras.backend as K
from keras.models import load_model
from keras.layers import *
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util
import tensorflow as tf
import numpy as np
import warnings
from tensorflow.python.util import deprecation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
deprecation._PRINT_DEPRECATION_WARNINGS = False
warnings.filterwarnings("ignore")

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


os.environ["CUDA_VISIBLE_DEVICES"] = ""


def args():
    """
    Argument Parsing Handler:
    -m <path_to_keras> :
    Path to keras model

    -o <model_output> :
    Path to directory that will store pb model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--path_to_keras", type=str,
                        help="Path to keras model.", default='')
    parser.add_argument("-o", "--model_output", type=str,
                        help="Path to directory that will store pb model.", default='')
    return parser.parse_args()


def main(args):
    model = load_model(args.path_to_keras, custom_objects={
                       'cross_entropy_balanced': cross_entropy_balanced})
    frozen_model_path = args.model_output
    frozen_model_name = 'fseg-60.pb'

    if not os.path.isdir(frozen_model_path):
        os.mkdir(frozen_model_path)

    tf.get_logger().setLevel('INFO')
    K.set_image_data_format('channels_last')

    orig_output_node_names = [node.op.name for node in model.outputs]
    converted_output_node_names = orig_output_node_names

    sess = K.get_session()
    constant_graph = graph_util.convert_variables_to_constants(sess,
                                                               sess.graph.as_graph_def(),
                                                               converted_output_node_names)
    graph_io.write_graph(constant_graph, frozen_model_path,
                         frozen_model_name, as_text=False)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arg_obj = args()
        assert arg_obj.path_to_keras != '', '[ERROR] No keras path given.'
        assert arg_obj.model_output != '', '[ERROR] No output path given.'
        main(arg_obj)
