#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import argparse
import torch
from texture_net import TextureNet

def args():
    """
    Argument Parsing Handler:
    -m <path_to_keras> :
    Path to keras model

    -o <model_output> :
    Path to directory that will store pb model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--path_to_state", type=str,
                        help="Path to Pytorch state dictionary.", default='')
    parser.add_argument("-o", "--model_output", type=str,
                        help="Path to directory that will store .onnx model.", default='')
    return parser.parse_args()

def pt_to_onnx(pt_filename, output_filename):
    network = TextureNet(n_classes=2)
    device = torch.device('cpu')
    network.load_state_dict(torch.load(pt_filename, map_location=device))
    network.eval()

    input_var = torch.randn(1, 1, 65, 65, 65,)  # device='cuda')
    torch.onnx.export(network, input_var, output_filename, export_params=True)
    
if __name__ == '__main__':
    arg_obj = args()
    assert arg_obj.path_to_state != '', '[ERROR] No keras path given.'
    assert arg_obj.model_output != '', '[ERROR] No output path given.'
    pt_to_onnx(arg_obj.path_to_state, arg_obj.model_output)