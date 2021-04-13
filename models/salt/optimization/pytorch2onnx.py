#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import torch
from texture_net import TextureNet


def pt_to_onnx(pt_filename, output_filename):
    network = TextureNet(n_classes=2)
    network.load_state_dict(torch.load(pt_filename))
    network.eval()

    input_var = torch.randn(1, 1, 65, 65, 65,)  # device='cuda')
    torch.onnx.export(network, input_var, output_filename, export_params=True)
