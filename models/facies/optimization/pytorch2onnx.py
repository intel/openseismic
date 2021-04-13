#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import torch
from texture_net import TextureNet


def pt_to_onnx(pkl_filepath, output_filename="deconvnet_test_skip.onnx"):
    model = torch.load(pkl_filepath)
    model.eval()

    input_var = torch.randn(1, 1, 255, 701)
    torch.onnx.export(
        model, input_var, output_filename,
        export_params=True,
        opset_version=10,
        dynamic_axes={
            'input.1': [1, 2, 3], '403': [1, 2, 3]
        }
    )
