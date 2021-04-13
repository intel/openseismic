#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import os
import json
import argparse
import subprocess
from pathlib import PurePath


def get_params_dict(filename):
    """
    This function turns a filename string, opens and loads a JSON object as
    a dictionary, and returns this data dictionary.

    :param filename: The file name of a JSON file, most likely the configuration file.
    :return: Param Dictionary (data)
    """
    with open(filename) as f:
        data = json.load(f)
        return data


def json_args():
    """
    Parses the arguments passed in by the user. The only argument needed, in this case,
    is the file name to the configuration JSON file.

    :param: None
    :return: Parsed Args (parser.parse_args())
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--run_config", type=str,
                        help="Path to OpenSeismic JSON config", default='')
    return parser.parse_args()


if __name__ == "__main__":
    """
    This code parses the JSON configuration file given by the user and executes
    commands per the available arguments in the configuration file. Specifically,
    the code block is concerned with arguments for:
        (1) Running pre model optimization, which is important when OpenVINO
            does not support the current model framework.
        (2) Running OpenVINO model optimization
        (3) Conducting inference on the data with the specified model/processors.
    """
    json_obj = get_params_dict(json_args().run_config)
    pre_mo_output = mo_output = inf_output = 0
    executable_path = PurePath(os.getcwd()).joinpath('executables')

    if 'pre_model_optimizer_params' in list(json_obj.keys()):
        if len(json_obj['pre_model_optimizer_params'].keys()) > 0:
            params = json_obj['pre_model_optimizer_params']
            param_arr = [
                f"--{k} {params[k]}" for k in params.keys() if k != 'script']
            param_str = " ".join(param_arr)
            script = params['script']

            # Calling pre_mo script
            pre_mo_output = subprocess.call([script, param_str])

    assert pre_mo_output == 0, "[ERROR] Pre Model Conversion Unsuccessful."

    if 'model_optimizer_params' in list(json_obj.keys()):
        if len(json_obj['model_optimizer_params'].keys()) > 0:
            params = json_obj['model_optimizer_params']
            param_arr = [f"--{k} {params[k]}" for k in params.keys()]
            param_str = " ".join(param_arr)

            # Calling mo script
            mo_path = executable_path.joinpath('mo.sh')
            mo_output = subprocess.call([str(mo_path), param_str])

    assert mo_output == 0, "[ERROR] Model Conversion Unsuccessful."

    if 'inference_params' in list(json_obj.keys()):
        if len(json_obj['inference_params'].keys()) > 0:
            params = json_obj['inference_params']
            param_arr = [f"--{k} {params[k]}" for k in params.keys()]
            param_str = " ".join(param_arr)

            # Calling infer script
            infer_path = executable_path.joinpath('infer.sh')
            inf_output = subprocess.call([str(infer_path), param_str])

    assert inf_output == 0, "[ERROR] Model Inference Unsuccessful."

    if 'visualize_params' in list(json_obj.keys()):
        if len(json_obj['visualize_params'].keys()) > 0:
            params = json_obj['visualize_params']
            param_arr = [f"--{k} {params[k]}" for k in params.keys()]
            param_str = " ".join(param_arr)

            # Calling visualize script
            vis_path = executable_path.joinpath('visualize.sh')
            inf_output = subprocess.call([str(vis_path), param_str])

    assert inf_output == 0, "[ERROR] Visualization unsuccessful."
