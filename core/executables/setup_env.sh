##
## Copyright (C) 2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0
##
##

# Setting up OpenVINO Environment

export PATH="$INTEL_OPENVINO_DIR/model-optimizer:$PATH"
export PYTHONPATH="$INTEL_OPENVINO_DIR/model-optimizer:$PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:$INTEL_OPENVINO_DIR/bin/intel64/Release/lib/python_api/python3.6/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INTEL_OPENVINO_DIR/bin/intel64/Release/lib/
