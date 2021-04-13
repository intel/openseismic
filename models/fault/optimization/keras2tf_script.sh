#!/bin/bash
##
## Copyright (C) 2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0
##
##

for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
            *) ARGS="$ARGS $KEY" ;;
    esac
done

source /opt/intel/openvino/bin/setupvars.sh
python3 $PWD/demo/demo_faultseg/model_keras/fseg_keras2tf.py $ARGS
