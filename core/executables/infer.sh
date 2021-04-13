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

# Uncomment below if using Intel Distribution
# source /opt/intel/openvino/bin/setupvars.sh
source $PWD/executables/setup_env.sh
python3 $PWD/python/infer.py $ARGS
