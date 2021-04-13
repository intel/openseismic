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

python3 $PWD/mnt/example3_optimization/pytorch2onnx.py $ARGS
