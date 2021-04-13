#!/bin/bash
##
## Copyright (C) 2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0
##
##
mkdir -p runs data_mnt/Dutch_F3_data data_mnt/Synthetic_fault_data/seis
cd data_mnt/Dutch_F3_data
gdown https://drive.google.com/uc?id=0B7brcf-eGK8CUUZKLXJURFNYeXM
mv Dutch\ Government_F3_entire_8bit\ seismic.segy f3_8bit.segy
cd ../Synthetic_fault_data/seis
gdown https://drive.google.com/uc?id=1O_PmvT8cyP59RrDhT2F2AGisJa5CLI0Z
gdown https://drive.google.com/uc?id=1sWpDDLfJeBXFMfpGR27hndlsWOG9NJlW
gdown https://drive.google.com/uc?id=17WfqmnRy1PTbnoWlnAs5JqgO2gEAV5n1
gdown https://drive.google.com/uc?id=1xXxX6X1miCuiOZH4qgFnLaKGRyCQ1TRB
gdown https://drive.google.com/uc?id=16TtWuUJ58A87qtCQYQtD8M-6GgcFbzSQ
