##
## Copyright (C) 2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0
##
##
FROM ubuntu:18.04
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
software-properties-common && add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
cpio \
curl \
git \
lsb-release \
pciutils \
python3.7 \
python3.7-dev \
python3-pip \
python3-setuptools \
curl \
gnupg \
wget \
git \
libssl-dev \
vim \
sudo && \
rm -rf /var/lib/apt/lists/*

# Make python point to python3 which points to python3.7
RUN ln -sf /usr/bin/python3.7 /usr/bin/python3 
RUN ln -sf /usr/bin/python3.7 /usr/bin/python 

RUN pip3 install --upgrade pip
RUN pip3 install numpy

# Downloading Open Source
RUN git clone -b releases/2021/2 https://github.com/openvinotoolkit/openvino.git && \
cd openvino && git submodule update --init --recursive

# Build and Install OpenVINO
RUN cd openvino && chmod +x install_build_dependencies.sh && \
./install_build_dependencies.sh && \
cd inference-engine/ie_bridges/python && \
python3 -m pip install -r requirements.txt && \
cd /openvino/model-optimizer/install_prerequisites/ &&\
./install_prerequisites.sh && \
cd /openvino && mkdir build && cd build && \
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=`which python3` .. && \
make --jobs=$(nproc --all)

# Install Open Seismic dependencies.
COPY ./requirements.txt /os_requirements.txt
RUN python3 -m pip install -r os_requirements.txt


# Setup OpenVINO envirnoment
ENV INTEL_OPENVINO_DIR="/openvino"
RUN mkdir -p /openvino/deployment_tools/tools/ && \
    ln -s  /openvino/inference-engine/tools/benchmark_tool /openvino/deployment_tools/tools/benchmark_tool
ENV PYTHONPATH=/openvino/bin/intel64/Release/lib/python_api/python3.7/:$PYTHONPATH
ENV LD_LIBRARY_PATH=/openvino/inference-engine/temp/tbb/lib:$LD_LIBRARY_PATH
ENV PATH=/openvino/model-optimizer/:$PATH

# Copying Necessary Files
COPY ./core/ /core/
WORKDIR /core

