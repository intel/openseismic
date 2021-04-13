#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import os
import errno
import shutil
import argparse


def copyfile(src, dst, file):
    """
    This simple function copies a file named ``file`` from the ``src`` path to the ``dst`` path.

    :param src: Path of the source file.
    :param dst: Path of the destination file.
    :param file: Name of the file to be copied.
    :return: None
    """
    sep = os.path.sep
    if not os.path.exists('.' + sep + dst + sep + file):
        if not os.path.exists('.' + sep + dst):
            os.mkdir('.' + sep + dst)
        with open('.' + sep + dst + sep + file, 'w'):
            pass
    src, dst = '.' + sep + src + sep + file, '.' + sep + dst + sep + file
    try:
        shutil.copytree(src, dst)
    except OSError as exc:
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else:
            raise


def args():
    """
    This function helps parse command line arguments. The configuration JSON
    file arguments will be passed to their respective submodules (conversion,
    OpenVINO optimization, and inference). However, this function is purely
    concerned with inference arguments. The other arguments are used for the
    user's own conversion script or the model optimizer script that comes with
    OpenVINO.

    The main arguments are as follows:
     - ``model``: OpenVINO model folder path for swappable components
     - ``given_model_name``: Open Seismic model name
     - ``data``: Data folder path
     - ``benchmarking``: Flag to enable/disable model benchmarking
     - ``output``: Prediction output path
     - ``infer_type``: (``sync`` / ``async`` / ``fine_cube_sync`` / ``fine_cube_async`` / ``coarse_cube_sync`` / ``section_sync`` / ``section_async``) Type of inference
     - ``streams``: Number of streams for inference

    The optional arguments are as follows. Note that although they are optional,
    certain inference tasks need these arguments in order to run:
     - ``slice``: (``full`` / ``inline`` / ``crossline`` / ``timeslice``) ONLY FOR FINE CUBE/SECTION INFER. Slice type for cutting into data
     - ``window``: (``...;l1i:l2i,w1i:w2i,h1i:h2i;...``) ONLY FOR COARSE CUBE INFER. Infer on ``cube_i[l1i:l2i,w1i:w2i,h1i:h2i]``.
     - ``subsampl``: ONLY FOR FINE CUBE/SECTION INFER. Look at every n-th point
     - ``slice_no``: ONLY FOR FINE CUBE/SECTION INFER. Slice number
     - ``im_size``: ONLY FOR FINE CUBE/COARSE CUBE/SECTION INFER. Side length of cube/section to window in
     - ``return_to_fullsize``: ONLY FOR FINE CUBE INFER. Output will be the full size of the input (done through interpolation)

    :return: Argument Namespace Object mapping arguments to their respective values.
    """
    def interpret_bool(i):
        if isinstance(i, bool):
            return i
        if i.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif i.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')
            
    def interpret_list(i):
        if isinstance(i, list):
            return i
        if isinstance(i, str):
            ret_list = i.split(';')
            ret_list = [[k.split(":") for k in j.split(",")] for j in ret_list]
        raise argparse.ArgumentTypeError(
            'list or ...;#:#,#:#,#:#;... value expected')

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        help="OpenVINO Model Folder Path \
                              for Swappable Componenets",
                        default='')
    parser.add_argument("-g", "--given_model_name", type=str,
                        help="ISI Model Name",
                        default='')
    parser.add_argument("-d", "--data", type=str, help="Data Folder Path",
                        default='')
    parser.add_argument("-b", "--benchmarking",
                        help="Flag to enable/disable model benchmarking",
                        action='store_true')
    parser.add_argument("-o", "--output", type=str,
                        help="Prediction Output Path",
                        default='')
    parser.add_argument("-api", "--infer_type", type=str,
                        help="Syncrounous or Asyncronous Infer \
                              with cube window option",
                        choices=["sync", "async",
                                 "coarse_cube_sync", 
                                 "coarse_cube_async", 
                                 "fine_cube_sync",
                                 "fine_cube_async",
                                 "section_sync", "section_async"],
                        default='')
    parser.add_argument("-s", "--streams", type=int,
                        help="Number of streams to do async tasks.",
                        default=6)

    # Extra Params for Cubed Inference
    parser.add_argument("-sl", "--slice", type=str,
                        help="(full/inline/crossline/timeslice) \
                              ONLY FOR CUBE/SECTION INFER. \
                              Slice type for cutting into data.",
                        default='full')
    parser.add_argument("-wn", "--window", type=interpret_list, 
                        help="(...;l1i:l2i,w1i:w2i,h1i:h2i;...) \
                        ONLY FOR COARSE CUBE INFER. \
                        Infer on cubed_i[l1i:l2i,w1i:w2i,h1i:h2i].",
                        default=[])
    parser.add_argument("-sub", "--subsampl", type=int,
                        help="ONLY FOR CUBE/SECTION INFER. \
                              Look at every n-th point.",
                        default=16)
    parser.add_argument("-sln", "--slice_no", type=int,
                        help="ONLY FOR CUBE/SECTION INFER. Slice number.",
                        default=339)
    parser.add_argument("-ims", "--im_size", type=int,
                        help="ONLY FOR CUBE/SECTION INFER. \
                              Side length of cube to window in.",
                        default=65)
    parser.add_argument("-full_ret", "--return_to_fullsize",
                        type=interpret_bool,
                        help="ONLY FOR CUBE/SECTION INFER. \
                              Output will be the full size \
                              of the input (done through interpolation).",
                        default=False, const=True, nargs='?')
    return parser.parse_args()
