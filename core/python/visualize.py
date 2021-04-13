#!/usr/bin/env python3
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import logging
import os
import sys
from argparse import ArgumentParser, SUPPRESS

import matplotlib.pyplot as plt

import numpy as np


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-i', '--input',
                      help='Required. Path to seismic datafolder.',
                      required=True, type=str)
    args.add_argument('-o', '--output',
                      help='Required. Path to save seismic interpretations.',
                      default='visualization', type=str)
    args.add_argument('-m', '--model_type',
                      help='Type of model task.',
                      choices=['facies', 'salt', 'fault'],
                      type=str, default='facies')
    args.add_argument('-t', '--slice_type',
                      help='Type of slice .',
                      choices=['inline', 'crossline', 'timeline'],
                      type=str, default='crossline')
    args.add_argument('-s', '--slice_no',
                      help='Index of slice .',
                      type=int, default=0)
    return parser


def get_logger(dir_path):
    ''' Setting up Logging and return logger.'''
    INFO = 5
    logging.addLevelName(INFO, 'VISUALIZATION')

    def info(self, message, *args, **kws):
        self.log(INFO, message, *args, **kws)
    logging.Logger.info = info

    logging.basicConfig(format='[%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(os.path.join(dir_path, 'info.log')),
                        logging.StreamHandler(sys.stdout)]
                        )
    logger = logging.getLogger('visualization')
    logger.setLevel(INFO)
    return logger


def decode_segmap(label_mask):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
            the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
            in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = np.asarray([[69, 117, 180], [145, 191, 219],
                                [224, 243, 248], [254, 224, 144],
                                [252, 141, 89], [215, 48, 39]])
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, len(label_colours)):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def show_facies_interpretation(args, labels):
    from matplotlib.colors import LinearSegmentedColormap

    res_image = decode_segmap(labels.squeeze())

    color_list = np.asarray([[69, 117, 180], [145, 191, 219],
                            [224, 243, 248], [254, 224, 144],
                            [252, 141, 89], [215, 48, 39]]) / 255
    cm = LinearSegmentedColormap.from_list('custom_cmap', color_list, N=6)
    

    fig = plt.figure()
    ax = plt.subplot()
    fig.suptitle("Facies classification results", fontsize=22)
    im = ax.imshow(res_image, cmap=cm)
    ax.set_title('Interpretation of the slice')

    cbaxes = fig.add_axes([0.81, 0.2, 0.01, 0.6])
    cb = fig.colorbar(im, ax=ax, cax=cbaxes,
                      ticks=[0.23, 0.36, 0.5, 0.65, 0.78, 0.93])
    cb.ax.set_yticklabels(['upper_ns', 'middle_ns', 'lower_ns',
                          'rijnland_chalk', 'scruff', 'zechstein'],
                          fontsize=9, ha="left")
    plt.savefig(os.path.join(args.output, 'interpretation.png'))
    plt.show()
    return os.path.join(args.output, 'interpretation.png')


def show_fault_interpretation(args, labels, slice_index=0):
    assert slice_index >= 0, \
        'Invalid slice index argument, slice index must not be negative'
    labels[labels > 0.5] = 1
    labels[labels <= 0.5] = 0
    x, y, z = labels.shape
    fig = plt.figure()
    fig.suptitle("Fault segmentation", fontsize=22)
    plt.subplot(1, 3, 1)
    assert slice_index < x, f'Invalid slice index, must be in {[0, x - 1]}'
    plt.imshow(labels[slice_index, :, :], interpolation='nearest', aspect=1)
    plt.title(f"Time slice, index - {slice_index}")
    plt.subplot(1, 3, 2)
    assert slice_index < y, f'Invalid slice index, must be in {[0, y - 1]}'
    plt.imshow(labels[:, slice_index,:], interpolation='nearest', aspect=1)
    plt.title(f"Cross slice, index - {slice_index}")
    plt.subplot(1, 3, 3)
    assert slice_index < z, f'Invalid slice index, must be in {[0, z - 1]}'
    plt.imshow(labels[:, :, slice_index], interpolation='nearest', aspect=1)
    plt.title(f"Inline slice, index - {slice_index}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'interpretation.png'))
    plt.show()
    return os.path.join(args.output, 'interpretation.png')


def show_salt_interpretation(args, labels, slice_index=0, slice_type='crossline'):
    if len(labels.shape) > 3:
        labels = labels.argmax(axis=0)
    assert slice_index >= 0, \
        'Invalid slice index argument, slice index must not be negative'
    x, y, z = labels.shape
    fig = plt.figure()
    fig.suptitle("Salt segmentation", fontsize=22)
    if slice_type == 'timeline':
        assert slice_index < x, f'Invalid slice index, must be in {[0, x - 1]}'
        img = labels[slice_index, :, :]
    elif slice_type == 'crossline':
        assert slice_index < y, f'Invalid slice index, must be in {[0, y - 1]}'
        img = labels[:, slice_index, :]
    elif slice_type == 'inline':
        assert slice_index < z, f'Invalid slice index, must be in {[0, z - 1]}'
        img = labels[:, :, slice_index]
    plt.imshow(img, interpolation='nearest', cmap='Greys', aspect=1)
    plt.title(f"{slice_type} slice, index - {slice_index}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'interpretation.png'))
    plt.show()
    return os.path.join(args.output, 'interpretation.png')

def get_datafile_path(input_path):
    for dirs, _, files in os.walk(input_path):
        if files:
            return os.path.join(dirs, files[0])
    return None
     

def main(args, logger):
    path_to_file = get_datafile_path(args.input)
    data_file = np.load(path_to_file)
    logger.info(f"Data file {path_to_file} -  loaded")
    if args.model_type == 'facies':
        saved_datapath = show_facies_interpretation(args, data_file.T)
    elif args.model_type == 'fault':
        saved_datapath = show_fault_interpretation(args, data_file, args.slice_no)
    elif args.model_type == 'salt':
        saved_datapath = show_salt_interpretation(args, data_file, args.slice_no, args.slice_type)
    logger.info(f"Visualization complete! File saved to: {saved_datapath}" )


if __name__ == '__main__':
    args = build_argparser().parse_args()

    all_runs_subdirs = [os.path.join('runs', d) for d in os.listdir('./runs')]
    latest_run_subdir = max(all_runs_subdirs, key=os.path.getmtime)
    args.input = os.path.join(latest_run_subdir, args.input)
    args.output = os.path.join(latest_run_subdir, args.output)
    os.makedirs(args.output, exist_ok=True)

    logger = get_logger(latest_run_subdir)
    main(args, logger)
