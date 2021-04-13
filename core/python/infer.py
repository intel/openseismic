#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
import os
import sys
import logging
import warnings
import datetime
import importlib
import subprocess
import infer_section
import infer_regular
import infer_fine_cube
import infer_coarse_cube

from functools import partial

from utils.infer_util import OV_simulator
import utils.model_paths as model_paths
import utils.argparse_util as argparse_util

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setting up Logging
log_folder =  os.path.join('runs', datetime.datetime.now().strftime("%B%d_%I-%M-%S_%p_%Y"))
path_to_log_file =  os.path.join(log_folder, 'info.log')
os.makedirs(os.path.dirname(path_to_log_file), exist_ok=True)
logging.basicConfig(
    format='[%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(path_to_log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

OUTPUT, SWAP, INFER, ARG = 5, 6, 7, 8

logging.addLevelName(OUTPUT, 'OUTPUT')


def out(self, message, *args, **kws):
    self.log(OUTPUT, message, *args, **kws)


logging.Logger.output = out

logging.addLevelName(SWAP, 'SWAP')


def swap(self, message, *args, **kws):
    self.log(SWAP, message, *args, **kws)


logging.Logger.swap = swap

logging.addLevelName(INFER, 'INFER')


def infer(self, message, *args, **kws):
    self.log(INFER, message, *args, **kws)


logging.Logger.infer = infer

logging.addLevelName(ARG, 'ARG PARSE')


def arg(self, message, *args, **kws):
    self.log(ARG, message, *args, **kws)


logging.Logger.arg = arg

# logging.basicConfig(format='[%(levelname)s] %(message)s')
logger = logging.getLogger()


def get_functions(is_benchmarking, model_folder_path, given_model_name,
                  infer_type='sync', streams=1):
    """
    Given at least a folder path to your model and an OpenSeismic model name,
    return a model class and its associated preprocess and postprocess functions.
    These functions will be based on the contents within the folder path. For instance,
    if a user specifies a model name and the model folder contains no user defined
    scripts, then the model class and its associates preproceess/postprocess functions
    come from OpenSeismic's predefined classes and functions associated with the given
    model name. Please refer to the Part 1 tutorial notebook for details.

    Note that these arguments will be filled by a user's configuration json file.

    :param model_folder_path: Path to user defined scripts have to do with model processes.
    :param given_model_name: A name of an OpenSeismic model.
    :param infer_type: The infer type for a specific inference task.
    :param streams: The number of streams to conduct inference with.
    :return: preprocess, postprocess, model
    """
    preprocess = None
    postprocess = None
    model_function = None
    model = None
    model_xml = ''
    model_bin = ''

    logger.setLevel(SWAP)

    sep = os.path.sep
    if os.path.exists(model_folder_path + sep + 'preprocessor.py'):
        logger.swap('Found custom preporcessing script.')
        preprocessor = importlib.import_module(
            model_folder_path.replace('/', '.') + 'preprocessor')
        preprocess = preprocessor.preprocess
    else:
        logger.swap(
            'Preprocessor not found. Defaulting to given model preprocess.')
        assert (given_model_name in list(model_paths.paths_dict.keys())),\
            '[ERROR] Argument for given model name does not match any name in Open Seismic.'
        prep_import = model_paths.demo_paths_for_imports[given_model_name] + '.preprocessor'
        preprocessor = importlib.import_module(prep_import)
        preprocess = preprocessor.preprocess

    if os.path.exists(model_folder_path + sep + 'postprocessor.py'):
        logger.swap('Found custom postporcessing script.')
        postprocessor = importlib.import_module(
            model_folder_path.replace('/', '.') + 'postprocessor')
        postprocess = postprocessor.postprocess
    else:
        logger.swap(
            'Postprocessor not found. Defaulting to given model postprocess.')
        assert (given_model_name in list(model_paths.paths_dict.keys())),\
            '[ERROR] Argument for given model name does not match any name in Open Seismic.'
        post_import = model_paths.demo_paths_for_imports[given_model_name] + '.postprocessor'
        postprocessor = importlib.import_module(post_import)
        postprocess = postprocessor.postprocess

    if os.path.exists(model_folder_path + sep + 'model.py'):
        logger.swap('Found custom model script.')
        custom_model_file = importlib.import_module(
            model_folder_path.replace('/', '.') + 'model')
        model_function = custom_model_file.Model

        xmls_bins = [f for f in os.listdir(model_folder_path)
                     if f.endswith('.xml') or f.endswith('.bin')]
        assert (len(xmls_bins) <=
                2), "[ERROR] Multiple xml/bin files in model folder"

        for fname in xmls_bins:
            if fname.endswith('.xml'):
                logger.swap('Found custom .xml file.')
                model_xml = model_folder_path + sep + fname
            elif fname.endswith('.bin'):
                logger.swap('Found custom .bin file.')
                model_bin = model_folder_path + sep + fname

        if model_xml != '' and model_bin != '':
            logger.swap(
                'Loading custom bin and xml into custom model initialization.')
            assert (
                'sync' in infer_type), "[ERROR] Argument invalid for infer type."
            assert (streams > 0), "[ERROR] Argument invalid for streams."
            if 'sync' in infer_type and 'async' not in infer_type:
                model = model_function(model_xml, model_bin, 1)
            elif 'async' in infer_type:
                model = model_function(model_xml, model_bin, streams)
            logger.swap('Successfully loaded custom bin and xml.')

    # Defaulting logic control for model initialization
    if model_function is None and model_xml != '' and model_bin != '':
        logger.swap(
            'Custom model initialization not found. Using default model init with custom bin and xml.')
        assert (os.path.exists(model_xml)
                ), f"[ERROR] given xml path: {model_xml} invalid"
        assert (os.path.exists(model_bin)
                ), f"[ERROR] given bin path: {model_bin} invalid"

        assert (
            'sync' in infer_type), "[ERROR] Argument invalid for infer type."
        assert (streams > 0), "[ERROR] Argument invalid for streams."
        if 'sync' in infer_type and 'async' not in infer_type:
            model = OV_simulator(model_xml, model_bin, 1)
        elif 'async' in infer_type:
            model = OV_simulator(model_xml, model_bin, streams)

    elif model_function is not None and (model_xml == '' or model_bin == ''):
        logger.swap(
            'Custom bin or xml not found. Using given model xml and bin with custom model init.')
        assert (given_model_name in list(model_paths.paths_dict.keys())),\
            '[ERROR] Argument for given model name does not match any name in Open Seismic.'
        given_xml, given_bin = model_paths.paths_dict[given_model_name]
        assert (os.path.exists(given_xml)
                ), f"[ERROR] given xml path: {given_xml} invalid"
        assert (os.path.exists(given_bin)
                ), f"[ERROR] given bin path: {given_bin} invalid"

        assert (
            'sync' in infer_type), "[ERROR] Argument invalid for infer type."
        assert (streams > 0), "[ERROR] Argument invalid for streams."
        if 'sync' in infer_type and 'async' not in infer_type:
            model = model_function(given_xml, given_bin, 1)
        elif 'async' in infer_type:
            model = model_function(given_xml, given_bin, streams)

    elif model_function is None and model_xml == '' and model_bin == '':
        logger.swap(
            'Custom bin, xml, and model init not found. Using default model init with given model.')
        assert (given_model_name in list(model_paths.paths_dict.keys())),\
            '[ERROR] Argument for given model name does not match any name in Open Seismic.'
        model_func_file_path = model_paths.demo_paths_for_imports[given_model_name] + '.model'
        given_xml, given_bin = model_paths.paths_dict[given_model_name]
        assert (os.path.exists(given_xml)
                ), f"[ERROR] given xml path: {given_xml} invalid"
        assert (os.path.exists(given_bin)
                ), f"[ERROR] given bin path: {given_bin} invalid"

        model_func_file = importlib.import_module(model_func_file_path)
        model_func = model_func_file.Model

        assert (
            'sync' in infer_type), "[ERROR] Argument invalid for infer type."
        assert (streams > 0), "[ERROR] Argument invalid for streams."
        if 'sync' in infer_type and 'async' not in infer_type:
            model = model_func(given_xml, given_bin, 1)
        elif 'async' in infer_type:
            model = model_func(given_xml, given_bin, streams)

    def log_subprocess_output(pipe):
        for line in iter(pipe.readline, b''):
            # print('Benchmark output - ' + str(line)[2:-3])
            logging.log(7, 'Benchmark output - ' + str(line)[2:-3])

    
    if is_benchmarking:
        logger.setLevel(INFER)
        logger.info('Start benchmarking')
        path_to_xml = model_xml if model_xml != '' else given_xml
        params = ["-m", path_to_xml, "-d", "CPU", "--api", infer_type]
        openvino_dir = os.getenv('INTEL_OPENVINO_DIR')
        try:
            process = subprocess.Popen([sys.executable, openvino_dir + '/deployment_tools/tools/benchmark_tool/benchmark_app.py', *params],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            with process.stdout:
                log_subprocess_output(process.stdout)
            with process.stderr:
                log_subprocess_output(process.stderr)
                raise Exception

        except Exception:
            logging.info('Error, benchmarking unsuccessful')

    return preprocess, postprocess, model


if __name__ == '__main__':
    """
    This code sets up the inference task, and based on the arguments given,
    it will execute the task specific by the user in their configuration json
    file. Current inference tasks include:
        (1) Regular sync/async: This inference task is for more general purpose
            inference. 
        (2) Fine Cube sync/async: This inference task is designed for salt 
            identification tasks or tasks which involve taking in a mini cube 
            of a larger input and outputting a unit volume to store in a larger
            cube. There is an option to interpolate the storage cube to the
            size of the input.
        (3) Coarse Cube sync: This inference task is designed for fault 
            segmentation or tasks which involve taking in a mini cube of a 
            larger input and outputting a mini cube of the same size to be
            stored in a larger storage cube. 
        (4) Section sync/async: this inference task is designed for facies
            classification or tasks which involve conducting inference over
            sections of a larger sheet.
    """
    arg_obj = argparse_util.args()
    sep = os.path.sep
    arg_obj.output =  log_folder + sep + arg_obj.output
    get_functions = partial(get_functions, arg_obj.benchmarking)

    logger.setLevel(ARG)
    logger.arg("model={} data={} output={}".format(
        arg_obj.model, arg_obj.data, arg_obj.output
    ))

    assert len(arg_obj.data) > 0, "[ERROR] Data folder not specified."
    assert os.path.exists(arg_obj.data), "[ERROR] Data folder not found."
    assert (len(arg_obj.model) > 0 or len(arg_obj.given_model_name) > 0),\
        "[ERROR] Given model name or model folder path needs to be specified."

    if arg_obj.infer_type == 'sync':
        infer_regular.infer_sync(arg_obj, logger, get_functions)
    elif arg_obj.infer_type == 'async':
        infer_regular.infer_async(arg_obj, logger, get_functions)
    elif arg_obj.infer_type == 'fine_cube_sync':
        infer_fine_cube.infer_fine_cubed_sync(arg_obj, logger, get_functions)
    elif arg_obj.infer_type == 'fine_cube_async':
        infer_fine_cube.infer_fine_cubed_async(arg_obj, logger, get_functions)
    elif arg_obj.infer_type == "coarse_cube_sync":
        infer_coarse_cube.infer_coarse_cubed_sync(arg_obj, logger, get_functions)
    elif arg_obj.infer_type == "coarse_cube_async":
        print("[ERROR] Not implemented yet.")
    elif arg_obj.infer_type == 'section_sync':
        infer_section.infer_section_sync(arg_obj, logger, get_functions)
    elif arg_obj.infer_type == 'section_async':
        infer_section.infer_section_async(arg_obj, logger, get_functions)
    else:
        assert False, "[ERROR] Invalid argument for infer type."
