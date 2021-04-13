Inference Handler Documentation
===============================

Handling Runs
______________

``handle_run.py``

This code parses the JSON configuration file given by the user and executes commands per the available arguments in the configuration file. Specifically, the code block is concerned with arguments for:
 - Running pre model optimization, which is important when OpenVINO does not support the current model framework.
 - Running OpenVINO model optimization
 - Conducting inference on the data with the specified model/processors.

Functions in ``handle_run.py``:

.. currentmodule:: core.python.handle_run
.. autofunction:: get_params_dict

.. currentmodule:: core.python.handle_run
.. autofunction:: json_args

Handling Inference
__________________

``infer.py``

This code sets up the inference task, and based on the arguments given, it will execute the task specific by the user in their configuration JSON file. Current inference tasks include:
 - Regular sync/async: This inference task is for more general purpose inference.
 - Fine Cube sync/async: This inference task is designed for salt identification tasks or tasks which involve taking in a mini cube of a larger input and outputting a unit volume to store in a larger cube. There is an option to interpolate the storage cube to the size of the input.
 - Coarse Cube sync: This inference task is designed for fault segmentation or tasks which involve taking in a mini cube of a larger input and outputting a mini cube of the same size to be stored in a larger storage cube.
 - Section sync/async: this inference task is designed for facies classification or tasks which involve conducting inference over sections of a larger sheet.

Functions in ``infer.py``:

.. currentmodule:: core.python.infer
.. autofunction:: get_functions



