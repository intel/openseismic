Utilities Documentation
==================================
Open Seismic uses some utilities functions in order to conduct proper inference.
The documentation below goes in-depth as to the purpose and usage of the major
utility functions.

Inference Argument Parsing and Copying Files
________________________________________________

.. automodule:: core.python.utils.argparse_util
.. autofunction:: args

.. automodule:: core.python.utils.argparse_util
.. autofunction:: copyfile

Inference Utility Functions and Classes
___________________________________________

.. automodule:: core.python.utils.infer_util
.. autoclass:: OV_simulator
    :members: __init__, infer, reshape_input, get_input_shape, get_input_shape_dict, get_inputs, get_outputs, get_requests, get_idle_request_id

.. automodule:: core.python.utils.infer_util
.. autoclass:: InferReqWrap
    :members: __init__, callback, start_async, get_preds, infer

.. automodule:: core.python.utils.infer_util
.. autoclass:: InferRequestsQueue
    :members: __init__, reset_times, get_duration_in_seconds, put_idle_request, get_idle_request, wait_all

.. automodule:: core.python.utils.infer_util
.. autofunction:: loader

.. automodule:: core.python.utils.infer_util
.. autofunction:: segy_to_np

JSON to Command Line Arguments
_____________________________________

.. automodule:: core.python.utils.jsonparse_util
.. autofunction:: get_params_string

Model Paths Dictionary
_________________________
This ``model_paths.py`` file in the ``core/python/utils`` folder maps Open Seismic model names
to a tuple containing (1) the xml path and (2) the bin path. The dictionary is structured
as follows:

.. literalinclude:: ../core/python/utils/model_paths.py
    :language: python
    :lines: 16-20

