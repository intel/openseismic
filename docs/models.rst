Model and Processors Documentation
==================================
Open Seismic offers public state-of-the-art models for users to try.
Below introduces the classes and their associated processor scripts that
are used to conduct inference on seismic data. Read their
descriptions in order to understand how to specify them in your configuration
JSON file.

FaultSeg
________
The FaultSeg repository can be found here: `FaultSeg repository <https://github.com/xinwucwp/faultSeg>`_.
The OpenVINO IR xml and bin files can be found in ``models/fault/model/``.
In order to specify FaultSeg in your configuration file, set ``given_model_name`` to
``fseg`` under ``inference_params``.

.. automodule:: models.fault.model.model
.. autoclass:: Model
    :members: __init__, infer, reshape_input, get_input_shape, get_input_shape_dict, get_inputs, get_outputs, get_requests, get_idle_request_id

.. automodule:: models.fault.model.preprocessor
.. autofunction:: preprocess

.. automodule:: models.fault.model.postprocessor
.. autofunction:: postprocess

Salt
________
Literature pertaining to the Salt model be found here: `Salt literature link <https://library.seg.org/doi/abs/10.1190/tle37070529.1>`_.
The OpenVINO IR xml and bin files can be found in ``models/salt/model/``.
In order to specify FaultSeg in your configuration file, set ``given_model_name`` to
``salt`` under ``inference_params``.

.. automodule:: models.salt.model.model
.. autoclass:: Model
    :members: __init__, infer, reshape_input, get_inputs, get_outputs, get_requests, get_idle_request_id

.. automodule:: models.salt.model.preprocessor
.. autofunction:: preprocess

.. automodule:: models.salt.model.postprocessor
.. autofunction:: postprocess

Facies
________
The Facies repository can be found here: `Facies repository <https://github.com/yalaudah/facies_classification_benchmark>`_.
The OpenVINO IR xml and bin files can be found in ``models/facies/model/``.
In order to specify FaultSeg in your configuration file, set ``given_model_name`` to
``facies`` under ``inference_params``.

.. automodule:: models.facies.model.model
.. autoclass:: Model
    :members: __init__, infer, reshape_input, get_input_shape, get_inputs, get_outputs, get_requests, get_idle_request_id

.. automodule:: models.facies.model.preprocessor
.. autofunction:: preprocess

.. automodule:: models.facies.model.postprocessor
.. autofunction:: postprocess
