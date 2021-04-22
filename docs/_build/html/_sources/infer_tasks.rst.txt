Inference Tasks Documentation
===============================
OpenSeismic allows users to not only conduct regular inference on data,
but it also helps users conduct critical inference tasks on seismic data.
Below introduce the functions responsible for conducting inference. Read their
descriptions in order to understand how to specify them in your configuration
JSON file.

Regular Inference
_________________

.. currentmodule:: core.python.infer_regular
.. autofunction:: infer_sync

.. currentmodule:: core.python.infer_regular
.. autofunction:: infer_async

Coarse Cube Inference
_____________________

.. currentmodule:: core.python.infer_coarse_cube
.. autofunction:: infer_coarse_cubed_sync

Fine Cube Inference
___________________

.. currentmodule:: core.python.infer_fine_cube
.. autofunction:: infer_fine_cubed_sync

.. currentmodule:: core.python.infer_fine_cube
.. autofunction:: infer_fine_cubed_async

Section Inference
_________________

.. currentmodule:: core.python.infer_section
.. autofunction:: infer_section_sync

.. currentmodule:: core.python.infer_section
.. autofunction:: infer_section_async