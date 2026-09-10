API Reference
=============

The core functionality is exposed from the top-level ``pytsg`` namespace.

Simple API
----------

.. autofunction:: pytsg.read_tsg

.. autofunction:: pytsg.read_spectra

.. autofunction:: pytsg.read_cras

.. autofunction:: pytsg.read_lidar

Data model
----------

.. automodule:: pytsg.models
   :members:

Deprecated API
--------------

The legacy package reader and direct-reader names remain supported. New code
should use the top-level readers above.

.. autofunction:: pytsg.parse_tsg.read_package

.. autofunction:: pytsg.parse_tsg.read_tsg_bip_pair

.. autofunction:: pytsg.parse_tsg.read_hires_dat
