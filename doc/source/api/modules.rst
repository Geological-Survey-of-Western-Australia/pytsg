API Reference
=============

The core functionality of :mod:`pytsg` is exposed from the top-level namespace. In most instances the
:func:`pytsg.read_tsg` function will be what users will want to use as it exposes all data available
within a TSG package. There are also functions which allow for direct reading of spectra
(:func:`pytsg.read_spectra`), imagery (CRAS) (:func:`pytsg.read_cras`), and LiDAR (:func:`pytsg.read_lidar`).



Basic API
----------

.. automodule:: pytsg
   :members:
   :member-order: bysource

Advanced API
------------

More advanced functions for working with TSG datasets.

:mod:`pytsg.feature`
~~~~~~~~~~~~~~~~~~~~

.. automodule:: pytsg.feature
   :members:

:mod:`pytsg.readers.cras`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pytsg.readers.cras
   :members:

:mod:`pytsg.readers.lidar`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pytsg.readers.lidar
   :members:

:mod:`pytsg.readers.package`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pytsg.readers.package
   :members:
   :private-members:

:mod:`pytsg.readers.spectra`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pytsg.readers.spectra
   :members:

:mod:`pytsg.models`
~~~~~~~~~~~~~~~~~~~

.. automodule:: pytsg.models
   :members:

Deprecated API
--------------

.. note:: The legacy package reader and direct-reader names in the :mod:`pytsg.parse_tsg` module remain supported however new code
         should use the top-level readers described above as the deprecated functions will be removed in a future version.

.. py:function:: pytsg.parse_tsg.read_package(foldername: Union[str, Path], read_cras_file: bool = False, extract_cras: bool = False, imageoutput: Union[str, Path, None] = None, backing_file: Union[Path, str, None] = None)

   **Deprecated:** Use :func:`pytsg.read_tsg` instead.

.. py:function:: pytsg.parse_tsg.read_tsg_bip_pair(tsg_file: Union[Path, str], bip_file: Union[Path, str], spectrum: str)

   **Deprecated:** Use :func:`pytsg.read_spectra` instead.

.. py:function:: pytsg.parse_tsg.read_hires_dat(path: Union[str, Path], *, per_spectra: bool = True)

   **Deprecated:** Use :func:`pytsg.read_lidar` instead.

.. py:function:: pytsg.parse_tsg.read_cras(filename: Union[str, Path], backing_file: Union[str, Path, None] = None)

   **Deprecated:** Use :func:`pytsg.read_cras` instead.

.. py:function:: pytsg.parse_tsg.composite_spectra(spectra: Spectra, length: int = 4)

   **Deprecated:** Use :func:`pytsg.readers.cras.composite_spectra` instead.

.. py:function:: pytsg.parse_tsg.extract_chips(filename: Union[str, Path], outfolder: Union[str, Path], spectra: Spectra, centre_cut: bool = True)

   **Deprecated:** Use :func:`pytsg.readers.cras.extract_chips` instead.

.. py:function:: pytsg.parse_tsg.generate_chips(filename: Union[str, Path], spectra: Spectra, centre_cut: bool = True, batch_size: int = 256)

   **Deprecated:** Use :func:`pytsg.readers.cras.generate_chips` instead.
