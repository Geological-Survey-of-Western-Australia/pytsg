Quick Start
===========

Here are the basics to get you started. For more detailed examples, take a look at the :doc:`examples gallery<../examples/index>`.

TSG File Structure
------------------

:term:`The Spectral Geologist` software can be used to import and process hyperspectral data from a variety of sources including
the :term:`HyLogger` hyperspectral logging system, PIMA, ASD, ENVI spectral libraries, etc. While HyLogger datasets available from
Australia's :term:`National Virtual Core Library` will contain spectra, :term:`scalars <scalar>`, :term:`LiDAR` data, and 
:term:`imagery <CRAS>` other datasets may only contain the spectra and :term:`scalars <scalar>`. 

The :term:`HyLogger` is a hyperspectral mineralogical drill core scanner that captures hyperspectral data in
multiple wavelength ranges, RGB imagery, and profilemetry data (LiDAR). The TSG data package stores the spectra, 
and derived scalars, for each spectral range separately. For a HyLogger 4 dataset called "HOLENAME" which contains
NIR/MIR/TIR spectra we would expect the following files to be present:

.. code-block:: console

   HOLENAME/
   ├── HOLENAME_tsg.bip
   ├── HOLENAME_tsg.tsg
   ├── HOLENAME_tsg_cras.bip
   ├── HOLENAME_tsg_hires.dat
   ├── HOLENAME_tsg_mir.bip
   ├── HOLENAME_tsg_mir.tsg
   ├── HOLENAME_tsg_tir.bip
   └── HOLENAME_tsg_tir.tsg

.. table::
   :widths: 30 70

   +------------------------+-------------------------------------------------------------------------------------------------------------------------+
   |File                    | Description                                                                                                             |
   +========================+=========================================================================================================================+
   | HOLENAME_tsg.bip       | A binary file containing the spectra and scalars for the NIR data in band interleaved by pixel (BIP) format.            |
   +------------------------+-------------------------------------------------------------------------------------------------------------------------+
   | HOLENAME_tsg.tsg       | Contains a description of the NIR related data in the associated `*_tsg.bip`` file.                                     |
   +------------------------+-------------------------------------------------------------------------------------------------------------------------+
   | HOLENAME_tsg_cras.bip  | A compressed raster (:term:`CRAS`) file containing the core imagery which can be co-registered with the spectra.        |
   +------------------------+-------------------------------------------------------------------------------------------------------------------------+
   | HOLENAME_tsg_hires.dat | LiDAR/Profilimetry data. The LiDAR is usually higher resolution than the spectral samples. e.g. 128 points per spectra. |
   +------------------------+-------------------------------------------------------------------------------------------------------------------------+
   | HOLENAME_tsg_mir.bip   | A binary file containing the spectra and scalars for the MIR data in band interleaved by pixel (BIP) format.            |
   +------------------------+-------------------------------------------------------------------------------------------------------------------------+
   | HOLENAME_tsg_mir.tsg   | Contains a description of the MIR related data in the associated `*_tsg_mir.bip`` file.                                 |
   +------------------------+-------------------------------------------------------------------------------------------------------------------------+
   | HOLENAME_tsg_tir.bip   | A binary file containing the spectra and scalars for the TIR data in band interleaved by pixel (BIP) format.            |
   +------------------------+-------------------------------------------------------------------------------------------------------------------------+
   | HOLENAME_tsg_tir.tsg   | Contains a description of the TIR related data in the associated `*_tsg_tir.bip`` file.                                 |
   +------------------------+-------------------------------------------------------------------------------------------------------------------------+

Load a TSG Dataset
------------------

Loading a TSG package is as simple as calling the :func:`pytsg.read_tsg` function with the path to the TSG dataset directory. The returned object is a
:class:`pytsg.TSG` object which contains the spectra, :term:`scalars <scalar>` for each spectral range, :term:`LiDAR` data, and the :term:`imagery <CRAS>`.

.. code-block:: python

   >>> import pytsg
   >>> ds = pytsg.read_tsg("example_data/PE257D")
   >>> ds
   TSG(
      available=('nir', 'tir', 'mir', 'lidar'),
      nir: Spectra(spectra_shape=(625, 531), spectra_dtype=float32, wavelength_shape=(531,), sampleheaders_shape=(625, 7), scalars_shape=(625, 123)),
      tir: Spectra(spectra_shape=(625, 387), spectra_dtype=float32, wavelength_shape=(387,), sampleheaders_shape=(625, 7), scalars_shape=(625, 123)),
      mir: Spectra(spectra_shape=(625, 559), spectra_dtype=float32, wavelength_shape=(559,), sampleheaders_shape=(625, 7), scalars_shape=(625, 93)),
      cras: absent,
      lidar: ndarray(shape=(625,), dtype=float32),
   )

By default :func:`pytsg.read_tsg` won't load the :term:`CRAS` file, but you can set the `include_cras` argument to `True` to load it as well.

.. code-block:: python

   >>> import pytsg
   >>> ds = pytsg.read_tsg("example_data/PE257D", include_cras=True)
   >>> ds
   TSG(
      available=('nir', 'tir', 'mir', 'cras', 'lidar'),
      nir: Spectra(spectrum_name='nir', spectra_shape=(625, 531), spectra_dtype=float32, wavelength_shape=(531,), sampleheaders_shape=(625, 7), scalars_shape=(625, 123)),
      tir: Spectra(spectrum_name='tir', spectra_shape=(625, 387), spectra_dtype=float32, wavelength_shape=(387,), sampleheaders_shape=(625, 7), scalars_shape=(625, 123)),
      mir: Spectra(spectrum_name='mir', spectra_shape=(625, 559), spectra_dtype=float32, wavelength_shape=(559,), sampleheaders_shape=(625, 7), scalars_shape=(625, 93)),
      cras: Cras(image_shape=(86250, 1926, 3), image_dtype=uint8, tray_count=1, section_count=5),
      lidar: ndarray(shape=(625,), dtype=float32),
   )

Load the spectra directly
-------------------------

You can load the :class:`Spectra <pytsg.Spectra>` directly from the `.bip` file using :func:`pytsg.read_spectra`.

.. code-block:: python

   >>> import pytsg
   >>> tir_spectra = spec = pytsg.read_spectra("example_data/PE257D/PE257D_0001_tsg_tir.tsg", "example_data/PE257D/PE257D_0001_tsg_tir.bip", spectrum_name = "tir")
   >>> tir_spectra
   Spectra(spectrum_name='tir', spectra_shape=(625, 387), spectra_dtype=float32, wavelength_shape=(387,), sampleheaders_shape=(625, 7), scalars_shape=(625, 123))

Load the CRAS imagery directly
------------------------------

You can load just the core imagery (:class:`Cras <pytsg.Cras>`) of a :term:`HyLogger` dataset's `_cras.bip` file using :func:`pytsg.read_cras`.

.. code-block:: python

   >>> import pytsg
   >>> imagery = pytsg.read_cras("example_data/PE257D/PE257D_0001_tsg_cras.bip")
   >>> imagery
   Cras(image_shape=(86250, 1926, 3), image_dtype=uint8, tray_count=1, section_count=5)


Load the LiDAR data directly
----------------------------

You can load the :term:`LiDAR` data as a numpy array from a :term:`HyLogger` dataset's `_hires.dat` file using :func:`pytsg.read_lidar`.

.. code-block:: python

   >>> import pytsg
   >>> # Return the mean value per sample
   >>> lidar = pytsg.read_lidar("example_data/PE257D/PE257D_0001_tsg_hires.dat")
   >>> lidar.shape
   (625,)
   >>> # Return the full LiDAR profile per sample
   >>> lidar = pytsg.read_lidar("example_data/PE257D/PE257D_0001_tsg_hires.dat", per_spectra=False)
   >>> lidar.shape
   (80000,)

.. toctree::
   :maxdepth: 2
