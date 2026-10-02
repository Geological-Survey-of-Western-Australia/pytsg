Glossary
========

.. glossary::
    :sorted:

    CRAS
        A compressed raster (:term:`CRAS`) file is used by :term:`TSG` to store RGB photography of the core which can be co-registered with the spectra. 

    HyLogger
        The HyLogger is a hyperspectral logging system developed by CSIRO, and now sold by Epiroc, for the mineralogical analysis of drill core and
        other geological materials. The HyLogger is a hyperspectral mineralogical drill core scanner that captures hyperspectral data in
        multiple wavelength ranges, RGB imagery, and profilemetry data (LiDAR).

    Joint Constrained Least Squares
    jCLST
        The jCLST unmixing algorithm, included in The Spectral Geologist software package, uses the HyLogger thermal infrared spectra, with the shortwave
        infrared spectra, to provide an estimate of modal mineralogy as described in `Green et al. (2025) <https://doi.org/10.1080/08120099.2025.2464839>`_

    LiDAR
    Profilimetry
        The LiDAR data captured by the :term:`HyLogger` is used to provide a high resolution profile of the drill core.

    National Virtual Core Library
    NVCL
        The National Virtual Core Library (NVCL) is a national centralised hyperspectral core database, allowing for state and
        territory geological surveys to digitise, catalogue and virtually publish their drill core collections.

    Scalar
        TSG uses the term "scalar" to refer to a set of imported or calculated values associated with the loaded 
        spectral data. This may be calculated from the spectra (e.g. TSA classification, 2200D, etc.) or from imported
        data (e.g. core depth, lithology, etc.).

    Spectral Reference Library
    SRL
        A Spectral Reference Library (SRL) is a collection of spectral data for minerals, rocks and other materials.

    The Spectral Assistant
    TSA
        The Spectral Assistant (TSA) is a module of TSG developed by CSIRO used to automatically assess the likely mineral contributors to a spectrum using 
        spectral unmixing. The TSA weights (e.g. "Wt1 uTSAS") indicate the relative contribution of each mineral to the unknown spectrum based on the spectra
        available in the :term:`SRL` during unmixing, **and should not be interpreted as mineral abundances.**

    The Spectral Geologist
    TSG
        The Spectral Geologist (TSG) is the industry standard tool for the mineralogical analysis of
        VIS/NIR/SWIR/MIR and TIR reflectance spectra.  https://research.csiro.au/thespectralgeologist/

.. toctree::
   :maxdepth: 2