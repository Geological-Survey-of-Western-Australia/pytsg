Contributing
============

Developer Installation
----------------------

1. ``pytsg`` uses ``uv`` to manage dependencies and virtual environments. To install ``uv``, follow the
   instructions at https://docs.astral.sh/uv/getting-started/installation/.

2. Install all optional dependencies for development and testing:

    .. code-block:: bash
    
        uv sync --all-extras --all-groups 

Building ``pytsg`` Package
--------------------------

The ``pytsg`` package can be built with the following command:

.. code-block:: bash

   uv build

Building Documentation
----------------------

To build the documentation locally you can use the following command in the doc directory:

.. code-block:: bash

   make html

To build and serve the documentation live, you can use the following command in the doc directory:

.. code-block:: bash

   make livehtml

To check that all links in the documentation are valid, you can use the following command in
the doc directory:

.. code-block:: bash

   make linkcheck

.. toctree::
   :maxdepth: 2
