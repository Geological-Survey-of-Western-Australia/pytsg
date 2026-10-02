Contributing
============

Developer Installation
----------------------

1. ``pytsg`` uses ``uv`` to manage dependencies and virtual environments. To install ``uv``, follow the
   instructions at https://docs.astral.sh/uv/getting-started/installation/.

2. Clone the repository and navigate to the root directory of the project.

   .. code-block:: bash

      git clone https://github.com/Geological-Survey-of-Western-Australia/pytsg.git
      cd pytsg

3. Install all optional dependencies for development and testing:

   .. code-block:: bash

      uv sync --all-extras --all-groups

.. note::
   There is a Makefile included in the root directory of this project that provides a convenient way to run common tasks. 

   If you have ``make`` installed you can see the available targets by running the following from the root of this project:

   .. code-block:: bash

      make help

Testing
-------

``pytsg`` uses `pytest <https://docs.pytest.org/>`_ for testing. To run the tests, use the following command:

.. code-block:: bash

   # Run tests using uv:
   uv run pytest -v

   # or using make:
   make test

or you can run the tests with coverage reporting:

.. code-block:: bash

   # Generate test coverage report using uv:
   uv run pytest --cov --cov-report=term --cov-report=html --tb=short --disable-warnings
    
   # or using make:
   make coverage

Code Quality
------------

``pytsg`` uses `ruff <https://docs.astral.sh/ruff/>`_ for code formatting and linting, and `ty <https://docs.astral.sh/ty/>`_ for type checking. 

.. code-block:: bash

   # Check code formatting and linting issues
   uv run ruff check .

   # Fix code linting issues
   uv run ruff check --fix .

   # Format code
   uv run ruff format

   # Check type annotations
   uv run ty check .

``ruff`` can also be integrated with your IDE (e.g. VS Code) to provide real-time feedback on code quality by automatically running on save. Please
refer to the `ruff documentation <https://docs.astral.sh/ruff/editors/>`_ for instructions on how to set this up.

.. code-block:: json
   :caption: Example ``.vscode/settings.json`` for configuring the `ruff extension <https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`_ in VS Code

   {
      "[python]": {
         "editor.defaultFormatter": "charliermarsh.ruff",
         "editor.formatOnSave": true,
         "editor.codeActionsOnSave": {
               "source.fixAll.ruff": "explicit",
               "source.organizeImports.ruff": "explicit"
         }
      },
   }


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
