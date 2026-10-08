# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

from docutils import nodes

sys.path.insert(0, os.path.abspath("../src/pytsg"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pytsg"
copyright = "2022-%Y, Ben Chi, Geoscience Data Integrations Team - GSWA, & contributors"
author = "Ben Chi"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",  # to document the api
    "sphinx.ext.viewcode",  # to add view code links
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",  # for parsing numpy/google docstrings
    "sphinx_gallery.gen_gallery",  # to generate a gallery of examples
    "sphinx_autodoc_typehints",
    "myst_parser",  # for parsing md files
    "sphinx.ext.autosectionlabel",  # enables links to sections
    "sphinx_llm.txt",  # Generate llms.txt and .md for LLMs and agents to consume
]

# Suppress Markdown-builder warnings for all unknown node types (True)
# or for an exact, case-sensitive sequence of node class names such
# as ["caption", "desc_inline"].
llms_txt_suppress_unknown_node_warnings = True

autosectionlabel_prefix_document = True

autosummary_generate = True

sphinx_gallery_conf = {
    "filename_pattern": r"\.py",
    "ignore_pattern": r"__init__\.py",
    "examples_dirs": "../../examples",  # path to your example scripts
    "gallery_dirs": "examples",  # path to where to save gallery generated output
    "write_computation_times": False,
}

templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "_templates"]


rst_epilog = """
.. |br| raw:: html

   <br />
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["css/dmpe.css"]

# -- llms.txt improvements --------------------------------------------------------


def _strip_gallery_download_note(app, doctree, docname):
    """Drop Sphinx-Gallery's 'Go to the end to download' note from the llms.txt
    markdown build so each example's real title is used in llms.txt."""
    if app.builder.name != "llms-markdown":
        return  # keep the note on the normal HTML site
    for node in list(doctree.findall(nodes.note)):
        if "sphx-glr-download-link-note" in node.get("classes", []):
            node.parent.remove(node)


def setup(app):
    app.connect("doctree-resolved", _strip_gallery_download_note)
