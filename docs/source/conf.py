# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import toml
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------

project = 'SPECTPSFToolbox'
copyright = '2024, Luke Polson'
author = 'Luke Polson'

# The full version, including alpha/beta/rc tags
with open('../../pyproject.toml', 'r') as f:
    release = toml.load(f)['project']['version']

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    'sphinx.ext.viewcode',
    "sphinx.ext.napoleon",
    "sphinx_design",
    "nbsphinx",
    "autoapi.extension",
    "IPython.sphinxext.ipython_console_highlighting"
]

# Where to autogen API
autoapi_dirs = ['../../src/spectpsftoolbox']
autoapi_options = [ 'members', 'undoc-members', 'private-members', 'show-inheritance', 'show-module-summary', 'special-members', 'imported-members', 'recursive']
autoapi_template_dir = "_templates/autoapi"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
#html_logo = 'images/PT1.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

# typehints
autodoc_typehints = "description"
autodoc_inherit_docstrings=True

# Add link to github
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/lukepolson/SPECTPSFToolbox",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    'fixed_sidebar': True,
    'sidebar_width': '220px',
    
}

linkcheck_anchors = False