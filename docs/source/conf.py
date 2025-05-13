import os 
import sys
sys.path.insert(0, os.path.abspath('../../src/pyvale/'))
sys.path.insert(0, os.path.abspath('../../src/pyvale/dic/'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Pyvale'
copyright = '2025, The CAV Team'
author = 'The CAV Team at United Kingdom Atomic Energy Authority (UKAEA)'
release = '2025.4.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'sphinx_codeautolink',
              'sphinx_autodoc_typehints',
              'breathe',
              'myst_parser']

breathe_projects = {"test": "./doxygen/xml"}
breathe_default_project = "test"

language = 'english'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_theme_options = {
    "light_logo": "logo_no_background.png",
    "dark_logo": "logo_no_background_inverted.png",
}
html_title = "The Python Validation Engine"
html_css_files = ["custom.css"]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

autosummary_generate = True
